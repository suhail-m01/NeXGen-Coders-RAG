"""
documents.py -- local extraction of PDF / DOC / DOCX / TXT / CSV into
ContentItem objects.

    PDF   PyMuPDF text per page, OCR fallback for scanned pages, page_number kept
    DOCX  python-docx paragraphs + tables, section kept (never a page number)
    DOC   LibreOffice headless -> DOCX -> the DOCX path above
    TXT   read and chunk
    CSV   pandas, one searchable row representation per row, row_index kept

Nothing here touches the network.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import settings
from ocr import OcrResult
from schemas import (
    ContentItem,
    ErrorCode,
    IngestionResult,
    Modality,
    Relationship,
    RelationshipType,
    Source,
    SourceLocation,
    SourceType,
    content_hash,
    derive_id,
)

logger = logging.getLogger(__name__)

EXTENSION_TO_TYPE = {
    ".pdf": SourceType.PDF,
    ".doc": SourceType.DOC,
    ".docx": SourceType.DOCX,
    ".txt": SourceType.TXT,
    ".md": SourceType.TXT,
    ".csv": SourceType.CSV,
}

# A page yielding fewer characters than this is treated as scanned and
# routed through OCR instead of being indexed as an empty page.
MIN_PAGE_CHARS_BEFORE_OCR = 80


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_blocks(
    text: str,
    base_heading: tuple[str, ...] | str = (),
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list:
    """Split text into Chunk objects, honouring settings.chunking_mode.

    Callers use this rather than chunk_text when they want the heading path each
    chunk was found under, which becomes the citation's section. In "fixed" mode
    the chunks come back with an empty path, so the call sites do not branch.
    """
    if settings.chunking_mode.strip().lower() == "structure":
        from chunking import split_document

        return split_document(
            text, max_chars=chunk_size, overlap=overlap, base_heading=base_heading
        )

    from chunking import Chunk

    return [
        Chunk(piece, tuple([base_heading] if isinstance(base_heading, str)
                           else base_heading))
        for piece in chunk_text(text, chunk_size, overlap)
    ]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into strings, honouring settings.chunking_mode."""
    if settings.chunking_mode.strip().lower() == "structure":
        from chunking import chunk_strings

        return chunk_strings(text, max_chars=chunk_size, overlap=overlap)
    return _chunk_fixed(text, chunk_size, overlap)


def _chunk_fixed(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """The character-window splitter: split on a window with overlap, preferring
    to break at a sentence or whitespace boundary so chunks stay readable as
    citations. Kept because it is the fallback for text with no structure at all
    and because a chunking change should be reversible from .env."""
    size = chunk_size or settings.chunk_size
    over = overlap if overlap is not None else settings.chunk_overlap
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer a sentence end, then any whitespace, in the last 30%.
            pivot = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if pivot < size * 0.7:
                pivot = window.rfind(" ")
            if pivot > size * 0.5:
                end = start + pivot + 1
        chunk = text[start:end].strip()
        if len(chunk) >= settings.min_chunk_chars or not chunks:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - over, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def resolve_source_type(path: Path) -> SourceType:
    return EXTENSION_TO_TYPE.get(path.suffix.lower(), SourceType.UNKNOWN)


def _validate(path_str: str) -> tuple[Path, SourceType]:
    path = Path(path_str).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{ErrorCode.FILE_NOT_FOUND.value}: {path_str}")
    if path.stat().st_size > settings.max_file_bytes:
        raise ValueError(f"{ErrorCode.FILE_TOO_LARGE.value}: {path.name}")
    source_type = resolve_source_type(path)
    if source_type is SourceType.UNKNOWN:
        raise ValueError(f"{ErrorCode.UNSUPPORTED_FILE_TYPE.value}: {path.suffix}")
    return path, source_type


def _build_source(path: Path, source_type: SourceType) -> Source:
    data = path.read_bytes()
    return Source(
        source_id=content_hash(data),
        filename=path.name,
        source_type=source_type,
        file_path=str(path),
        mime_type=_mime_for(source_type),
        file_size=len(data),
    )


def _mime_for(source_type: SourceType) -> str:
    return {
        SourceType.PDF: "application/pdf",
        SourceType.DOC: "application/msword",
        SourceType.DOCX: (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        SourceType.TXT: "text/plain",
        SourceType.CSV: "text/csv",
    }.get(source_type, "application/octet-stream")


# ---------------------------------------------------------------------------
# Per-format extraction -> list of (text, SourceLocation, extra_metadata)
# ---------------------------------------------------------------------------

Extracted = tuple[str, SourceLocation, dict]


# A caption is paired with a table by its number when the page numbers its tables,
# and by position when it has exactly one of each. Anything else gets a plain
# "Table N" section rather than a caption guessed from a neighbouring paragraph:
# a wrong caption is worse than none, because it is the text a reader cites.
_TABLE_CAPTION_RE = re.compile(
    r"(?:Table|Tab\.|Exhibit)\s*(\d+)\s*[:.\-\u2014]\s*([^\n]+)",
    re.IGNORECASE,
)


def _table_captions(page_text: str) -> dict[int, str]:
    captions: dict[int, str] = {}
    for match in _TABLE_CAPTION_RE.finditer(page_text or ""):
        try:
            number = int(match.group(1))
        except (TypeError, ValueError):
            continue
        body = " ".join(match.group(2).split())
        if body:
            captions[number] = f"Table {number}: {body}"
    return captions


def _prune_grid(cells) -> list[list[str]]:
    """Flatten every cell to one line and drop rows and columns that are empty.

    Dropping an all-empty column loses nothing -- there is nothing in it -- but
    keeping it costs a pipe in every row and tells the model the table has a field
    it does not. The demo deck's literature survey is six columns wide with one
    column blank all the way down, which is exactly the artefact a layout heuristic
    produces when two ruled columns sit close together.
    """
    grid = [
        [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
        for row in cells or []
    ]
    if not grid:
        return []
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    kept = [index for index in range(width) if any(row[index] for row in grid)]
    if not kept:
        return []
    grid = [[row[index] for index in kept] for row in grid]
    return [row for row in grid if any(row)]


def _grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a pruned grid as GitHub-flavoured markdown. Row 0 is the header."""
    if not grid or not grid[0]:
        return ""
    lines = ["| " + " | ".join(cell or " " for cell in grid[0]) + " |",
             "|" + "|".join(["---"] * len(grid[0])) + "|"]
    lines.extend("| " + " | ".join(cell or " " for cell in row) + " |" for row in grid[1:])
    return "\n".join(lines)


def _table_bbox(found) -> tuple[float, float, float, float] | None:
    raw = getattr(found, "bbox", None)
    if raw is None:
        return None
    try:
        coords = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if len(coords) != 4:
        return None
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _cell_rects(found) -> list[list]:
    """One row of cell rectangles per row, `None` where a cell does not exist.

    Read from `table.rows[i].cells[j]` rather than from the flat `table.cells`
    tuple, which on this corpus is shorter than rows x columns and so cannot be
    reshaped reliably -- merged and sliver cells are simply absent from it.
    """
    rows = []
    for row in getattr(found, "rows", []) or []:
        rects = []
        for cell in getattr(row, "cells", []) or []:
            if cell is None:
                rects.append(None)
                continue
            try:
                coords = tuple(float(value) for value in cell)
            except (TypeError, ValueError):
                rects.append(None)
                continue
            rects.append(coords if len(coords) == 4 else None)
        rows.append(rects)
    return rows


def _centre(word) -> tuple[float, float]:
    return ((word[0] + word[2]) / 2.0, (word[1] + word[3]) / 2.0)


def _claim(words, taken: list[bool], box) -> list[int]:
    """The indices of the words whose centre falls in `box` and are not yet claimed.

    Claiming is what makes the split exact. A word belongs to the first cell that
    contains it and to nothing else, so prose and tables together hold every word on
    the page exactly once -- no duplication where two ruled cells overlap, and no
    silence where a note sits inside the table's box but outside every cell. That
    second case is not hypothetical: the demo deck draws a "RESEARCH GAP" callout
    inside the ruled area on page 5, and an area-overlap rule swallowed it.
    """
    x0, y0, x1, y1 = box
    claimed = []
    for index, word in enumerate(words):
        if taken[index]:
            continue
        cx, cy = _centre(word)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            taken[index] = True
            claimed.append(index)
    return claimed


def _words_to_text(words, indices) -> str:
    """Rebuild reading-order text from word tuples, keeping line breaks."""
    lines: dict[tuple[int, int], list] = {}
    for index in indices:
        word = words[index]
        lines.setdefault((word[5], word[6]), []).append((word[7], word[4]))
    return "\n".join(
        " ".join(text for _, text in sorted(group))
        for _, group in sorted(lines.items())
    )


def _release(taken: list[bool], indices) -> None:
    """Hand claimed words back to the prose.

    Called whenever a candidate region is rejected -- too small to be a table, or
    unreadable -- because a word claimed by a cell that never became an item would
    otherwise vanish from the page: not in the prose, not in any grid. Losing text
    quietly is the one failure an extraction change must not have.
    """
    for index in indices:
        taken[index] = False


def _prose_without(words, taken: list[bool]) -> str:
    """The page's text with every word claimed by a table cell left out."""
    return _words_to_text(words, [i for i in range(len(words)) if not taken[i]]).strip()


def _page_tables(page, words, page_index: int, warnings: list[str]):
    """Detect the tables on one page. Returns (table dicts, claimed-word mask).

    Cell text is rebuilt from the page's own words rather than taken from
    `table.extract()`, and this is the difference between a table that searches and
    one that does not. Extraction inside a ruled cell concatenates the spans it
    finds without their spaces: the same cell reads "Combinesdense retrieval
    withsequence generation" through extract() and "Combines dense retrieval with
    sequence generation" through the word list. The second is what BM25 and the
    embedding model were trained to read, and the first silently detaches every
    fused pair from the query that would have found it.

    Every failure degrades to the page as it was indexed before this existed: a
    detector that raises, a grid that will not read, or a region too small to be a
    table all leave the prose untouched and say so in the warnings list.
    """
    taken = [False] * len(words)
    try:
        finder = page.find_tables()
    except Exception as exc:  # pragma: no cover -- depends on the PDF's internals
        warnings.append(
            f"page {page_index}: table detection failed "
            f"({type(exc).__name__}: {exc}); indexed as plain text"
        )
        return [], taken

    found_tables = list(getattr(finder, "tables", []) or [])
    if not found_tables:
        return [], taken

    captions = _table_captions(page.get_text("text") or "")
    tables: list[dict] = []

    for index, found in enumerate(found_tables, start=1):
        claimed: list[int] = []
        try:
            grid = []
            for row_rects in _cell_rects(found):
                row = []
                for box in row_rects:
                    if box is None:
                        row.append("")
                        continue
                    indices = _claim(words, taken, box)
                    claimed.extend(indices)
                    row.append(_words_to_text(words, indices))
                grid.append(row)
            grid = _prune_grid(grid)
        except Exception as exc:  # pragma: no cover -- depends on the PDF's internals
            _release(taken, claimed)
            warnings.append(
                f"page {page_index}: table {index} could not be read "
                f"({type(exc).__name__}: {exc}); indexed as plain text"
            )
            continue

        filled = sum(1 for row in grid for cell in row if cell) if grid else 0
        markdown = _grid_to_markdown(grid) if grid else ""
        if (not grid or not markdown
                or len(grid) < settings.pdf_table_min_rows
                or len(grid[0]) < settings.pdf_table_min_cols
                or filled < settings.pdf_table_min_cells):
            # Too small to be a table, so it stays in the prose, where it reads
            # better than it would as a two-cell grid. Its words go back first.
            _release(taken, claimed)
            continue

        caption = captions.get(index)
        if caption is None and len(found_tables) == 1 and len(captions) == 1:
            caption = next(iter(captions.values()))

        tables.append({
            "text": f"{caption}\n\n{markdown}" if caption else markdown,
            "section": caption or f"Table {index}",
            "caption": caption,
            "index": index,
            "rows": len(grid),
            "cols": len(grid[0]),
            "cells": filled,
        })

    if not tables:
        # Nothing was accepted, so nothing may be withheld from the prose.
        return [], [False] * len(words)
    return tables, taken


def _extract_pdf(path: Path, warnings: list[str]) -> list[Extracted]:
    import pymupdf

    out: list[Extracted] = []
    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            ocr_used = False
            ocr_confidence: float | None = None

            if len(text) < MIN_PAGE_CHARS_BEFORE_OCR:
                ocr = _ocr_pdf_page(page, warnings, page_index)
                if len(ocr.text) > len(text):
                    from ocr import annotate

                    text = annotate(ocr.text, ocr)
                    ocr_used = True
                    ocr_confidence = round(ocr.confidence, 1)

            # Tables are lifted out after the OCR decision, which has to see the
            # whole page: a page that is mostly a scanned table must not look
            # text-rich just because the grid is about to be removed from it.
            page_tables: list[dict] = []
            if settings.pdf_table_extraction_enabled:
                words = page.get_text("words") or []
                page_tables, taken = _page_tables(page, words, page_index, warnings)
                if page_tables and not ocr_used:
                    text = _prose_without(words, taken)

            if text:
                for chunk in chunk_blocks(text):
                    out.append((
                        chunk.text,
                        SourceLocation(
                            page_number=page_index,
                            section=chunk.section or None,
                        ),
                        {
                            "ocr_used": ocr_used,
                            "ocr_confidence": ocr_confidence,
                            **chunk.metadata(),
                        },
                    ))

            for table in page_tables:
                out.append((
                    table["text"],
                    SourceLocation(page_number=page_index, section=table["section"]),
                    {
                        "is_table": True,
                        "table_index": table["index"],
                        "table_rows": table["rows"],
                        "table_cols": table["cols"],
                        "table_cells": table["cells"],
                        "table_caption": table["caption"],
                    },
                ))
    return out


def _ocr_pdf_page(page, warnings: list[str], page_index: int) -> OcrResult:
    """Rasterise one PDF page and OCR it. Returns an empty result on any
    failure -- a scanned page we cannot read is skipped, never fabricated."""
    try:
        import io

        from PIL import Image

        from ocr import ocr_with_confidence, warning_for

        pixmap = page.get_pixmap(dpi=200)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        result = ocr_with_confidence(image)
        if result.unreliable:
            warnings.append(f"Page {page_index}: {warning_for(result)}")
        return result
    except Exception as exc:
        warnings.append(f"OCR failed on page {page_index}: {exc}")
        logger.warning("PDF OCR failed on page %s: %s", page_index, exc)
        return OcrResult()


def _extract_docx(path: Path, warnings: list[str]) -> list[Extracted]:
    """python-docx cannot know page boundaries -- pagination is computed by
    the renderer, not stored in the file -- so DOCX items carry a section
    heading and never a page_number."""
    import docx

    document = docx.Document(str(path))
    out: list[Extracted] = []
    current_section = "Document body"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        # Blank-line separated, not space separated: python-docx hands over one
        # paragraph per entry, and the splitter needs those boundaries to keep a
        # paragraph whole instead of packing by character count.
        joined = "\n\n".join(buffer)
        for chunk in chunk_blocks(joined, base_heading=current_section):
            out.append((
                chunk.text,
                SourceLocation(section=chunk.section or current_section),
                dict(chunk.metadata()),
            ))
        buffer.clear()

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style is not None and para.style.name.lower().startswith("heading"):
            flush()
            current_section = text
            continue
        buffer.append(text)
    flush()

    for table_index, table in enumerate(document.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue
        # Structure mode gets one row per line so the splitter can see the header
        # row and repeat it onto every chunk of the table: "99.0" on its own is
        # not evidence. Fixed mode keeps the original single-line form, which is
        # what the published baseline measured.
        if settings.chunking_mode.strip().lower() == "structure":
            table_text = "\n".join(rows)
        else:
            table_text = f"Table {table_index}: " + "; ".join(rows)
        for chunk in chunk_blocks(table_text, base_heading=f"Table {table_index}"):
            out.append((
                chunk.text,
                SourceLocation(section=chunk.section or f"Table {table_index}"),
                {"is_table": True, **{k: v for k, v in chunk.metadata().items()
                                      if k != "is_table"}},
            ))
    return out


def _convert_doc_to_docx(path: Path) -> Path:
    """LibreOffice headless conversion. Raises if LibreOffice is absent --
    we never pretend .doc was read."""
    soffice = settings.soffice_cmd
    if not Path(soffice).exists():
        found = shutil.which("soffice")
        if not found:
            raise RuntimeError(
                f"{ErrorCode.CONVERSION_UNAVAILABLE.value}: LibreOffice not found. "
                "Install it to enable legacy .doc support."
            )
        soffice = found

    tmpdir = Path(tempfile.mkdtemp(prefix="evidenceai_doc_"))
    # An isolated user profile lets conversion work even while the user has
    # LibreOffice open; without it the second instance refuses to start.
    # stdin is closed because soffice waits on console input when attached.
    profile = (tmpdir / "profile").as_uri()
    proc = subprocess.run(
        [soffice, "--headless", "--norestore", "--invisible",
         f"-env:UserInstallation={profile}",
         "--convert-to", "docx", "--outdir", str(tmpdir), str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
    )
    converted = tmpdir / (path.stem + ".docx")
    if not converted.exists():
        raise RuntimeError(
            f"{ErrorCode.CONVERSION_UNAVAILABLE.value}: LibreOffice conversion "
            f"produced no output. stderr={proc.stderr[:300]}"
        )
    return converted


def _extract_txt(path: Path, warnings: list[str]) -> list[Extracted]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        (
            chunk.text,
            SourceLocation(section=chunk.section or None),
            dict(chunk.metadata()),
        )
        for chunk in chunk_blocks(text)
    ]


def _extract_csv(path: Path, warnings: list[str]) -> list[Extracted]:
    """Each row becomes a self-describing 'column: value' sentence, which embeds
    far better than a raw comma-separated line -- then rows are packed into
    chunks by the same structure-aware splitter every other format uses.

    Packing matters more than it looks. One item per row means a four-row table
    is four items, and retrieval caps how many items a single source may
    contribute (mmr_max_per_source). With that cap at two, the model can never
    see all four rows, so "which quarter is highest" is unanswerable no matter
    how good the ranking is: the row holding the answer is filtered out as a
    duplicate source. A small table now arrives whole; a large one still splits,
    at the chunk cap instead of at every row.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    columns = [str(c) for c in frame.columns]
    rows: list[tuple[int, str]] = []
    for row_index, row in frame.iterrows():
        parts = [
            f"{col}: {row[col]}"
            for col in columns
            if str(row[col]).strip() not in ("", "nan", "None")
        ]
        if not parts:
            continue
        rows.append((int(row_index), f"Row {int(row_index) + 1} -- " + "; ".join(parts)))

    if not rows:
        return []

    if settings.chunking_mode.strip().lower() != "structure":
        # Fixed mode keeps the original one-item-per-row contract. Row packing is
        # a structure-mode improvement and is not applied here, because the
        # published evaluation baseline (evaluation/baseline.json) was measured
        # against one item per row and evaluation.py --compare fails any build
        # that silently changes it.
        return [
            (text, SourceLocation(section=f"Row {index + 1}"), {"row_index": index})
            for index, text in rows
        ]

    out: list[Extracted] = []
    for chunk in chunk_blocks("\n".join(text for _, text in rows)):
        # Which rows landed in this chunk, so the citation can still say
        # "Rows 1-4" and the metadata keeps the original row indices.
        numbers = [
            index + 1 for index, text in rows
            if f"Row {index + 1} --" in chunk.text
        ]
        if len(numbers) == 1:
            section = f"Row {numbers[0]}"
        elif numbers:
            section = f"Rows {numbers[0]}-{numbers[-1]}"
        else:
            section = chunk.section or "Table"
        out.append((
            chunk.text,
            SourceLocation(section=section),
            {"row_indices": [n - 1 for n in numbers], "row_count": len(numbers),
             **{k: v for k, v in chunk.metadata().items() if k != "is_table"}},
        ))
    return out


EXTRACTORS = {
    SourceType.PDF: _extract_pdf,
    SourceType.DOCX: _extract_docx,
    SourceType.TXT: _extract_txt,
    SourceType.CSV: _extract_csv,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(path_str: str) -> IngestionResult:
    """Turn one document into an IngestionResult. Never raises: failures come
    back as success=False with an error_code the caller can act on."""
    source: Source | None = None
    warnings: list[str] = []
    try:
        path, source_type = _validate(path_str)
        source = _build_source(path, source_type)

        working_path, working_type = path, source_type
        if source_type is SourceType.DOC:
            working_path = _convert_doc_to_docx(path)
            working_type = SourceType.DOCX
            warnings.append("Converted .doc to .docx via LibreOffice headless.")

        extractor = EXTRACTORS.get(working_type)
        if extractor is None:
            raise ValueError(f"{ErrorCode.UNSUPPORTED_FILE_TYPE.value}: {working_type}")

        extracted = extractor(working_path, warnings)
        if settings.chunking_mode.strip().lower() == "structure":
            # Furniture only becomes its own chunk under structure-aware
            # chunking; in fixed mode those lines sit inside page-sized chunks
            # and there is nothing to remove.
            extracted = _dedupe(extracted, warnings)
        if not extracted:
            return IngestionResult(
                source=source,
                items=[],
                success=False,
                error="No extractable text found in document.",
                error_code=ErrorCode.CORRUPT_FILE,
                warnings=warnings,
            )

        items = [
            ContentItem(
                item_id=derive_id(source.source_id, "doc", index),
                source_id=source.source_id,
                modality=Modality.TEXT,
                content=text,
                location=location,
                metadata={
                    "chunk_index": index,
                    "source_type": source_type.value,
                    **extra,
                },
            )
            for index, (text, location, extra) in enumerate(extracted)
        ]
        return IngestionResult(
            source=source,
            items=items,
            relationships=build_document_relationships(items),
            success=True,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("process_document failed for %s: %s", path_str, exc)
        return _failure(path_str, source, exc, warnings)


def _dedupe(extracted: list[Extracted], warnings: list[str]) -> list[Extracted]:
    """Drop repeated page furniture: heading-only chunks whose text already
    appeared in this document.

    Running heads, footers and watermarks are extracted from every page, so one
    banner line becomes one junk item per page. Structure-aware chunking makes
    them visible -- a furniture line is a heading with no body under it, and it
    now surfaces as its own chunk instead of hiding inside a page of prose -- so
    they can be removed precisely. Only chunks flagged `heading_only` are
    candidates: a body paragraph that genuinely repeats (a refrain, a restated
    figure) is evidence and is kept.
    """
    seen: set[str] = set()
    kept: list[Extracted] = []
    dropped = 0
    for text, location, extra in extracted:
        key = " ".join(text.split())
        if extra.get("heading_only") and key and key in seen:
            dropped += 1
            continue
        if key:
            seen.add(key)
        kept.append((text, location, extra))
    if dropped:
        warnings.append(
            f"Dropped {dropped} repeated heading/furniture line(s) identical to "
            "an earlier one (running head, footer or watermark)."
        )
    return kept


def build_document_relationships(items: list[ContentItem]) -> list[Relationship]:
    """SAME_PAGE between chunks sharing a page. SAME_SOURCE is derivable from
    source_id alone and is computed at query time rather than stored O(n^2)."""
    by_page: dict[int, list[ContentItem]] = {}
    for item in items:
        if item.location.page_number is not None:
            by_page.setdefault(item.location.page_number, []).append(item)

    relationships: list[Relationship] = []
    for page, page_items in by_page.items():
        for i in range(len(page_items) - 1):
            a, b = page_items[i], page_items[i + 1]
            relationships.append(Relationship(
                relationship_id=derive_id(a.item_id, b.item_id, "same_page"),
                source_item_id=a.item_id,
                target_item_id=b.item_id,
                relationship_type=RelationshipType.SAME_PAGE,
                confidence=1.0,
                metadata={"page_number": page},
            ))
    return relationships


def _failure(
    path_str: str,
    source: Source | None,
    exc: Exception,
    warnings: list[str],
) -> IngestionResult:
    message = str(exc)
    code = ErrorCode.CORRUPT_FILE
    for candidate in ErrorCode:
        if message.startswith(candidate.value):
            code = candidate
            break

    if source is None:
        path = Path(path_str)
        source = Source(
            source_id=derive_id(path_str),
            filename=path.name or "unknown",
            source_type=resolve_source_type(path),
            file_path=str(path),
        )
    return IngestionResult(
        source=source,
        items=[],
        success=False,
        error=message,
        error_code=code,
        warnings=warnings,
    )
