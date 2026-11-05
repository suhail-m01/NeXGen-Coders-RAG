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

            if not text:
                continue
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
