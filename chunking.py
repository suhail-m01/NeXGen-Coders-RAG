"""
chunking.py -- structure-aware splitting of extracted text into evidence items.

Why not a character window
--------------------------
The fixed splitter normalises whitespace first (`" ".join(text.split())`), which
throws away exactly the information a document is organised by. Headings get
glued into the prose they label, a table row can be cut in half at character
800, and a chunk that starts mid-paragraph carries no clue where it came from.
Retrieval then has to guess the topic of an anonymous slab of text, and the
citation can only say "Page 3".

What this module keeps
----------------------
* Headings -- Markdown (`## x`), numbered (`3.1 x`), ALL-CAPS and Title Case
  lines -- form a heading stack, and every chunk knows the path above it. The
  path lands in `SourceLocation.section`, so a citation reads
  "Page 4 · Risks and Outlook" and the evidence block handed to the model
  carries the same context.
* Paragraphs are built from consecutive prose lines (PDF extraction wraps one
  paragraph over many lines) and are never split unless a single paragraph is
  longer than the cap, in which case it breaks at sentence ends.
* Table rows stay whole, and every chunk of rows repeats the header row: "99.0"
  without "Q3 2024 | accuracy_pct" beside it is not evidence, it is a number.
* `chunk_size` is a cap, not a target. A 240-character section stays a
  240-character chunk instead of being padded out with the next section, which
  is what length-normalised reranking and MMR both assume.

Item content stays verbatim document text. The heading path is metadata, not a
prefix: what the model quotes and what the numeric guard checks should be
exactly what the source says.

Overlap
-------
The fixed splitter needs overlap because it cuts mid-sentence and both halves
lose the seam. Structure-aware chunking cuts at boundaries that already are
seams, so `chunk_overlap` is applied only where a single oversized block had to
be split into pieces.

Configuration
-------------
CHUNKING_MODE=structure (default) or fixed. Changing it changes how every
document is split, so run scripts\\reindex.py afterwards: ingestion is idempotent
by content hash and would otherwise leave the old chunks in place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings

# Small words are ignored when deciding whether a line is a Title Case heading:
# "Budget and Resourcing" is a heading, "the report was filed today" is not.
SMALL_WORDS = {
    "a", "an", "and", "or", "of", "the", "in", "on", "at", "to", "for", "with",
    "by", "from", "is", "was", "are", "were", "as", "into", "over", "under",
}

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SECTION_NUMBER = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?\s+([A-Z(].*)$")
ROW_PREFIX = re.compile(r"^Row \d+\s+--")
TABLE_PREFIX = re.compile(r"^(Table \d+)\s*:\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A heading is a label, not a sentence. Anything longer is prose that happens to
# start with a capital.
MAX_HEADING_CHARS = 90
# A line with this many cell separators is a table row, not prose.
MIN_ROW_SEPARATORS = 2


@dataclass
class Block:
    """One indivisible piece of a document: a heading, a paragraph or a row."""
    text: str
    kind: str                      # "heading" | "prose" | "row"
    level: int = 0                 # heading depth, 0 when not a heading
    header: str | None = None      # table header this row belongs to


@dataclass
class Chunk:
    """A piece of a document that becomes one ContentItem."""
    text: str
    heading_path: tuple[str, ...] = ()
    is_table: bool = False
    index: int = 0
    # True when the chunk is nothing but heading text: a banner, a running head,
    # a slide title with no body under it. Callers use this to drop repeated page
    # furniture without ever discarding real body text.
    heading_only: bool = False
    # Heading paths absorbed by merging. A fragment folded into its neighbour
    # keeps its text but would otherwise lose the heading it was found under, so
    # the path is recorded here instead of being dropped.
    merged_sections: tuple[str, ...] = ()

    @property
    def section(self) -> str:
        """The heading path as one string, for SourceLocation.section."""
        return " > ".join(self.heading_path)

    def metadata(self) -> dict:
        """Extra fields worth keeping on the item, empty when there are none."""
        extra: dict = {}
        if self.heading_path:
            extra["heading_path"] = list(self.heading_path)
        if self.is_table:
            extra["is_table"] = True
        if self.heading_only:
            extra["heading_only"] = True
        if self.merged_sections:
            extra["merged_sections"] = list(self.merged_sections)
        return extra


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

def _words(line: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", line)


def _is_all_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8


def _ends_like_a_sentence(line: str) -> bool:
    return line.rstrip().endswith((".", "!", "?", ",", ";", ":"))


def _is_title_case(line: str) -> bool:
    """A short line whose significant words are capitalised: "Open Risks"."""
    if len(line) > MAX_HEADING_CHARS or _ends_like_a_sentence(line):
        return False
    words = [w for w in _words(line) if w.lower() not in SMALL_WORDS]
    if len(words) < 2:
        return False
    return sum(1 for w in words if w[0].isupper()) / len(words) >= 0.6


def heading_level(line: str) -> int | None:
    """Heading depth of a line, or None when it is not a heading.

    Deliberately conservative. A missed heading costs a little context in the
    citation; a false heading chops a paragraph in half and mislabels both
    pieces, so every rule here requires the line to look like a label rather
    than merely start with a capital.
    """
    markdown = MARKDOWN_HEADING.match(line)
    if markdown:
        return len(markdown.group(1))
    if len(line) > MAX_HEADING_CHARS or _ends_like_a_sentence(line):
        return None
    if _is_all_caps(line):
        return 1
    numbered = SECTION_NUMBER.match(line)
    if numbered:
        # "3.1 Module Design" is a section. "1 Name" is the first entry of a
        # numbered list on a slide, and treating it as a heading detaches every
        # line that follows from the text it belongs to.
        rest = numbered.group(2).strip()
        if "." in numbered.group(1) or len(rest) >= 12:
            return 1 + numbered.group(1).count(".")
        return None
    if _is_title_case(line):
        return 2
    return None


def _is_row(line: str) -> bool:
    """A table row: a CSV-style 'Row N --' line, or a line of separated cells."""
    if ROW_PREFIX.match(line):
        return True
    if " | " in line:
        return True
    return line.count("; ") >= MIN_ROW_SEPARATORS and len(line) <= 300


def _is_header_row(row: str) -> bool:
    """The first line of a table, which names the columns.

    Header rows are labels: they carry no digits. That distinction matters
    because a header is repeated onto every chunk of rows, and repeating a data
    row instead would silently duplicate a figure in the evidence.
    """
    return not any(c.isdigit() for c in row) and ("|" in row or "--" in row)


def _split_joined_rows(line: str) -> list[str]:
    """Undo a table that was flattened into one line with '; ' separators."""
    if "; " in line and ("|" in line or "--" in line):
        return [part.strip() for part in line.split("; ") if part.strip()]
    return [line]


def _blocks(text: str) -> list[Block]:
    """Turn raw extracted text into headings, paragraphs and table rows."""
    blocks: list[Block] = []
    paragraph: list[str] = []
    rows: list[str] = []
    table_header: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(paragraph).strip()
            if joined:
                blocks.append(Block(joined, "prose"))
            paragraph.clear()

    def flush_rows() -> None:
        for row in rows:
            blocks.append(Block(row, "row", header=table_header))
        rows.clear()

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            flush_paragraph()
            flush_rows()
            continue

        prefix = TABLE_PREFIX.match(line)
        if prefix:
            flush_paragraph()
            flush_rows()
            table_header = prefix.group(1)
            line = line[prefix.end():].strip()
            if not line:
                continue

        level = heading_level(line)
        if level is not None and not _is_row(line):
            flush_paragraph()
            flush_rows()
            blocks.append(Block(line, "heading", level=level))
            continue

        if _is_row(line):
            flush_paragraph()
            for row in _split_joined_rows(line):
                if table_header is None and _is_header_row(row):
                    table_header = row
                    continue
                rows.append(row)
            continue

        # Prose. PDF extraction wraps one paragraph across many lines, so
        # consecutive prose lines belong together until a blank line or heading.
        flush_rows()
        paragraph.append(line)

    flush_paragraph()
    flush_rows()
    return blocks


def _split_oversized(text: str, cap: int, overlap: int) -> list[str]:
    """Break one block that is longer than the cap, at sentence ends.

    This is the only place overlap is applied: these pieces really were cut
    mid-thought, so each one carries the tail of the previous.
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        # One sentence longer than the cap. Fall back to a word-boundary window
        # rather than dropping text: losing evidence is worse than an ugly cut.
        pieces, start = [], 0
        while start < len(text):
            end = min(start + cap, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                end = space if space > start else end
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return [p for p in pieces if p]

    def tail_of(piece: str) -> str:
        if overlap <= 0 or not piece:
            return ""
        tail = piece[-overlap:]
        cut = tail.find(" ")
        return (tail[cut + 1:] if cut >= 0 else tail).strip()

    pieces: list[str] = []
    current: list[str] = []
    length = 0                      # exactly len(" ".join(current))
    carry = ""

    for sentence in sentences:
        if len(sentence) > cap:
            # One sentence longer than the cap (a run-on OCR line, a table
            # flattened into prose). Word-split it: the cap bounds the context
            # budget, so an oversized piece is not an option.
            if current:
                pieces.append(" ".join(current))
                current, length = [], 0
            pieces.extend(_split_oversized(sentence, cap, overlap) or [sentence[:cap]])
            carry = ""
            continue

        if current and length + len(sentence) + 1 > cap:
            piece = " ".join(current)
            pieces.append(piece)
            current, length = [], 0
            carry = tail_of(piece)
            if carry and len(carry) + len(sentence) + 1 > cap:
                # The seam would push the next piece over the cap. Dropping it
                # loses a little continuity; exceeding the cap would quietly
                # break the context budget, which is worse.
                carry = ""

        if carry:
            current.append(carry)
            length = len(carry)
            carry = ""
        if current:
            length += 1 + len(sentence)
        else:
            length = len(sentence)
        current.append(sentence)

    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def split_document(
    text: str,
    *,
    max_chars: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
    base_heading: tuple[str, ...] | str = (),
) -> list[Chunk]:
    """Split extracted text into chunks that respect the document's structure.

    `base_heading` is what the caller already knows -- a DOCX section name, a
    page label -- and sits at the bottom of the heading stack.
    """
    cap = max_chars or settings.chunk_size
    over = overlap if overlap is not None else settings.chunk_overlap
    floor = min_chars or settings.min_chunk_chars
    # How much body text must have accumulated before a new heading is allowed to
    # start a chunk. The floor alone is not enough: a slide deck is dozens of
    # short labelled lines, and breaking at every one of them produced 65
    # two-line items from a 15-page deck. Those fragments then crowd the top-k
    # and, because mmr_max_per_source limits how many items one file may
    # contribute, the single fragment that answers the question loses its slot to
    # its own siblings. Half the cap keeps sections distinct in a prose document
    # (whose sections run to hundreds of characters) while letting a deck's
    # labels pack together.
    break_at = max(floor, cap // 2)
    if isinstance(base_heading, str):
        base_heading = (base_heading,) if base_heading else ()

    blocks = _blocks(text or "")
    if not blocks:
        return []

    chunks: list[Chunk] = []
    stack: list[str] = list(base_heading)   # heading path in force
    pieces: list[str] = []                  # body pieces of the chunk being built
    is_table = False
    header: str | None = None               # header row of the chunk being built
    active_header: str | None = None        # survives flushes: a table spans chunks
    pending_headings: list[str] = []        # headings seen since the last flush
    pending_depth = len(stack)              # stack depth before those headings

    def body_length() -> int:
        return sum(len(p) for p in pieces) + max(0, len(pieces) - 1)

    def flush() -> None:
        """Emit whatever the buffer holds.

        Body text carries the heading path as its section. Headings with no body
        under them become a chunk of their own, because their text is still the
        document's text -- a banner line, a slide title, a lone caption -- and
        dropping it would silently lose evidence.
        """
        nonlocal pieces, is_table, header, pending_headings, pending_depth
        if pieces:
            was_table = is_table
            if was_table:
                rows = [header] + pieces if header else list(pieces)
                body = "; ".join(rows)
            else:
                body = " ".join(pieces).strip()
            pieces, is_table, header = [], False, None
            if body:
                chunks.append(Chunk(body, tuple(stack), is_table=was_table))
        elif pending_headings:
            chunks.append(Chunk(
                " ".join(pending_headings).strip(),
                tuple(stack[:pending_depth]),
                heading_only=True,
            ))
        pending_headings, pending_depth = [], len(stack)

    def emit(text_piece: str, table: bool) -> None:
        """Add one block to the buffer, flushing first when it would not fit.

        A table never shares a chunk with prose: rows are joined with "; " so the
        header keeps sitting beside them, prose with " ", and mixing the two
        would put a paragraph inside a row list.
        """
        nonlocal is_table, header
        # A table chunk is emitted as "header; row; row", so the header counts
        # against the cap even though it is not one of the pieces.
        prospective = header if pieces else active_header
        extra = len(prospective) + 2 if (table and prospective) else 0
        if pieces and (
            table != is_table or body_length() + extra + len(text_piece) + 1 > cap
        ):
            flush()
            prospective = active_header
            extra = len(prospective) + 2 if (table and prospective) else 0
        if table:
            is_table = True
            header = prospective
        pieces.append(text_piece)

    for block in blocks:
        if block.kind == "heading":
            depth = len(base_heading) + max(1, block.level) - 1
            # Truncating rather than padding: when a document jumps straight to a
            # deep level, a shorter honest path beats invented parents.
            new_stack = stack[:depth]
            dropped = [h for h in pending_headings if h not in new_stack]
            if dropped and pieces:
                # Body text is waiting under headings this one is about to
                # truncate off the stack. Flush first and those headings survive
                # as that body's section path; otherwise they would be in neither
                # the section nor any chunk and their text would vanish.
                flush()
                dropped = []
            if dropped:
                chunks.append(Chunk(
                    " ".join(dropped).strip(), tuple(new_stack), heading_only=True
                ))
                pending_headings = [h for h in pending_headings if h in new_stack]
            # A heading starts a new chunk only once there is enough body text to
            # be worth separating (see break_at).
            if body_length() >= break_at:
                flush()
            if not pending_headings:
                # A heading's own text is about to become either the section path
                # or a chunk body, so the path it is filed under stops before it.
                pending_depth = min(len(stack), depth)
            pending_headings.append(block.text)
            stack = new_stack + [block.text]
            active_header = None
            continue

        if block.kind == "row":
            if block.header:
                active_header = block.header
            if len(block.text) > cap:
                flush()
                for piece in _split_oversized(block.text, cap, over):
                    chunks.append(Chunk(piece, tuple(stack), is_table=True))
                continue
            emit(block.text, table=True)
            continue

        active_header = None
        if len(block.text) > cap:
            flush()
            for piece in _split_oversized(block.text, cap, over):
                chunks.append(Chunk(piece, tuple(stack)))
            continue
        emit(block.text, table=False)

    flush()

    # Two merging passes, both length-driven. A chunk shorter than the floor is
    # usually one orphaned sentence or a slide label; folded back into a
    # neighbour it stops competing for top-k slots as a two-word item. Nothing is
    # ever dropped -- a small chunk is a worse citation than no chunk, and a
    # missing one is neither.
    merged = _merge_same_path(chunks, floor=floor, cap=cap)
    merged = _merge_fragments(merged, floor=floor, cap=cap)

    for position, chunk in enumerate(merged):
        chunk.index = position
    return merged


def _merge_same_path(chunks: list[Chunk], *, floor: int, cap: int) -> list[Chunk]:
    """Fold a too-short chunk back into its neighbour when both sit under the
    same heading path and still fit under the cap."""
    merged: list[Chunk] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and len(chunk.text) < floor
            and previous.heading_path == chunk.heading_path
            and previous.is_table == chunk.is_table
            and len(previous.text) + len(chunk.text) + 1 <= cap
        ):
            separator = "; " if chunk.is_table else " "
            merged[-1] = _absorb(previous, chunk, separator)
            continue
        merged.append(chunk)
    return merged


def _absorb(previous: Chunk, chunk: Chunk, separator: str) -> Chunk:
    """Merge `chunk` into `previous`, keeping both texts and both heading paths.

    The surviving section is the neighbour's -- it is the heading the combined
    text mostly sits under, and a citation with four headings in it is worse than
    one with the right one. The absorbed path is not thrown away: it goes into
    merged_sections, so the item still records where its second half came from.
    """
    absorbed = (
        (chunk.section,)
        if chunk.heading_path and chunk.heading_path != previous.heading_path
        else ()
    )
    return Chunk(
        previous.text + separator + chunk.text,
        previous.heading_path,
        previous.is_table,
        heading_only=previous.heading_only and chunk.heading_only,
        merged_sections=previous.merged_sections + absorbed,
    )


def _merge_fragments(chunks: list[Chunk], *, floor: int, cap: int) -> list[Chunk]:
    """Fold a fragment into the chunk before it, across heading paths, while the
    neighbour is itself still short."""
    limit = max(cap // 2, floor * 3)
    merged: list[Chunk] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and len(chunk.text) < floor
            and previous.is_table == chunk.is_table
            and len(previous.text) + len(chunk.text) + 1 <= limit
        ):
            separator = "; " if chunk.is_table else " "
            merged[-1] = _absorb(previous, chunk, separator)
            continue
        merged.append(chunk)
    return merged


def chunk_strings(
    text: str,
    *,
    max_chars: int | None = None,
    overlap: int | None = None,
    base_heading: tuple[str, ...] | str = (),
) -> list[str]:
    """split_document, for callers that only want the text."""
    return [c.text for c in split_document(
        text, max_chars=max_chars, overlap=overlap, base_heading=base_heading
    )]


def describe(text: str) -> str:
    """One-line summary of how a document would split -- for debugging and for
    the reindex report, so a chunking change is visible without reading chunks."""
    chunks = split_document(text)
    headings = {h for c in chunks for h in c.heading_path}
    return (
        f"{len(chunks)} chunk(s), {sum(len(c.text) for c in chunks)} chars, "
        f"{len(headings)} heading(s), {sum(1 for c in chunks if c.is_table)} table chunk(s)"
    )
