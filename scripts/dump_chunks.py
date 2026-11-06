"""Show what a file becomes when it is chunked -- without indexing any of it.

    python scripts\\dump_chunks.py --file data\\documents\\project_report.pdf
    python scripts\\dump_chunks.py --file slides.pptx --chunking-mode structure
    python scripts\\dump_chunks.py --source project_report.pdf --grep "1.2 million"
    python scripts\\dump_chunks.py --all --stats
    python scripts\\dump_chunks.py --file notes.docx --full --out chunks.txt
    python scripts\\dump_chunks.py --source deck.pdf --json

Why this exists
---------------
Every chunking change in this project was argued about in the dark until the
chunks could be read. A table split across two chunks, a heading that never
reached the body, a CSV collapsed into one item, a slide deck where page 11 is
OCR and page 3 is text: all of these are visible in one screen of output and
invisible in a retrieval score. This prints them.

Two modes, because they answer two different questions.

`--file` runs the real extraction and chunking path and stops before anything is
embedded or written. It answers "what would this document become?" and it is the
only way to try `--chunking-mode structure` or a different `--chunk-size` without
rebuilding the index and without editing `.env`. Nothing is persisted: no index
write, no history row, no source record.

`--source` reads what is *already* in the index. It answers "what is actually
citable right now?", which is not the same question -- the stored items are the
ground truth a citation points at, and they can differ from a fresh dry run
because the settings or the code changed since ingestion. `--parents` adds the
parent blocks structure mode stores for small-to-big retrieval.

What it does not tell you
-------------------------
A chunk that reads well here can still lose in retrieval to a near-duplicate of
itself, and this tool says nothing about ranking, reranking or what a model would
do with the text. It shows the evidence as it is stored, not the answer. Use
`evaluation.py --retrieval-only` for the second question.

Requirements: no Ollama, no GPU and no network for documents, spreadsheets and
slides. An image dry run needs Tesseract and may call the vision model if one is
configured (it warns and falls back to OCR when it cannot); an audio dry run
needs faster-whisper and transcribes for real, which is slow.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


JSON_MODE = False


def _say(line: str = "") -> None:
    """Print a human-facing line, to stderr when the run is emitting JSON.

    `--json > chunks.json` has to produce a parseable file, so the preamble and
    the warnings cannot share stdout with the payload. They are still shown: a
    dry run that hides its warnings is a dry run that misleads.
    """
    print(line, file=sys.stderr if JSON_MODE else sys.stdout)


def _location(item) -> str:
    """Where this item sits in its source, using only the fields that are set.

    A location is rendered from what the processor actually recorded. Inventing a
    page number for an audio segment, or a timestamp for a paragraph, is exactly
    the kind of confident detail this project is built to avoid.
    """
    parts = []
    loc = item.location
    if loc.page_number is not None:
        parts.append(f"p.{loc.page_number}")
    if loc.section:
        parts.append(loc.section)
    if loc.timestamp_start is not None:
        end = loc.timestamp_end
        parts.append(
            f"{_clock(loc.timestamp_start)}-{_clock(end)}" if end is not None
            else _clock(loc.timestamp_start)
        )
    if loc.image_path and not parts:
        parts.append(Path(loc.image_path).name)
    return " · ".join(parts) if parts else "-"


def _clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _preview(text: str, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    if limit <= 0 or len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + " …"


def _flags(item) -> str:
    """The metadata worth seeing at a glance, because it explains a weird chunk.

    These are the annotations the guards and the UI rely on: an OCR'd item with a
    low digit confidence, a chunk that is only a heading, a parent expanded from a
    child. Without them a dump invites the wrong conclusion about why a chunk
    looks truncated.
    """
    meta = item.metadata or {}
    marks = []
    if meta.get("ocr_unreliable"):
        marks.append("ocr-unreliable")
    if isinstance(meta.get("ocr_confidence"), (int, float)):
        marks.append(f"ocr {float(meta['ocr_confidence']):.0f}%")
    if meta.get("vision_ok") is False:
        marks.append("no-vision")
    if meta.get("heading_only"):
        marks.append("heading-only")
    if meta.get("expanded_from"):
        marks.append("parent")
    if meta.get("retrieved_child"):
        marks.append("retrieved-child")
    if meta.get("merged_sections"):
        marks.append(f"merged {len(meta['merged_sections'])}")
    if meta.get("is_table"):
        # The key extractors actually write. This read "table" until a PDF table
        # grid was dumped and showed no mark, which is how it was caught.
        marks.append("table")
    if meta.get("context"):
        marks.append("contextual")
    if meta.get("speaker") is not None:
        marks.append(f"speaker {meta['speaker']}")
    return " ".join(marks)


def _rows(items, chars: int, needle: str | None) -> list[dict]:
    rows = []
    for position, item in enumerate(items, start=1):
        text = item.content or ""
        if needle and needle.lower() not in text.lower():
            continue
        rows.append({
            "n": position,
            "item_id": item.item_id,
            "modality": item.modality.value,
            "location": _location(item),
            "chars": len(text),
            "flags": _flags(item),
            "text": text if chars <= 0 else _preview(text, chars),
            # What sat in front of this chunk when it was embedded, if contextual
            # retrieval wrote a blurb at ingest. Reported separately and left out of
            # `chars` because it is not citable text: it never reaches an answer, an
            # excerpt or a screen, and an inspector that mixed the two would be
            # showing a chunk that does not exist anywhere in the product.
            "indexed_context": (item.metadata or {}).get("context"),
            "parent_id": item.parent_id,
        })
    return rows


def _stats(items) -> dict:
    """Sizes per modality, because one modality's chunks are rarely another's."""
    out: dict = {}
    for item in items:
        bucket = out.setdefault(item.modality.value, [])
        bucket.append(len(item.content or ""))
    summary = {}
    for modality, sizes in sorted(out.items()):
        summary[modality] = {
            "chunks": len(sizes),
            "chars_min": min(sizes),
            "chars_median": int(statistics.median(sizes)),
            "chars_mean": round(statistics.fmean(sizes), 1),
            "chars_max": max(sizes),
            "chars_total": sum(sizes),
        }
    return summary


def _print_table(rows: list[dict], chars: int, empty_note: str = "") -> None:
    if not rows:
        print(empty_note or "Nothing to show.")
        print()
        return
    width = 0 if chars <= 0 else min(chars, 100)
    for row in rows:
        head = (f"{row['n']:>3} | {row['item_id'][:12]:12s} | {row['modality']:6s} | "
                f"{row['location'][:28]:28s} | {row['chars']:>5} ch")
        if row["flags"]:
            head += f" | {row['flags']}"
        print(head)
        if width == 0:
            print("      " + row["text"])
        else:
            print(f"      {row['text']}")
        print()


def _dump_file(path: Path, args) -> int:
    from config import settings
    from rag_pipeline import RAGPipeline

    overrides = []
    if args.chunking_mode:
        overrides.append(f"chunking_mode {settings.chunking_mode} -> {args.chunking_mode}")
        settings.chunking_mode = args.chunking_mode
    if args.chunk_size:
        overrides.append(f"chunk_size {settings.chunk_size} -> {args.chunk_size}")
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        overrides.append(f"chunk_overlap {settings.chunk_overlap} -> {args.chunk_overlap}")
        settings.chunk_overlap = args.chunk_overlap

    processor = RAGPipeline.route(path)
    if processor is None:
        print(f"Unsupported file type '{path.suffix}'.", file=sys.stderr)
        print("Documents, spreadsheets, slides, PDFs, images and audio are "
              "supported; see rag_pipeline.route().", file=sys.stderr)
        return 2

    _say("Dry run -- nothing is embedded, written or indexed.")
    _say(f"file          {path}")
    _say(f"processor     {processor.__module__}.{processor.__name__}")
    _say(f"chunking      mode={settings.chunking_mode} size={settings.chunk_size} "
         f"overlap={settings.chunk_overlap}")
    if overrides:
        _say(f"overrides     {'; '.join(overrides)} (this process only)")
    _say("=" * 78)

    result = processor(str(path))
    if not result.success:
        print(f"EXTRACTION FAILED: {result.error}", file=sys.stderr)
        if result.error_code:
            print(f"  code: {result.error_code.value}", file=sys.stderr)
        return 2

    for warning in result.warnings or []:
        _say(f"[warning] {warning}")
    if result.warnings:
        _say("-" * 78)

    items = list(result.items or [])
    rows = _rows(items, args.chars, args.grep)
    if args.json:
        _emit_json({"file": str(path), "source": result.source.model_dump(mode="json"),
                    "warnings": list(result.warnings or []),
                    "chunking": {"mode": settings.chunking_mode,
                                 "chunk_size": settings.chunk_size,
                                 "chunk_overlap": settings.chunk_overlap},
                    "stats": _stats(items), "chunks": rows}, args.out)
    else:
        _say(f"{len(items)} chunk(s) from {result.source.filename} "
             f"[{result.source.source_type.value}]"
             + (f" -- {len(rows)} matching '{args.grep}'" if args.grep else ""))
        _say("=" * 78)
        _print_table(rows, args.chars,
                     f"No chunk of this file contains '{args.grep}'." if args.grep
                     else "Extraction produced no chunks.")
        if args.stats:
            _print_stats(_stats(items))
    return 0


def _dump_source(pipeline, needle_id: str, args) -> int:
    from config import settings

    store = pipeline.store
    source = store.get_source(needle_id)
    if source is None:
        matches = store.sources_by_filename(needle_id)
        if len(matches) == 1:
            source = matches[0]
        elif len(matches) > 1:
            print(f"'{needle_id}' matches {len(matches)} indexed versions:", file=sys.stderr)
            for candidate in matches:
                print(f"  {candidate.source_id}  collection={candidate.collection}",
                      file=sys.stderr)
            print("Pass a source_id to pick one.", file=sys.stderr)
            return 2
    if source is None:
        print(f"No indexed source '{needle_id}'.", file=sys.stderr)
        print("Known sources:", file=sys.stderr)
        for known in store.list_sources():
            print(f"  {known.source_id}  {known.filename}", file=sys.stderr)
        return 2

    items = store.get_items_by_source(source.source_id)
    _say("Indexed source -- what is citable right now, which is not necessarily "
         "what a fresh dry run would produce.")
    _say(f"source        {source.filename}  [{source.source_type.value}]")
    # The id IS the content hash, which is why re-ingesting identical bytes is a
    # no-op and why a changed file becomes a second source rather than an update.
    _say(f"source_id     {source.source_id}  (content hash of the file's bytes)")
    _say(f"collection    {source.collection}")
    _say(f"file          {source.file_path}")
    _say(f"size          {source.file_size if source.file_size is not None else '-'} bytes")
    _say(f"items         {len(items)}")
    _say("=" * 78)

    rows = _rows(items, args.chars, args.grep)
    stats = _stats(items)
    parents = store.get_parents_by_source(source.source_id) if args.parents else []

    if args.json:
        _emit_json({"source": source.model_dump(mode="json"), "stats": stats,
                    "chunks": rows,
                    "parents": _rows(parents, args.chars, None)}, args.out)
    else:
        _print_table(rows, args.chars,
                     f"No stored chunk contains '{args.grep}'." if args.grep
                     else "This source has no items in the index.")
        if args.parents:
            if parents:
                _say(f"{len(parents)} parent block(s) -- retrieved children expand to these:")
                _say("=" * 78)
                _print_table(_rows(parents, args.chars, None), args.chars)
            else:
                _say("No parent blocks are stored for this source. Parents exist only "
                     "when CHUNKING_MODE=structure was in effect at ingest time; the "
                     f"index was built in '{settings.chunking_mode}' mode.")
        if args.stats:
            _print_stats(stats)
    return 0


def _print_stats(stats: dict) -> None:
    print("CHUNK SIZES")
    print(f"  {'modality':8s} {'chunks':>6} {'min':>6} {'median':>7} {'mean':>8} "
          f"{'max':>6} {'total':>8}")
    for modality, block in stats.items():
        print(f"  {modality:8s} {block['chunks']:>6} {block['chars_min']:>6} "
              f"{block['chars_median']:>7} {block['chars_mean']:>8} "
              f"{block['chars_max']:>6} {block['chars_total']:>8}")


def _emit_json(payload: dict, out: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", type=Path,
                        help="Run extraction and chunking on this file and print the "
                             "chunks. Nothing is embedded or written.")
    target.add_argument("--source",
                        help="Print the chunks already stored for an indexed source "
                             "(filename or source_id).")
    target.add_argument("--all", action="store_true",
                        help="Print every indexed source with its chunk count and sizes.")
    parser.add_argument("--parents", action="store_true",
                        help="With --source: also print the parent blocks structure mode "
                             "stores for small-to-big retrieval.")
    parser.add_argument("--grep", metavar="TEXT",
                        help="Only chunks whose text contains this (case-insensitive).")
    parser.add_argument("--chars", type=int, default=280,
                        help="Characters of each chunk to show, 0 for the whole text "
                             "(same as --full). Default 280.")
    parser.add_argument("--full", action="store_true", help="Show whole chunk texts.")
    parser.add_argument("--stats", action="store_true",
                        help="Also print chunk-size statistics per modality.")
    parser.add_argument("--chunking-mode", choices=["fixed", "structure"],
                        help="With --file: chunk in this mode for this run only, so the "
                             "two can be compared without rebuilding the index.")
    parser.add_argument("--chunk-size", type=int,
                        help="With --file: override CHUNK_SIZE for this run only.")
    parser.add_argument("--chunk-overlap", type=int,
                        help="With --file: override CHUNK_OVERLAP for this run only.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--out", type=Path, help="With --json: write here instead of stdout.")
    args = parser.parse_args()
    if args.full:
        args.chars = 0
    global JSON_MODE
    JSON_MODE = bool(args.json)

    if args.file:
        path = args.file.expanduser().resolve()
        if not path.exists():
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        return _dump_file(path, args)

    from rag_pipeline import get_pipeline

    pipeline = get_pipeline()
    if args.source:
        return _dump_source(pipeline, args.source, args)

    sources = pipeline.store.list_sources()
    if not sources:
        print("The index is empty. Ingest something first:")
        print("  python scripts\\ingest_demo.py")
        return 2
    print(f"{len(sources)} indexed source(s)")
    print("=" * 78)
    payload = []
    for source in sources:
        items = pipeline.store.get_items_by_source(source.source_id)
        parents = pipeline.store.get_parents_by_source(source.source_id)
        stats = _stats(items)
        total = sum(block["chunks"] for block in stats.values())
        print(f"{source.filename:34s} {source.source_type.value:10s} "
              f"collection={source.collection:12s} items={len(items):>3} "
              f"parents={len(parents):>2}")
        if args.stats and stats:
            _print_stats(stats)
            print()
        payload.append({"source": source.model_dump(mode="json"), "stats": stats,
                        "chunks": total, "parents": len(parents)})
    if args.json:
        _emit_json({"sources": payload, "index": pipeline.store.stats()}, args.out)
    elif not args.stats:
        print("\nUse --source <name> to see a source's chunks, or --file <path> for a "
              "dry run of a file that is not indexed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
