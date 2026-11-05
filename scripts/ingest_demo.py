"""
scripts/ingest_demo.py -- build the searchable index from the demo dataset.

Ingests everything under data\\documents, data\\images and data\\audio through
the real pipeline (extract -> embed -> FAISS + SQLite -> relationships), then
prints a per-file table and the resulting index totals.

Idempotent: source_id is a content hash and existing item_ids are skipped, so
re-running will not create duplicate vectors. By default a file whose hash is
already in the index is skipped BEFORE extraction, which matters because
extraction is the expensive part - Whisper re-transcribes a whole recording
even when the resulting items would be discarded as duplicates. Use --force to
re-extract anyway (for example after changing an extractor or a model).

Run from the project root:
    .venv\\Scripts\\python scripts\\ingest_demo.py
    .venv\\Scripts\\python scripts\\ingest_demo.py --dir D:\\my\\files
    .venv\\Scripts\\python scripts\\ingest_demo.py --force

An edited file is a NEW source, because source_id is a content hash: without
--replace the previous version stays indexed and keeps being cited beside the
new one (either can rank higher). --replace removes the older version(s) of the
same filename as it indexes the new one:

    .venv\\Scripts\\python scripts\\ingest_demo.py --dir D:\\reports --replace

To re-extract the whole corpus under changed settings (chunking, OCR, models),
use scripts\\reindex.py instead: --force re-extracts but keeps the old items.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag_pipeline import get_pipeline  # noqa: E402
from schemas import content_hash  # noqa: E402


def _find_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", action="append", default=None,
                        help="ingest this directory instead of the demo data "
                             "(repeatable)")
    parser.add_argument("--collection", default="general",
                        help="Logical collection to file these sources under "
                             "(default: general). Search can be scoped to it.")
    parser.add_argument("--force", action="store_true",
                        help="re-extract files that are already indexed")
    parser.add_argument("--replace", action="store_true",
                        help="when a filename is already indexed with different "
                             "content, remove the older version(s) instead of "
                             "keeping both")
    args = parser.parse_args()

    if args.dir:
        targets = [Path(d).resolve() for d in args.dir]
    else:
        targets = [settings.documents_dir, settings.images_dir, settings.audio_dir]

    pipeline = get_pipeline()
    settings.ensure_dirs()

    missing = [t for t in targets if not t.exists()]
    for path in missing:
        print(f"  [WARN] not found: {path}")

    all_files = [f for t in targets if t.exists() for f in _find_files(t)]
    if not all_files:
        print("\nNo files to ingest.")
        if not args.dir:
            print("Create the demo dataset first:")
            print("    .venv\\Scripts\\python create_demo_data.py")
            print("Or point at your own folder:")
            print("    .venv\\Scripts\\python scripts\\ingest_demo.py --dir D:\\my\\files")
        return 1

    print(f"Ingesting {len(all_files)} file(s) from {len(targets)} folder(s)...")
    print(f"Index location: {settings.index_dir}\n")

    header = f"  {'FILE':<28} {'TYPE':<9} {'ITEMS':>6} {'RELS':>5}  {'SECS':>6}  STATUS"
    print(header)
    print("  " + "-" * (len(header) - 2))

    started = time.time()
    ok_count = 0
    fail_count = 0
    skipped_count = 0
    total_items = 0
    total_rels = 0

    for path in all_files:
        file_start = time.time()

        # Cheap pre-checks. Skipping BEFORE extraction is the point: extraction
        # is the expensive stage (Whisper re-transcribes a whole recording even
        # when every resulting item would be discarded as a duplicate).
        try:
            size = path.stat().st_size
        except OSError as exc:
            fail_count += 1
            print(f"  {path.name:<28} {'-':<9} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  UNREADABLE {exc}")
            continue

        if size > settings.max_file_bytes:
            skipped_count += 1
            print(f"  {path.name:<28} {'-':<9} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  SKIP {size:,} bytes exceeds "
                  f"MAX_FILE_BYTES ({settings.max_file_bytes:,})")
            continue

        try:
            digest = content_hash(path.read_bytes())
        except Exception as exc:
            digest = None
            print(f"      [warn] could not hash {path.name}: {exc}")

        if digest and not args.force and pipeline.store.source_exists(digest):
            skipped_count += 1
            print(f"  {path.name:<28} {'-':<9} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  SKIP already indexed "
                  f"(hash {digest}) - use --force to re-extract")
            continue

        try:
            result = pipeline.ingest_file(
                str(path), collection=args.collection, replace=args.replace
            )
        except Exception as exc:
            fail_count += 1
            print(f"  {path.name:<28} {'-':<9} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  EXCEPTION {exc.__class__.__name__}: {exc}")
            continue

        items = len(result.items)
        rels = len(result.relationships)
        source_type = getattr(result.source, "source_type", None)
        type_name = getattr(source_type, "value", str(source_type))[:9]
        elapsed = time.time() - file_start

        if result.success and items:
            ok_count += 1
            total_items += items
            total_rels += rels
            status = "OK"
        else:
            fail_count += 1
            status = f"FAILED: {result.error or result.error_code or 'no items extracted'}"

        print(f"  {path.name:<28} {type_name:<9} {items:>6} {rels:>5} "
              f"{elapsed:>6.1f}  {status}")

        for warning in result.warnings:
            print(f"      [warn] {warning}")

    elapsed_total = time.time() - started
    print(f"\n  Ingested {ok_count} file(s), skipped {skipped_count}, "
          f"{fail_count} failure(s) in {elapsed_total:.1f}s")
    print(f"  New items this run: {total_items}   relationships: {total_rels}")
    if ok_count == 0 and fail_count == 0 and skipped_count:
        print("  Nothing was new - the index already covers these files.")

    stale = pipeline.stale_versions()
    if stale:
        print(f"\n  [WARN] {len(stale)} filename(s) have more than one indexed version:")
        for name, versions in sorted(stale.items()):
            ids = ", ".join(v.source_id[:12] for v in versions)
            print(f"         {name}: {ids}")
        if args.replace:
            print("         (--replace was requested; these came from files not in this run)")
        print("         Re-run with --replace to keep only the newest, or delete them in the UI.")

    health = pipeline.health()
    index = health.get("index", {})
    print(f"  Index totals: sources={index.get('sources')} items={index.get('items')}")

    tesseract = health.get("tesseract", {})
    if not tesseract.get("available"):
        print("\n  [WARN] Tesseract OCR is unavailable, so images were indexed")
        print("         without OCR text. Install it with:")
        print("           winget install UB-Mannheim.TesseractOCR")
        print("         then re-run this script.")

    whisper = health.get("whisper", {})
    if whisper and not whisper.get("available", True):
        print("\n  [WARN] faster-whisper is unavailable, so audio was skipped.")

    if fail_count:
        print("\nSome files failed. Fix the causes above and re-run - already")
        print("ingested files are skipped rather than duplicated.")
        return 1

    print("\nIndex ready. Next:")
    print("    .\\start.ps1                        launch API + UI")
    print("    .venv\\Scripts\\python scripts\\smoke_test.py    ask test questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
