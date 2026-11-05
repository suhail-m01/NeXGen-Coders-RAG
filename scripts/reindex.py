"""
scripts/reindex.py -- rebuild the whole index from data\\ with current settings.

Use this after changing anything that affects how a source becomes items:
CHUNK_SIZE / CHUNK_OVERLAP, the OCR thresholds, the embedding model, diarization.
Re-running ingest_demo.py will NOT apply those changes, because ingestion is
idempotent by content hash: an unchanged file is skipped before extraction, so
new settings never touch it. --force re-extracts but leaves the old items in
place beside the new ones. This script starts from an empty index instead.

What is preserved and what is not:
  * Collections are preserved -- each source is re-filed under the collection it
    had, matched on file path first and filename second.
  * Chat history (history\\chat_history.db) and accounts (auth\\users.db) live
    outside index\\ and are untouched. Saved traces stay readable, which is the
    point of storing them.
  * Everything else in index\\ is rebuilt from scratch.

Run from the project root:
    .venv\\Scripts\\python scripts\\reindex.py --dry-run
    .venv\\Scripts\\python scripts\\reindex.py
    .venv\\Scripts\\python scripts\\reindex.py --yes --dir D:\\my\\files
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag_pipeline import get_pipeline  # noqa: E402


def _find_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def _collection_map(pipeline) -> tuple[dict[str, str], dict[str, str]]:
    """(by file path, by filename) -> collection, read before the wipe."""
    by_path: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for source in pipeline.store.list_sources():
        by_path.setdefault(str(Path(source.file_path).resolve()), source.collection)
        by_name.setdefault(source.filename, source.collection)
    return by_path, by_name


def _wipe_index() -> list[str]:
    removed = []
    for path in (settings.faiss_path, settings.db_path):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", action="append", default=None,
                        help="reindex this directory instead of data\\ "
                             "(repeatable)")
    parser.add_argument("--collection", default="general",
                        help="collection for files that were not indexed "
                             "before (default: general)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would happen and change nothing")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask for confirmation")
    args = parser.parse_args()

    targets = (
        [Path(d).resolve() for d in args.dir] if args.dir
        else [settings.documents_dir, settings.images_dir, settings.audio_dir]
    )
    settings.ensure_dirs()
    for path in targets:
        if not path.exists():
            print(f"  [WARN] not found: {path}")

    files = [f for t in targets if t.exists() for f in _find_files(t)]
    if not files:
        print("\nNo files to reindex.")
        return 1

    pipeline = get_pipeline()
    before = pipeline.store.stats()
    by_path, by_name = _collection_map(pipeline)

    print("Reindex plan")
    print("=" * 70)
    print(f"  index           {settings.index_dir}")
    print(f"  now             {before['sources']} source(s), {before['items']} item(s)")
    print(f"  files to ingest {len(files)}")
    print(f"  collections     preserved for {len(by_path)} known file(s); "
          f"new files -> '{args.collection}'")
    print()
    for path in files:
        resolved = str(path.resolve())
        collection = by_path.get(resolved) or by_name.get(path.name) or args.collection
        origin = "" if resolved in by_path or path.name in by_name else "  (new)"
        print(f"  {path.name:<32} -> {collection}{origin}")

    if args.dry_run:
        print("\n--dry-run: nothing was changed.")
        return 0

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("\nRefusing to wipe the index without a terminal to confirm on.")
            print("Re-run with --yes (or --dry-run to inspect the plan first).")
            return 1
        answer = input("\nDelete the index and rebuild it from these files? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; the index was not touched.")
            return 0

    removed = _wipe_index()
    print(f"\nRemoved: {', '.join(removed) or 'nothing (index was already empty)'}")
    print("Chat history and accounts were left untouched.\n")

    # A fresh store: the pipeline still holds the old connection and FAISS
    # index in memory, so build a new one rather than writing through it.
    from rag_pipeline import RAGPipeline

    pipeline.store.close()
    fresh = RAGPipeline()

    header = f"  {'FILE':<32} {'COLLECTION':<16} {'ITEMS':>6} {'RELS':>5}  {'SECS':>6}  STATUS"
    print(header)
    print("  " + "-" * (len(header) - 2))

    started = time.time()
    ok = failed = 0
    for path in files:
        file_start = time.time()
        resolved = str(path.resolve())
        collection = by_path.get(resolved) or by_name.get(path.name) or args.collection
        try:
            result = fresh.ingest_file(str(path), collection=collection)
        except Exception as exc:
            failed += 1
            print(f"  {path.name:<32} {collection:<16} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  ERROR {exc}")
            continue
        if result.success:
            ok += 1
            print(f"  {path.name:<32} {collection:<16} {len(result.items):>6} "
                  f"{len(result.relationships):>5} {time.time() - file_start:>6.1f}  OK")
        else:
            failed += 1
            code = result.error_code.value if result.error_code else "FAILED"
            print(f"  {path.name:<32} {collection:<16} {'-':>6} {'-':>5} "
                  f"{time.time() - file_start:>6.1f}  {code}: {result.error}")
        for warning in result.warnings:
            print(f"      [warn] {warning}")

    after = fresh.store.stats()
    print(f"\n  Reindexed {ok} file(s), {failed} failure(s) in {time.time() - started:.1f}s")
    print(f"  Index totals: sources={after['sources']} items={after['items']} "
          f"relationships={after['relationships']}")
    print(f"  Collections: {fresh.store.list_collections()}")
    fresh.store.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
