"""Print or write the run manifest: the exact build behind a result.

    python scripts/manifest.py                 # human-readable, to stdout
    python scripts/manifest.py --json          # machine-readable, to stdout
    python scripts/manifest.py --out m.json    # write JSON to a file
    python scripts/manifest.py --with-index    # include live index counts

Useful before a demo or an evaluation run: it names the commit, the models,
every retrieval, chunking, guard and generation setting, the auth mode and the
dataset version, so a number in a report can be traced to the build that
produced it. The same dict is served by GET /manifest and embedded in every
evaluation report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability import build_manifest, manifest_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--out", default=None, help="write JSON to this path")
    parser.add_argument(
        "--with-index", action="store_true",
        help="open the index and include live counts (sources, items, parents)",
    )
    args = parser.parse_args()

    stats = None
    if args.with_index:
        from rag_pipeline import get_pipeline

        stats = get_pipeline().store.stats()

    manifest = build_manifest(index_stats=stats)
    text = json.dumps(manifest, indent=2)

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"Manifest written to {target}")
        return 0
    print(text if args.json else manifest_text(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
