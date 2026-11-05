"""
scripts/smoke_test.py -- ask a few real questions and show what comes back.

Runs a small set of questions against the existing index and prints the actual
answer, confidence, abstention status, latency and citation locations for each.
Useful as a first check after setup, and as a quick regression check after any
change to the pipeline or a model swap.

Run from the project root:
    .venv\\Scripts\\python scripts\\smoke_test.py
    .venv\\Scripts\\python scripts\\smoke_test.py --query "How many records are in the dataset?"
    .venv\\Scripts\\python scripts\\smoke_test.py --timeout 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_pipeline import get_pipeline  # noqa: E402
from schemas import QueryRequest  # noqa: E402

DEFAULT_QUERIES = [
    ("Tabular", "How many records are in the project metrics dataset and what accuracy was achieved?"),
    ("Audio", "What accuracy figure was mentioned in the project meeting recording?"),
    ("Document", "What is the current status of the project?"),
    ("Abstention", "What was the closing stock price of the company last quarter?"),
]


def _location_text(location) -> str:
    """Render a citation location using the schema's own formatter.

    SourceLocation.human() already knows how to show a page, an audio
    timestamp range (timestamp_start/timestamp_end) or a section, so prefer it
    and only fall back to raw fields if it is unavailable or empty.
    """
    renderer = getattr(location, "human", None)
    if callable(renderer):
        try:
            text = str(renderer()).strip()
            if text:
                return text
        except Exception:
            pass

    bits = []
    page = getattr(location, "page_number", None)
    if page is not None:
        bits.append(f"p.{page}")
    start = getattr(location, "timestamp_start", None)
    end = getattr(location, "timestamp_end", None)
    if start is not None:
        bits.append(f"{start:.1f}-{end:.1f}s" if end is not None else f"{start:.1f}s")
    section = getattr(location, "section", None)
    if section:
        bits.append(str(section))
    return " ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=None,
                        help="ask this instead of the default set (repeatable)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-query wall-clock budget in seconds (default: 300)")
    args = parser.parse_args()

    queries = ([(f"Custom {i + 1}", q) for i, q in enumerate(args.query)]
               if args.query else DEFAULT_QUERIES)

    pipeline = get_pipeline()
    health = pipeline.health()
    index = health.get("index", {})
    print(f"backend={health.get('backend')} online={health.get('online')}")
    print(f"index: sources={index.get('sources')} items={index.get('items')}")
    if not index.get("items"):
        print("\nThe index is empty. Ingest something first:")
        print("    .venv\\Scripts\\python scripts\\ingest_demo.py")
        return 1

    failures = 0
    for label, question in queries:
        print(f"\n[{label}]")
        print(f"  Q: {question}")
        start = time.time()
        try:
            response = pipeline.query(QueryRequest(query=question))
        except Exception as exc:
            failures += 1
            print(f"  ERROR after {time.time() - start:.0f}s: {exc.__class__.__name__}: {exc}")
            continue

        elapsed = time.time() - start
        if elapsed >= args.timeout:
            print(f"  [WARN] took {elapsed:.0f}s - over the {args.timeout:.0f}s budget")
        print(f"  {elapsed:.0f}s | abstained={response.abstained} "
              f"confidence={response.confidence}")
        answer = (response.answer or "").strip().replace("\n", " ")
        print(f"  A: {answer[:300]}")
        citations = response.citations or []
        if citations:
            print(f"  citations ({len(citations)}):")
            for citation in citations[:4]:
                print(f"    - {citation.filename} {_location_text(citation.location)}")
        else:
            print("  citations: none")

    print(f"\nDone. {len(queries) - failures}/{len(queries)} questions returned an answer.")
    print("Remember: 'confidence' is a normalised rerank score, not a calibrated")
    print("probability of correctness - always check the cited passages yourself.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
