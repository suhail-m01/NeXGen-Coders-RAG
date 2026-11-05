"""
scripts/calibrate_abstention.py -- set ABSTENTION_THRESHOLD from measured data.

The abstention gate compares the best cross-encoder rerank score against
ABSTENTION_THRESHOLD. A hand-picked threshold is a guess: too high and real
questions get refused, too low and unanswerable ones get answered anyway (the
measured failure mode of this project's own evaluation, Q6).

This script replaces the guess with a measurement, adapting SpectraRAG's
`scripts/calibrate_refusal.py` methodology to a fully offline pipeline:

  1. POSITIVE set - queries the corpus CAN answer, in REAL question phrasing:
     the evaluation dataset's expect_abstention=false queries, plus any --pos
     questions you supply. Verbatim sentences copied out of chunks are NOT
     used by default (--synthetic opts in): they score far higher than any
     paraphrased question, and calibrating on them would set a threshold that
     refuses ordinary user questions.
  2. NEGATIVE set - queries the corpus CANNOT answer: the dataset's
     expect_abstention=true queries plus built-in out-of-corpus probes,
     extended with any --neg questions you supply.
  3. Every query is scored through the real retrieval+rerank path; the top
     rerank score per query is recorded.
  4. A threshold sweep picks the score that best separates the two sets.

Nothing is changed unless you pass --apply, which writes the value into .env
(other lines untouched). Without --apply it is pure diagnosis.

Run from the project root:
    .venv\\Scripts\\python scripts\\calibrate_abstention.py
    .venv\\Scripts\\python scripts\\calibrate_abstention.py --apply
    .venv\\Scripts\\python scripts\\calibrate_abstention.py --neg "What is our ISO audit date?"
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag_pipeline import get_pipeline  # noqa: E402
from schemas import QueryRequest  # noqa: E402

BUILT_IN_PROBES = [
    "What was the closing stock price of the company last quarter?",
    "Who is the chief executive officer of the organization?",
    "What is the projected revenue for the year 2040?",
    "Which football team won the league championship in 1994?",
    "How do I reset the password on my home router?",
    "What dinner reservations are booked for tonight?",
]

QUERY_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "dataset" / "queries.json"


def _top_rerank_score(pipeline, question: str) -> float:
    """Run the real retrieve+rerank path and return the best rerank score.

    Deliberately stops before the gate and before any LLM call: calibration
    must measure the evidence score itself, not the model's opinion of it.
    """
    representation = pipeline.build_query_representation(
        QueryRequest(query=question)
    )
    retrieved = pipeline.retriever.retrieve(
        representation, top_k=settings.top_k
    )
    if not retrieved:
        return -99.0
    reranked = pipeline.reranker.rerank(representation, retrieved, top_k=1)
    return float(reranked[0].rerank_score) if reranked else -99.0


def _sentence_query(content: str, rng: random.Random) -> str | None:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content or "")
                 if 40 <= len(s.strip()) <= 220]
    if not sentences:
        return None
    return rng.choice(sentences)


def build_sets(
    pipeline,
    extra_negatives: list[str],
    extra_positives: list[str],
    include_synthetic: bool,
    sample: int,
    seed: int,
):
    rng = random.Random(seed)
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []

    if QUERY_FILE.exists():
        dataset = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
        queries = dataset if isinstance(dataset, list) else dataset.get("queries", [])
        for entry in queries:
            question = entry.get("query")
            if not question:
                continue
            if entry.get("expect_abstention"):
                negatives.append((f"dataset:{entry.get('id')}", question))
            else:
                positives.append((f"dataset:{entry.get('id')}", question))
    else:
        print(f"  [warn] {QUERY_FILE} not found - using synthetic positives only")

    positives += [(f"custom-pos:{i}", q) for i, q in enumerate(extra_positives)]

    if include_synthetic:
        items = pipeline.store.all_items()
        rng.shuffle(items)
        for item in items[:sample]:
            question = _sentence_query(item.content, rng)
            if question:
                positives.append((f"item:{item.item_id[:8]}", question))

    negatives += [(f"probe:{i}", q) for i, q in enumerate(BUILT_IN_PROBES)]
    negatives += [(f"custom:{i}", q) for i, q in enumerate(extra_negatives)]
    return positives, negatives


def best_threshold(pos_scores: list[float], neg_scores: list[float]):
    """Sweep candidate thresholds; maximise separation accuracy.

    A positive must score >= threshold, a negative must score < threshold.
    Ties resolve toward the HIGHER threshold, i.e. toward refusing more,
    because an unanswered question is visibly safe whereas a confident
    wrong answer is not.
    """
    candidates = sorted(set(pos_scores + neg_scores))
    middles = [
        (a + b) / 2
        for a, b in zip(candidates, candidates[1:])
    ]
    best = None
    for threshold in candidates + middles:
        correct = (
            sum(1 for s in pos_scores if s >= threshold)
            + sum(1 for s in neg_scores if s < threshold)
        )
        accuracy = correct / (len(pos_scores) + len(neg_scores))
        if best is None or accuracy > best[1] + 1e-12 or (
            abs(accuracy - best[1]) <= 1e-12 and threshold > best[0]
        ):
            best = (threshold, accuracy)
    return best


def apply_to_env(value: float) -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    line = f"ABSTENTION_THRESHOLD={value:.4f}"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        if re.search(r"(?m)^\s*ABSTENTION_THRESHOLD\s*=", text):
            text = re.sub(r"(?m)^(\s*ABSTENTION_THRESHOLD\s*=\s*).+$",
                          lambda m: m.group(1) + f"{value:.4f}", text)
        else:
            text = text.rstrip("\n") + f"\n{line}\n"
        env_path.write_text(text, encoding="utf-8")
    else:
        env_path.write_text(line + "\n", encoding="utf-8")
    print(f"  [OK] .env updated: {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the calibrated threshold into .env")
    parser.add_argument("--neg", action="append", default=[],
                        help="extra out-of-corpus question (repeatable)")
    parser.add_argument("--pos", action="append", default=[],
                        help="extra answerable question in real phrasing (repeatable)")
    parser.add_argument("--synthetic", action="store_true",
                        help="also derive positives from verbatim chunk sentences "
                             "(NOT recommended: they score unlike real questions)")
    parser.add_argument("--sample", type=int, default=40,
                        help="how many indexed items to derive positives from")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pipeline = get_pipeline()
    index = pipeline.health().get("index", {})
    print(f"index: sources={index.get('sources')} items={index.get('items')}")
    print(f"current ABSTENTION_THRESHOLD = {settings.abstention_threshold}")
    if not index.get("items"):
        print("The index is empty - ingest files first (scripts\\ingest_demo.py).")
        return 1

    positives, negatives = build_sets(
        pipeline, args.neg, args.pos, args.synthetic, args.sample, args.seed
    )
    print(f"scoring {len(positives)} answerable and {len(negatives)} "
          f"unanswerable queries through retrieve+rerank...")

    pos_scores, neg_scores = [], []
    for label, question in positives:
        score = _top_rerank_score(pipeline, question)
        pos_scores.append(score)
        print(f"  [pos {score:8.3f}] {label}: {question[:70]}")
    for label, question in negatives:
        score = _top_rerank_score(pipeline, question)
        neg_scores.append(score)
        print(f"  [neg {score:8.3f}] {label}: {question[:70]}")

    if not pos_scores or not neg_scores:
        print("Need both an answerable and an unanswerable set to calibrate.")
        return 1

    threshold, accuracy = best_threshold(pos_scores, neg_scores)
    overlap = (max(neg_scores) >= min(pos_scores))

    print("\n--- separation ---")
    print(f"  answerable   : min {min(pos_scores):8.3f}  max {max(pos_scores):8.3f}")
    print(f"  unanswerable : min {min(neg_scores):8.3f}  max {max(neg_scores):8.3f}")
    print(f"  perfect separation possible: {'no (sets overlap)' if overlap else 'yes'}")
    print(f"  best threshold {threshold:.4f} with separation accuracy {accuracy:.3f}")

    if accuracy < 0.85:
        print("\nSeparation accuracy is below 0.85 - the rerank score alone does")
        print("not cleanly separate answerable from unanswerable on this corpus.")
        print("NOT applying. Consider adding more --neg examples from your real")
        print("domain, or reviewing chunk sizes; then re-run.")
        return 1

    if args.apply:
        apply_to_env(threshold)
        print("Restart the app (or re-import config) for the new threshold to take effect.")
    else:
        print(f"\nDry run. To adopt it:  "
              f"python scripts\\calibrate_abstention.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
