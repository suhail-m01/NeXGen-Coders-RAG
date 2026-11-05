"""
evaluation.py -- reproducible evaluation against a fixed, versioned dataset.

Run:  .venv\\Scripts\\python evaluation.py

Every run writes three artefacts into evaluation/reports/:
    report_<timestamp>.json   full per-query detail plus aggregates
    report_<timestamp>.csv    one row per query
    report_<timestamp>.md     human-readable summary

Each report stamps the dataset version, all four model names, the full
configuration, and the git commit, so two runs can be compared meaningfully.

Regression gate:
    --compare BASELINE.json diffs a run against a saved baseline metric by
    metric and exits non-zero if anything got worse, so a change that quietly
    costs retrieval quality fails instead of being admired in prose. Latency is
    reported but never gated: it is a property of the machine, not the system.
    --save-baseline promotes the run you just made to evaluation/baseline.json.
    Comparing across dataset versions is refused unless you pass
    --allow-dataset-mismatch, because the numbers would not mean the same thing.

Determinism, stated honestly:
    Retrieval, reranking, relationships and citations are deterministic --
    identical inputs give identical outputs on every run.
    Generation runs at temperature 0.0 with a fixed seed (ollama_seed), which
    makes answer-level metrics reproducible run to run on one machine. Ollama
    gives no cross-version seed guarantee, so a different Ollama build can still
    shift them. Set ollama_seed <= 0 to opt out and they become approximate.

Groundedness and correctness are scored by keyword and expected-source overlap.
--judge adds an advisory LLM-judge arm, but it never becomes the score of
record: offline, the judge is the same small local model that wrote the answer,
and at 0.5b it disagrees with itself under option-order swap (see
judge_consistency_rate in the report).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from numeric_guard import last_guard_detail, strip_caveat, strip_enumeration
from observability import build_manifest
from synthesis import strip_completion
from schemas import QueryModality, QueryRequest

DATASET_VERSION = "2.0"
DATASET_PATH = settings.evaluation_dir / "dataset" / "queries.json"
# The baseline a regression gate compares against. Shipped with the project so
# `--compare` works on a fresh clone without hunting for an old report.
BASELINE_PATH = settings.evaluation_dir / "baseline.json"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    """Set precision/recall of retrieved source filenames against expected."""
    if not expected:
        return (1.0 if not retrieved else 0.0), 1.0
    hits = len(set(retrieved) & set(expected))
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(expected)
    return precision, recall


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    """1/rank of the first relevant source; 0 if none appear."""
    for position, filename in enumerate(ranked, start=1):
        if filename in expected:
            return 1.0 / position
    return 0.0


def keyword_coverage(
    answer: str,
    keywords: list[str],
    alternatives: list[list[str]] | None = None,
) -> float:
    """Fraction of required facts present verbatim in the answer.

    Each group in `alternatives` is ONE required fact with several acceptable
    phrasings, so a correct answer that words it differently still scores 1.0.
    Two groups therefore mean two facts, not two options -- which is what makes
    an open-ended query
    measurable without demanding the dataset author's exact wording. A group
    counts as covered when any phrase in it appears.
    """
    groups = alternatives or []
    total = len(keywords) + len(groups)
    if not total:
        return 1.0
    lowered = answer.lower()
    hit_keywords = sum(1 for k in keywords if k.lower() in lowered)
    hit_groups = sum(
        1 for group in groups if any(phrase.lower() in lowered for phrase in group)
    )
    return (hit_keywords + hit_groups) / total


def citation_accuracy(citation_files: list[str], expected: list[str]) -> float:
    """Fraction of citations that point at an expected source."""
    if not citation_files:
        return 0.0
    return sum(1 for f in citation_files if f in expected) / len(citation_files)


def hallucinated_numbers(answer: str, evidence_text: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the supplied evidence.

    A blunt but honest check: it catches the common 99.4 -> 99.44 style
    corruption without pretending to be a semantic entailment model. List
    ordinals are excluded first: a numbered answer's "10." and "11." are
    structure, and counting them as invented figures once scored a correct answer
    as hallucinating seven numbers.
    """
    import re

    answer_numbers = set(re.findall(r"\d+\.\d+|\d{2,}", strip_enumeration(answer)))
    evidence_numbers = set(re.findall(r"\d+\.\d+|\d{2,}", evidence_text))
    return sorted(answer_numbers - evidence_numbers)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def default_dataset() -> dict:
    """The fixed evaluation set, and the source of truth for queries.json.

    Written to disk on first run so later runs use the committed file rather
    than regenerating it. Sixteen queries over the Project Alpha demo corpus:

      Q1-Q6   the original set, unchanged, so old reports stay interpretable
      Q7-Q8   multi-hop: the answer needs a fact from one source joined to a
              fact from another (a table maximum plus the prose reason for it)
      Q9-Q11  adversarial: a near-miss year, a figure that does not exist, and
              an entity that does not exist. These are where an evidence system
              earns its name -- retrieval confidence alone will answer all three
              happily and wrongly.
      Q12     table fidelity against a more salient near-miss number
      Q13-Q14 per-modality pins: one image (OCR) and one audio (ASR) query whose
              evidence exists in only that modality
      Q15-Q16 the long academic deck, which is the largest source in the corpus
              and about a different subject entirely

    Optional per-query fields, all read with .get() so an older queries.json
    keeps working:
      must_not_contain  phrases that must NOT appear in the answer. This is what
                        makes an adversarial query measurable: "did not invent a
                        dollar figure" is checkable, "was appropriately cautious"
                        is not.
      keyword_alternatives  acceptable phrasings of one fact, as a list of groups.
                        Each group is one required fact; any phrase inside it
                        satisfies that fact. Use for open-ended queries, where
                        otherwise the score measures whether the model happened to
                        pick the dataset author's wording.
      tags              grouping for the report (multi-hop, adversarial, ...)
      notes             ground truth and provenance, so a reviewer can check the
                        expectation against the corpus instead of trusting it
    """
    return {
        "dataset_version": DATASET_VERSION,
        "description": "Evidence AI fixed evaluation set over the Project Alpha "
                       "synthetic demo corpus. v2.0 adds multi-hop, adversarial "
                       "and per-modality queries to the original six.",
        "queries": [
            {
                "id": "Q1-pdf",
                "query": "How many records did the ingestion platform process in 2024?",
                "query_modality": "text",
                "expected_sources": ["project_report.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["1.2 million"],
                "tags": ["single-hop"],
                "notes": "Report p.2: 'During 2024 the ingestion platform "
                         "processed 1.2 million records'.",
            },
            {
                "id": "Q2-docx",
                "query": "What is the top open risk carried into 2025?",
                "query_modality": "text",
                "expected_sources": ["project_notes.docx", "project_report.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["latency"],
                "tags": ["single-hop"],
                "notes": "Notes, 'Open Risks': batch pipeline latency under peak "
                         "load is the top risk carried into 2025.",
            },
            {
                "id": "Q3-image",
                "query": "What does the project dashboard show about team size?",
                "query_modality": "text",
                "expected_sources": ["project_dashboard.png"],
                "expected_modality": "image",
                "expect_abstention": False,
                "answer_keywords": ["6"],
                "tags": ["modality:image", "ocr"],
                "notes": "Dashboard tile 'Team size / 6 engineers'. At native "
                         "resolution OCR fused adjacent glyphs and produced "
                         "'22024'; the 2x upscale plus digit-confidence rule is "
                         "what makes this answerable.",
            },
            {
                "id": "Q4-audio",
                "query": "What did the quarterly review recording say about accuracy?",
                "query_modality": "text",
                "expected_sources": ["project_meeting.wav"],
                "expected_modality": "audio",
                "expect_abstention": False,
                "answer_keywords": ["99.4"],
                "tags": ["modality:audio", "asr"],
                "notes": "Transcript segment 1: 'classification accuracy reached "
                         "99.4%'.",
            },
            {
                "id": "Q5-crossmodal",
                "query": "What evidence do we have about Project Alpha's 2024 "
                         "development?",
                "query_modality": "text",
                "expected_sources": [
                    "project_report.pdf", "project_dashboard.png",
                    "project_meeting.wav", "project_notes.docx",
                ],
                "expected_modality": "mixed",
                "expect_abstention": False,
                "answer_keywords": ["1.2 million", "99.4"],
                "tags": ["cross-modal", "synthesis"],
                "notes": "Deliberately broad: every Project Alpha source is "
                         "relevant. This is the query that measures synthesis "
                         "rather than lookup -- a 0.5b model tends to copy the "
                         "first evidence block instead of combining two figures.",
            },
            {
                "id": "Q6-unanswerable",
                "query": "What was Project Alpha's budget in 2040?",
                "query_modality": "text",
                "expected_sources": [],
                "expected_modality": "none",
                "expect_abstention": True,
                "answer_keywords": [],
                "tags": ["adversarial", "out-of-range"],
                "notes": "No 2040 anything exists in the corpus.",
            },
            {
                "id": "Q7-multihop-budget",
                "query": "Which quarter used the most budget, and what explanation "
                         "did finance ask for?",
                "query_modality": "text",
                "expected_sources": [
                    "project_metrics.csv", "project_notes.docx",
                    "project_report.pdf",
                ],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["Q4", "87", "underspend"],
                "tags": ["multi-hop", "table+prose"],
                "notes": "Hop 1 is a table maximum: budget_used_pct peaks at 87 "
                         "in the Q4 2024 CSV row. Hop 2 is prose: the notes say "
                         "finance asked for a written explanation of the "
                         "infrastructure underspend, and report p.3 says the "
                         "underspend was concentrated in infrastructure. Neither "
                         "source alone answers the question.",
            },
            {
                "id": "Q8-multihop-milestones",
                "query": "In which quarter was the NLP module deployed, and how many "
                         "milestones were completed in that quarter?",
                "query_modality": "text",
                "expected_sources": [
                    "project_notes.docx", "project_metrics.csv",
                    "project_dashboard.png",
                ],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["Q2", "4"],
                "tags": ["multi-hop"],
                "notes": "Hop 1 identifies Q2 2024 (dashboard tile, notes table, "
                         "audio 'second quarter of 2024'). Hop 2 counts that "
                         "quarter's milestones: 4, in both the CSV row and the "
                         "notes table. '4' is a weak keyword on its own, which is "
                         "why it is paired with 'Q2'.",
            },
            {
                "id": "Q9-adversarial-year",
                "query": "How many records did the ingestion platform process in 2023?",
                "query_modality": "text",
                "expected_sources": [],
                "expected_modality": "none",
                "expect_abstention": True,
                "answer_keywords": [],
                "must_not_contain": ["1.2 million", "1,200,000", "1200000", "240000"],
                "tags": ["adversarial", "near-miss-year"],
                "notes": "2023 appears in the corpus exactly once, as the launch "
                         "year (report p.1). Every record count belongs to 2024. "
                         "Answering with the 2024 figure means the system matched "
                         "the topic and ignored the constraint.",
            },
            {
                "id": "Q10-absent-figure",
                "query": "What was Project Alpha's total budget in US dollars?",
                "query_modality": "text",
                "expected_sources": ["project_report.pdf", "project_notes.docx"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["87"],
                "must_not_contain": ["$", "USD"],
                "tags": ["adversarial", "absent-figure"],
                "notes": "The corpus gives utilisation as a percentage (87 percent "
                         "of the allocated amount) and never an absolute sum. The "
                         "correct answer reports the percentage and says no dollar "
                         "figure is stated; must_not_contain catches a system that "
                         "manufactures one.",
            },
            {
                "id": "Q11-entity-confusion",
                "query": "How many engineers were on the Project Beta team?",
                "query_modality": "text",
                "expected_sources": [],
                "expected_modality": "none",
                "expect_abstention": True,
                "answer_keywords": [],
                "must_not_contain": ["6 engineers", "six engineers", "6", "six"],
                "tags": ["adversarial", "entity-confusion", "known-hard"],
                "notes": "There is no Project Beta. Marked known-hard on purpose: "
                         "the numeric guard checks figures against evidence but "
                         "nothing checks entity names, so the six-engineer fact "
                         "will probably be served up for Beta -- a first run "
                         "answered simply '6'. The bare number is forbidden too, "
                         "because 'Project Beta had 6' is the same leak in fewer "
                         "words. This query records the gap instead of hiding it.",
            },
            {
                "id": "Q12-table-near-miss",
                "query": "According to the metrics table, what was the classification "
                         "accuracy in Q3 2024?",
                "query_modality": "text",
                "expected_sources": ["project_metrics.csv"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["99.0"],
                "must_not_contain": ["99.4"],
                "tags": ["table-fidelity", "near-miss-number"],
                "notes": "CSV row 3 says 99.0 for Q3 2024. The adjacent Q4 row "
                         "says 99.4, and 99.4 also appears in the report, the "
                         "notes, the dashboard and the recording. Answering 99.4 "
                         "means salience beat the row that was asked for.",
            },
            {
                "id": "Q13-image-deployment",
                "query": "According to the dashboard image, when was the NLP module "
                         "deployed?",
                "query_modality": "text",
                "expected_sources": ["project_dashboard.png"],
                "expected_modality": "image",
                "expect_abstention": False,
                "answer_keywords": ["Q2 2024"],
                "tags": ["modality:image", "ocr"],
                "notes": "Dashboard tile 'NLP module deployed / Q2 2024', read by "
                         "OCR at 90.5% mean word confidence after adaptive "
                         "upscaling.",
            },
            {
                "id": "Q14-audio-budget",
                "query": "What did the quarterly review recording say about budget "
                         "utilisation?",
                "query_modality": "text",
                "expected_sources": ["project_meeting.wav"],
                "expected_modality": "audio",
                "expect_abstention": False,
                "answer_keywords": ["87"],
                "tags": ["modality:audio", "asr"],
                "notes": "Whisper renders the line as 'Budget utilization time at "
                         "87%' -- the verb is wrong, the figure is right -- so the "
                         "keyword is the number rather than the sentence.",
            },
            {
                "id": "Q15-longdoc",
                "query": "Which university is the offline multimodal RAG "
                         "mini-project submitted to?",
                "query_modality": "text",
                "expected_sources": ["MCA_MP_58-mini-project.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["Presidency University"],
                "tags": ["long-document", "cross-corpus"],
                "notes": "Page 1 of the 22-item academic deck: 'Submitted to the "
                         "Presidency University, Bengaluru'. That deck is the "
                         "largest source in the corpus and shares no vocabulary "
                         "with Project Alpha, so this measures whether it can "
                         "still be reached.",
            },
            {
                "id": "Q16-paper-limitation",
                "query": "What limitation of text-centric retrieval does the "
                         "mini-project paper identify?",
                "query_modality": "text",
                "expected_sources": ["MCA_MP_58-mini-project.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": [],
                # ONE fact, several acceptable phrasings. Two groups would mean
                # two required facts and halve the score for a correct answer.
                "keyword_alternatives": [
                    ["attribution", "page-level", "timestamp-level",
                     "visual", "audiovisual", "images and audio"],
                ],
                "tags": ["long-document", "cross-corpus", "open-ended"],
                "notes": "Page 3 gives two related criticisms of text-centric "
                         "retrieval: answers 'often lack accurate page-level, "
                         "document-level or timestamp-level source attribution', "
                         "and text-only retrieval ignores visual and audiovisual "
                         "material. Either is a correct answer, so both are "
                         "accepted through keyword_alternatives. Blaming cloud "
                         "dependency is not: that is a different criticism the "
                         "deck makes about generative AI in general.",
            },
        ],
    }


def load_dataset() -> dict:
    if DATASET_PATH.exists():
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = default_dataset()
    DATASET_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    _write_expectations(dataset)
    return dataset


def _write_expectations(dataset: dict) -> None:
    """Mirror expected answers and sources into their own directories, as
    the specification requires them to be separately inspectable."""
    answers_dir = settings.evaluation_dir / "expected_answers"
    sources_dir = settings.evaluation_dir / "expected_sources"
    answers_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    for query in dataset["queries"]:
        (answers_dir / f"{query['id']}.json").write_text(
            json.dumps({
                "id": query["id"],
                "answer_keywords": query["answer_keywords"],
                "expect_abstention": query["expect_abstention"],
            }, indent=2), encoding="utf-8",
        )
        (sources_dir / f"{query['id']}.json").write_text(
            json.dumps({
                "id": query["id"],
                "expected_sources": query["expected_sources"],
                "expected_modality": query["expected_modality"],
            }, indent=2), encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def git_commit() -> str:
    """Short commit hash, or 'unversioned' when the tree has no git history.

    Delegates to observability.git_commit so the harness and the API report the
    same commit for the same tree; only the fallback wording differs, and reports
    already committed to it.
    """
    from observability import git_commit as _commit

    value = _commit()
    return value if value != "unknown" else "unversioned"


def run_evaluation(judge: bool = False) -> dict:
    from rag_pipeline import get_pipeline

    dataset = load_dataset()
    pipeline = get_pipeline()
    per_query: list[dict] = []

    print(f"Evidence AI evaluation -- dataset v{dataset['dataset_version']}")
    print("=" * 78)

    for spec in dataset["queries"]:
        started = time.perf_counter()
        response = pipeline.query(QueryRequest(
            query=spec["query"],
            top_k=settings.top_k,
            rerank_top_k=settings.rerank_top_k,
            query_modality=QueryModality(spec["query_modality"]),
        ))
        latency = time.perf_counter() - started

        retrieved_files, seen = [], set()
        for result in response.retrieved_items:
            source = pipeline.store.get_source(result.item.source_id)
            name = source.filename if source else "unknown"
            if name not in seen:
                seen.add(name)
                retrieved_files.append(name)

        citation_files = [c.filename for c in response.citations]
        expected = spec["expected_sources"]

        precision, recall = precision_recall(retrieved_files, expected)
        mrr = reciprocal_rank(retrieved_files, expected)
        correctness = keyword_coverage(
            response.answer,
            spec["answer_keywords"],
            spec.get("keyword_alternatives"),
        )
        cite_accuracy = (
            1.0 if spec["expect_abstention"] and not citation_files
            else citation_accuracy(citation_files, expected)
        )
        evidence_text = " ".join(r.item.content for r in response.retrieved_items)
        # The guard's caveat is a disclaimer, not a claim, so it is stripped
        # before scoring what the system actually asserted. What it flagged is
        # reported separately rather than hidden.
        flagged, _ = last_guard_detail()
        # The completion line is stripped here for the same reason the caveat is:
        # its parentheticals are source locators, not asserted figures. Scoring
        # them would measure the label format instead of honesty -- a clock range
        # in a locator once scored a correct answer as hallucinating "00" and
        # "21". answer_correctness above still sees the full answer.
        invented = (
            [] if response.abstained
            else hallucinated_numbers(
                strip_completion(strip_caveat(response.answer)), evidence_text
            )
        )
        abstention_ok = response.abstained == spec["expect_abstention"]

        # Adversarial queries are only measurable if "did not invent X" is a
        # checkable claim, so the dataset can name phrases that must not appear.
        # Scored against the caveat-stripped answer: the guard's disclaimer is
        # not an assertion, and quoting a forbidden figure inside a warning
        # would be a false positive.
        forbidden = [
            phrase for phrase in spec.get("must_not_contain", [])
            if phrase.lower() in strip_caveat(response.answer).lower()
        ]

        if judge:
            # Advisory only. Abstentions are not judged: there is no answer to
            # score, and counting a correct refusal as "ungrounded" would be
            # nonsense.
            from llm_judge import judge_answer

            if response.abstained:
                record_judge = {
                    "judge_grounded": None, "judge_answers_question": None,
                    "judge_conflicts": None, "judge_quality": None,
                    "judge_consistency": None,
                    "judge_parse_failures": [], "judge_raw": {},
                }
            else:
                record_judge = judge_answer(
                    spec["query"], response.answer, evidence_text
                ).as_record()

        record = {
            "id": spec["id"],
            "query": spec["query"],
            "expected_sources": expected,
            "retrieved_sources": retrieved_files,
            "citation_sources": citation_files,
            "modalities": sorted({r.item.modality.value for r in response.retrieved_items}),
            "abstained": response.abstained,
            "expect_abstention": spec["expect_abstention"],
            "abstention_correct": abstention_ok,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mrr": round(mrr, 4),
            "citation_accuracy": round(cite_accuracy, 4),
            "answer_correctness": round(correctness, 4),
            "groundedness": round(1.0 if not invented else 0.0, 4),
            "hallucinated_numbers": invented,
            "unverified_numbers": flagged,
            "forbidden_hits": forbidden,
            "expected_modality": spec["expected_modality"],
            "tags": spec.get("tags", []),
            "notes": spec.get("notes", ""),
            "confidence": response.confidence,
            "latency_s": round(latency, 2),
            "answer": response.answer,
        }
        if judge:
            record.update(record_judge)
        per_query.append(record)

        status = (
            "PASS" if abstention_ok and correctness >= 0.5 and not forbidden
            else "CHECK"
        )
        guard = f" guard={','.join(flagged)}" if flagged else ""
        if forbidden:
            guard += f" LEAK={','.join(forbidden)}"
        if judge:
            quality = record_judge.get("judge_quality")
            guard += f" judge={quality if quality is not None else 'n/a'}"
        print(f"[{status}] {spec['id']:18s} P={precision:.2f} R={recall:.2f} "
              f"MRR={mrr:.2f} corr={correctness:.2f} "
              f"cite={cite_accuracy:.2f}{guard} {latency:.1f}s")

    def mean(field: str) -> float:
        values = [r[field] for r in per_query if isinstance(r[field], (int, float))]
        return round(sum(values) / len(values), 4) if values else 0.0

    aggregate = {
        "retrieval_precision": mean("precision"),
        "retrieval_recall": mean("recall"),
        "mrr": mean("mrr"),
        "citation_accuracy": mean("citation_accuracy"),
        "answer_correctness": mean("answer_correctness"),
        "groundedness": mean("groundedness"),
        "hallucination_rate": round(
            sum(1 for r in per_query if r["hallucinated_numbers"]) / len(per_query), 4
        ),
        "abstention_accuracy": round(
            sum(1 for r in per_query if r["abstention_correct"]) / len(per_query), 4
        ),
        # How often the numeric guard had to caveat an answer. Not a quality
        # score on its own: a higher rate can mean better honesty or worse OCR.
        "numeric_guard_rate": round(
            sum(1 for r in per_query if r["unverified_numbers"]) / len(per_query), 4
        ),
        # Adversarial queries: how often the answer contained a phrase the
        # dataset says must never appear (a wrong year's figure, an invented
        # currency amount, another project's headcount).
        "forbidden_leak_rate": round(
            sum(1 for r in per_query if r["forbidden_hits"]) / len(per_query), 4
        ),
        "mean_latency_s": mean("latency_s"),
    }

    # Per-modality means. An aggregate over a mixed set can hide a modality that
    # quietly collapsed -- image queries failing while text carries the average is
    # exactly the regression a single mean conceals.
    per_modality: dict[str, dict] = {}
    for modality in sorted({r["expected_modality"] for r in per_query}):
        rows = [r for r in per_query if r["expected_modality"] == modality]

        def mod_mean(field: str, rows: list[dict] = rows) -> float:
            values = [r[field] for r in rows if isinstance(r[field], (int, float))]
            return round(sum(values) / len(values), 4) if values else 0.0

        per_modality[modality] = {
            "queries": len(rows),
            "answer_correctness": mod_mean("answer_correctness"),
            "citation_accuracy": mod_mean("citation_accuracy"),
            "retrieval_recall": mod_mean("recall"),
            "groundedness": mod_mean("groundedness"),
            "abstention_accuracy": round(
                sum(1 for r in rows if r["abstention_correct"]) / len(rows), 4
            ),
            "forbidden_leak_rate": round(
                sum(1 for r in rows if r["forbidden_hits"]) / len(rows), 4
            ),
            "pass_rate": round(
                sum(
                    1 for r in rows
                    if r["abstention_correct"] and r["answer_correctness"] >= 0.5
                    and not r["forbidden_hits"]
                ) / len(rows), 4
            ),
        }

    # What the set actually covers, so a gap is visible rather than assumed away.
    tag_counts: dict[str, int] = {}
    for record in per_query:
        for tag in record["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    dataset_coverage = {
        "queries": len(per_query),
        "by_expected_modality": {
            modality: block["queries"] for modality, block in per_modality.items()
        },
        "by_tag": dict(sorted(tag_counts.items())),
        "sources_covered": sorted(
            {name for r in per_query for name in r["expected_sources"]}
        ),
    }

    if judge:
        def rate(field: str):
            values = [r[field] for r in per_query if isinstance(r[field], bool)]
            return round(sum(1 for v in values if v) / len(values), 4) if values else None

        qualities = [
            r["judge_quality"] for r in per_query
            if isinstance(r["judge_quality"], int)
        ]
        consistencies = [
            r["judge_consistency"] for r in per_query
            if isinstance(r["judge_consistency"], float)
        ]
        judged = [r for r in per_query if not r["abstained"]]
        # Read this one first: it says whether the judge was listening.
        consistency = (
            round(sum(consistencies) / len(consistencies), 4) if consistencies else None
        )
        aggregate.update({
            "judge_consistency_rate": consistency,
            "judge_grounded_rate": rate("judge_grounded"),
            "judge_answers_question_rate": rate("judge_answers_question"),
            "judge_conflict_rate": rate("judge_conflicts"),
            "judge_quality_mean": (
                round(sum(qualities) / len(qualities), 2) if qualities else None
            ),
            "judge_answers_scored": sum(
                1 for r in judged if r["judge_grounded"] is not None
            ),
            "judge_answers_asked": len(judged),
        })
        if consistency is not None and consistency < 0.6:
            aggregate["judge_verdict"] = (
                "UNUSABLE at this model size: the judge disagreed with itself "
                f"under option-order swap on {1 - consistency:.0%} of questions, "
                "so its scores are position bias, not judgement. Use a larger "
                "local model (3B+) for the judge arm."
            )

    return {
        "dataset_version": dataset["dataset_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        # Self-describing runs: the exact build, configuration and index behind
        # every number in this report, identical to what GET /manifest serves.
        "manifest": build_manifest(index_stats=pipeline.store.stats()),
        "models": {
            "llm": settings.ollama_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "whisper": f"{settings.whisper_model_size} "
                       f"({settings.whisper_device}/{settings.whisper_compute_type})",
        },
        "configuration": {
            "top_k": settings.top_k,
            "rerank_top_k": settings.rerank_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "related_to_threshold": settings.related_to_threshold,
            "citation_threshold": settings.citation_threshold,
            "abstention_threshold": settings.abstention_threshold,
            "abstention_threshold_image": settings.abstention_threshold_image,
            "coverage_guard_enabled": settings.coverage_guard_enabled,
            "numeric_guard_enabled": settings.numeric_guard_enabled,
            "ocr_confidence_enabled": settings.ocr_confidence_enabled,
            "ocr_min_word_confidence": settings.ocr_min_word_confidence,
            "ocr_min_digit_confidence": settings.ocr_min_digit_confidence,
            "hybrid_enabled": settings.hybrid_enabled,
            "mmr_enabled": settings.mmr_enabled,
            "diarization_enabled": settings.diarization_enabled,
            "streaming_enabled": settings.streaming_enabled,
            "max_context_items": settings.max_context_items,
            "temperature": settings.ollama_temperature,
            "seed": settings.ollama_seed,
            "num_ctx": settings.ollama_num_ctx,
        },
        "index": pipeline.store.stats(),
        "aggregate": aggregate,
        "per_modality": per_modality,
        "dataset_coverage": dataset_coverage,
        "per_query": per_query,
        "judge_note": (
            "Advisory only. Offline, the judge is the same local model that "
            "wrote the answer, so self-preference cannot be ruled out. Every "
            "binary question is asked twice with the option order swapped and "
            "only counts when both replies pick the same statement, which turns "
            "position bias into a measurable consistency rate instead of a "
            "flattering score. Read judge_consistency_rate first: if it is low "
            "the model is too small to judge and the other judge columns are "
            "noise. Deterministic metrics (keyword coverage, number checks, "
            "retrieval set math) remain the score of record."
            if judge else
            "LLM judge not run. Pass --judge to add advisory judge columns."
        ),
        "determinism_note": (
            "Retrieval, reranking, relationships and citations are deterministic. "
            "Generation runs at temperature 0.0 with a fixed seed (ollama_seed), "
            "which makes answer-level metrics reproducible run-to-run; set "
            "ollama_seed <= 0 to opt out and they become approximate again."
        ),
    }


# Gated metrics and which direction is better. Latency is deliberately absent:
# it measures the machine, and gating on it would fail a correct change run on a
# slower laptop. numeric_guard_rate is informational for the same reason it is
# not a quality score -- a higher rate can mean better honesty or worse OCR.
GATED_METRICS: dict[str, bool] = {
    "retrieval_precision": True,
    "retrieval_recall": True,
    "mrr": True,
    "citation_accuracy": True,
    "answer_correctness": True,
    "groundedness": True,
    "abstention_accuracy": True,
    "hallucination_rate": False,
    "forbidden_leak_rate": False,
}
PER_MODALITY_GATED: dict[str, bool] = {
    "answer_correctness": True,
    "citation_accuracy": True,
    "retrieval_recall": True,
    "groundedness": True,
    "pass_rate": True,
    "forbidden_leak_rate": False,
}
INFORMATIONAL_METRICS = ("numeric_guard_rate", "mean_latency_s")
PER_QUERY_GATED = ("answer_correctness", "citation_accuracy", "recall", "groundedness")


def _verdict(baseline: float, current: float, higher_is_better: bool,
             tolerance: float) -> str:
    delta = current - baseline
    if abs(delta) <= tolerance + 1e-9:
        return "unchanged"
    improved = delta > 0 if higher_is_better else delta < 0
    return "improved" if improved else "REGRESSION"


def compare_reports(current: dict, baseline: dict, tolerance: float = 0.0,
                    strict_per_query: bool = False) -> dict:
    """Diff this run against a saved baseline and decide pass/fail.

    Anything worse than the baseline beyond `tolerance` is a regression. With a
    fixed seed the deterministic metrics are reproducible, so the default
    tolerance is 0.0: a change either held the line or it did not.
    """
    rows: list[dict] = []
    regressions: list[str] = []

    base_agg = baseline.get("aggregate", {})
    cur_agg = current.get("aggregate", {})
    for metric, higher_is_better in GATED_METRICS.items():
        if metric not in base_agg or metric not in cur_agg:
            continue
        base_value = float(base_agg[metric])
        cur_value = float(cur_agg[metric])
        verdict = _verdict(base_value, cur_value, higher_is_better, tolerance)
        rows.append({
            "metric": metric, "baseline": base_value, "current": cur_value,
            "delta": round(cur_value - base_value, 4), "verdict": verdict,
            "higher_is_better": higher_is_better, "scope": "aggregate",
        })
        if verdict == "REGRESSION":
            regressions.append(f"{metric}: {base_value} -> {cur_value}")

    for metric in INFORMATIONAL_METRICS:
        if metric in base_agg and metric in cur_agg:
            rows.append({
                "metric": f"{metric} (informational)",
                "baseline": float(base_agg[metric]), "current": float(cur_agg[metric]),
                "delta": round(float(cur_agg[metric]) - float(base_agg[metric]), 4),
                "verdict": "not gated", "higher_is_better": None, "scope": "aggregate",
            })

    base_mod = baseline.get("per_modality", {})
    cur_mod = current.get("per_modality", {})
    for modality in sorted(set(base_mod) & set(cur_mod)):
        for metric, higher_is_better in PER_MODALITY_GATED.items():
            if metric not in base_mod[modality] or metric not in cur_mod[modality]:
                continue
            base_value = float(base_mod[modality][metric])
            cur_value = float(cur_mod[modality][metric])
            verdict = _verdict(base_value, cur_value, higher_is_better, tolerance)
            rows.append({
                "metric": f"{modality}/{metric}", "baseline": base_value,
                "current": cur_value,
                "delta": round(cur_value - base_value, 4), "verdict": verdict,
                "higher_is_better": higher_is_better, "scope": "modality",
            })
            if verdict == "REGRESSION":
                regressions.append(
                    f"{modality}/{metric}: {base_value} -> {cur_value}"
                )

    base_by_id = {r["id"]: r for r in baseline.get("per_query", [])}
    per_query_regressions: list[str] = []
    for record in current.get("per_query", []):
        before = base_by_id.get(record["id"])
        if before is None:
            per_query_regressions.append(f"{record['id']}: new query, no baseline")
            continue
        for metric in PER_QUERY_GATED:
            delta = float(record.get(metric, 0.0)) - float(before.get(metric, 0.0))
            if delta < -tolerance - 1e-9:
                per_query_regressions.append(
                    f"{record['id']} {metric}: {before.get(metric)} -> {record.get(metric)}"
                )
        if record.get("forbidden_hits") and not before.get("forbidden_hits"):
            per_query_regressions.append(
                f"{record['id']} leaked {record['forbidden_hits']} (baseline was clean)"
            )
        if before.get("abstention_correct") and not record.get("abstention_correct"):
            per_query_regressions.append(f"{record['id']} abstention behaviour broke")

    gated_failures = len(regressions)
    if strict_per_query:
        gated_failures += len([
            item for item in per_query_regressions if "new query" not in item
        ])

    missing = [qid for qid in base_by_id if qid not in {r["id"] for r in current["per_query"]}]
    headline = (
        f"Baseline `{baseline.get('timestamp', '?')}` (dataset v"
        f"{baseline.get('dataset_version')}, commit {baseline.get('git_commit')}) "
        f"vs this run. Tolerance {tolerance}. "
        + (f"FAIL: {gated_failures} gated regression(s)." if gated_failures
           else "PASS: no gated metric regressed.")
    )
    return {
        "baseline_timestamp": baseline.get("timestamp"),
        "baseline_git_commit": baseline.get("git_commit"),
        "baseline_dataset_version": baseline.get("dataset_version"),
        "tolerance": tolerance,
        "strict_per_query": strict_per_query,
        "rows": rows,
        "regressions": regressions,
        "per_query_regressions": per_query_regressions,
        "queries_missing_from_this_run": missing,
        "gated_failures": gated_failures,
        "passed": gated_failures == 0,
        "headline": headline,
    }


def write_reports(report: dict) -> dict[str, Path]:
    reports_dir = settings.evaluation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"report_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = reports_dir / f"report_{stamp}.csv"
    fields = [
        "id", "expected_modality", "tags", "abstained", "expect_abstention",
        "abstention_correct", "precision", "recall", "mrr", "citation_accuracy",
        "answer_correctness", "groundedness", "hallucinated_numbers",
        "unverified_numbers", "forbidden_hits", "latency_s",
        "judge_consistency", "judge_grounded", "judge_answers_question",
        "judge_conflicts", "judge_quality",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in report["per_query"]:
            row = dict(record)
            for field in ("tags", "hallucinated_numbers", "unverified_numbers",
                          "forbidden_hits", "expected_sources", "retrieved_sources",
                          "citation_sources"):
                if isinstance(row.get(field), list):
                    row[field] = "; ".join(str(v) for v in row[field])
            writer.writerow(row)

    md_path = reports_dir / f"report_{stamp}.md"
    aggregate = report["aggregate"]
    lines = [
        "# Evidence AI Evaluation Report", "",
        f"- Dataset version: **{report['dataset_version']}**",
        f"- Timestamp (UTC): {report['timestamp']}",
        f"- Git commit: `{report['git_commit']}`", "",
        "## Models", "",
        *[f"- {k}: `{v}`" for k, v in report["models"].items()], "",
        "## Configuration", "",
        *[f"- {k}: `{v}`" for k, v in report["configuration"].items()], "",
        "## Aggregate metrics", "",
        "| Metric | Value |", "| --- | --- |",
        *[f"| {k.replace('_', ' ').title()} | {v} |" for k, v in aggregate.items()], "",
        "## Per-modality", "",
        "An aggregate over a mixed set can hide one modality collapsing.", "",
        "| Expected modality | Qs | Correct | Cite | Recall | Ground | Abstain | Leaks | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for modality, block in report.get("per_modality", {}).items():
        lines.append(
            f"| {modality} | {block['queries']} | {block['answer_correctness']} | "
            f"{block['citation_accuracy']} | {block['retrieval_recall']} | "
            f"{block['groundedness']} | {block['abstention_accuracy']} | "
            f"{block['forbidden_leak_rate']} | {block['pass_rate']} |"
        )
    coverage = report.get("dataset_coverage", {})
    if coverage:
        lines += [
            "", "## Dataset coverage", "",
            *[f"- {k.replace('_', ' ').title()}: {v}" for k, v in coverage.items()],
            "",
        ]
    lines += [
        "## Per-query results", "",
        "| ID | Mod | Tags | Abstain (exp) | P | R | MRR | Cite | Correct | Ground | Guard | Leaks | s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["per_query"]:
        lines.append(
            f"| {record['id']} | {record['expected_modality']} | "
            f"{', '.join(record['tags']) or '-'} | {record['abstained']} "
            f"({record['expect_abstention']}) | {record['precision']} | "
            f"{record['recall']} | {record['mrr']} | {record['citation_accuracy']} | "
            f"{record['answer_correctness']} | {record['groundedness']} | "
            f"{', '.join(record['unverified_numbers']) or '-'} | "
            f"{', '.join(record['forbidden_hits']) or '-'} | "
            f"{record['latency_s']} |"
        )
    notes = [(r["id"], r["notes"]) for r in report["per_query"] if r.get("notes")]
    if notes:
        lines += ["", "## Ground truth notes", ""]
        lines += [f"- **{qid}** -- {note}" for qid, note in notes]
    judge_keys = [k for k in aggregate if k.startswith("judge_")]
    if judge_keys:
        lines += [
            "", "## LLM judge (advisory)", "",
            report.get("judge_note", ""), "",
            "| Metric | Value |", "| --- | --- |",
            *[f"| {k.replace('_', ' ').title()} | {aggregate[k]} |" for k in judge_keys],
            "",
        ]

    comparison = report.get("comparison")
    if comparison:
        lines += ["", "## Regression gate", "", comparison.get("headline", "")]
        if comparison.get("rows"):
            lines += [
                "", "| Metric | Baseline | This run | Delta | Verdict |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| {row['metric']} | {row['baseline']} | {row['current']} | "
                f"{row['delta']:+} | {row['verdict']} |"
                for row in comparison["rows"]
            ]
        if comparison.get("per_query_regressions"):
            lines += ["", "Queries that got worse:", ""]
            lines += [f"- {item}" for item in comparison["per_query_regressions"]]
        lines.append("")

    lines += ["", "## Determinism", "", report["determinism_note"], ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge", action="store_true",
        help="Also score each answer with the local model as an LLM judge. "
             "Advisory only, and slower: four extra completions per answer.",
    )
    parser.add_argument(
        "--compare", nargs="?", const=str(BASELINE_PATH), default=None,
        metavar="BASELINE_JSON",
        help=f"Diff this run against a saved report and fail on any regression. "
             f"With no path, uses {BASELINE_PATH.name}.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.0,
        help="How much a gated metric may fall before it counts as a regression "
             "(default 0.0 -- the deterministic metrics are reproducible, so any "
             "drop is real).",
    )
    parser.add_argument(
        "--strict-per-query", action="store_true",
        help="Also fail when an individual query regresses, not just aggregates.",
    )
    parser.add_argument(
        "--allow-dataset-mismatch", action="store_true",
        help="Gate anyway when the baseline was produced by a different dataset "
             "version. The numbers will not mean the same thing.",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help=f"Promote this run to {BASELINE_PATH.name} for future --compare runs.",
    )
    args = parser.parse_args()

    report = run_evaluation(judge=args.judge)

    exit_code = 0
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            print(f"\nNo baseline at {baseline_path}.")
            print("Run once with --save-baseline to create it.")
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        same_dataset = baseline.get("dataset_version") == report["dataset_version"]
        if not same_dataset and not args.allow_dataset_mismatch:
            print("\n" + "=" * 78)
            print(f"REFUSING TO GATE: the baseline is dataset v"
                  f"{baseline.get('dataset_version')} and this run is v"
                  f"{report['dataset_version']}.")
            print("Different query sets produce different aggregates, so a "
                  "pass/fail verdict across them would be meaningless.")
            print("Re-baseline with --save-baseline, or force it with "
                  "--allow-dataset-mismatch.")
            return 2
        comparison = compare_reports(
            report, baseline, tolerance=args.tolerance,
            strict_per_query=args.strict_per_query,
        )
        report["comparison"] = comparison

        print("\n" + "=" * 78)
        print(f"REGRESSION GATE vs {baseline_path.name} "
              f"({comparison['baseline_timestamp']}, commit "
              f"{comparison['baseline_git_commit']})")
        print("=" * 78)
        print(f"  {'METRIC':34s} {'BASELINE':>9} {'NOW':>9} {'DELTA':>9}  VERDICT")
        for row in comparison["rows"]:
            print(f"  {row['metric']:34s} {row['baseline']:>9} {row['current']:>9} "
                  f"{row['delta']:>+9}  {row['verdict']}")
        if comparison["regressions"]:
            print("\n  Gated regressions:")
            for item in comparison["regressions"]:
                print(f"    - {item}")
        if comparison["per_query_regressions"]:
            print("\n  Per-query regressions"
                  f"{' (gated)' if args.strict_per_query else ' (reported only)'}:")
            for item in comparison["per_query_regressions"]:
                print(f"    - {item}")
        if comparison["queries_missing_from_this_run"]:
            print("\n  [WARN] baseline queries absent from this run: "
                  + ", ".join(comparison["queries_missing_from_this_run"]))
        print(f"\n  {comparison['headline']}")
        exit_code = 0 if comparison["passed"] else 1

    paths = write_reports(report)

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nBaseline saved: {BASELINE_PATH}")
        print("Future runs: .venv\\Scripts\\python evaluation.py --compare")

    print("\n" + "=" * 78)
    print("AGGREGATE")
    for key, value in report["aggregate"].items():
        print(f"  {key:24s} {value}")
    if report.get("per_modality"):
        print("\nPER-MODALITY")
        for modality, block in report["per_modality"].items():
            print(f"  {modality:8s} n={block['queries']:<3d} "
                  f"correct={block['answer_correctness']:<7} "
                  f"cite={block['citation_accuracy']:<7} "
                  f"pass={block['pass_rate']}")
    print("\nReports written:")
    for label, path in paths.items():
        print(f"  {label:9s} {path}")
    if exit_code:
        print("\nGate FAILED -- see the regressions above.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
