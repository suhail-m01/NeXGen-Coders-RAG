"""Offline self-test for the evidence-integrity machinery.

    python scripts/selftest.py            # run everything
    python scripts/selftest.py -v         # print each check as it passes

No LLM, no Ollama, no network and no models: every check here runs on strings and
on a throwaway index in a temporary directory, so it finishes in a few seconds on
any machine that can import the project. That is deliberate. The evaluation
harness measures answer quality and needs Ollama; this proves the parts that must
never be wrong regardless of which model is attached -- the guards that refuse a
wrong-period or wrong-entity answer, the chunking that keeps a table whole, the
trace and manifest an auditor reads.

Each check is a behaviour that was measured against a real failure at some point;
the comment above it says which one. Exit code is 0 when everything passes and 1
otherwise, so it can gate a deploy or a demo.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# A throwaway index before anything reads settings, so the self-test can never
# touch the real one.
_TMP = Path(tempfile.mkdtemp(prefix="evidence-selftest-"))
os.environ.setdefault("INDEX_DIR", str(_TMP / "index"))
os.environ.setdefault("PYTHONPATH", os.environ.get("PYTHONPATH", ""))

CHECKS: list[tuple[str, bool, str]] = []
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    if VERBOSE:
        print(f"  [{'ok ' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def section(title: str) -> None:
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# Fixtures: the smallest objects that behave like retrieved evidence
# ---------------------------------------------------------------------------

def _items():
    from schemas import ContentItem, Modality, SourceLocation

    def make(item_id, content, heading=(), page=None, start=None, end=None,
             modality=Modality.TEXT, source_id="src-report"):
        return ContentItem(
            item_id=item_id, source_id=source_id, modality=modality, content=content,
            location=SourceLocation(
                page_number=page, timestamp_start=start, timestamp_end=end
            ),
            metadata={"heading_path": list(heading), "source_type": "pdf"},
        )

    return make


def _results(items):
    from schemas import RerankedResult

    return [
        RerankedResult(item=item, retrieval_score=0.5, rerank_score=1.0 - i * 0.1, rank=i)
        for i, item in enumerate(items)
    ]


class _FakeStore:
    def __init__(self, parents):
        self._parents = {parent.item_id: parent for parent in parents}

    def get_parent(self, parent_id):
        return self._parents.get(parent_id)


# ---------------------------------------------------------------------------
# constraint_guard: period and entity integrity
# ---------------------------------------------------------------------------

def test_constraint_guard() -> None:
    section("constraint_guard -- period and entity checks")
    from constraint_guard import (
        entity_mismatch, periods_in, period_keys, reorder_by_period_keys,
        temporal_violation,
    )

    make = _items()

    # "third quarter of 2024" must yield the paired key, which a literal
    # "Q3 2024" regex missed: the pair is found by proximity, not by wording.
    keys = period_keys("What was the accuracy in the third quarter of 2024?")
    check("ordinal quarter pairs with its year", "Q3 2024" in keys, str(keys))
    check("reversed wording pairs too",
          "Q3 2024" in period_keys("In fiscal 2024, Q3 accuracy?"))
    # Pairing is by proximity (40 characters), so a quarter and a year in the
    # same sentence do pair; only genuinely distant ones must not.
    check("quarter and year in one clause pair",
          "Q1 2024" in period_keys("Q1 results were strong. Later, in 2024, budget fell."))
    check("quarter and year far apart do not pair",
          "Q1 2024" not in period_keys(
              "Q1 results were strong across every region and every product line "
              "that reported. Much later, in 2024, the budget fell."))
    check("periods_in normalises", {"Q3 2024", "Q3", "2024"} <= periods_in("Q3 2024 accuracy"))

    # Q11: "Project Beta" asked, only Alpha in the corpus -> refuse.
    mismatch = entity_mismatch(
        "How many engineers were on the Project Beta team?",
        ["Project Alpha was staffed by a team of six engineers."],
    )
    check("entity near-miss fires", mismatch is not None and mismatch.asked == "Project Beta",
          str(mismatch))
    check("entity guard silent when the entity is present",
          entity_mismatch("How many engineers on Project Alpha?",
                          ["Project Alpha had six engineers."]) is None)
    check("entity guard silent with no sibling entity",
          entity_mismatch("How many engineers on Project Gamma?",
                          ["The team had six engineers."]) is None)

    # Q9: 2023 asked, 2024's figure answered -> refuse.
    violation = temporal_violation(
        "How many records were processed in 2023?",
        "The platform processed 1.2 million records.",
        ["During 2024 the platform processed 1.2 million records."],
    )
    check("temporal violation fires on wrong-period figure", violation is not None,
          str(violation))
    check("temporal guard silent for the right period",
          temporal_violation("How many records in 2024?", "1.2 million records.",
                             ["During 2024 the platform processed 1.2 million records."]) is None)
    check("temporal guard silent when no figure is asserted",
          temporal_violation("How many records in 2023?", "The sources do not say.",
                             ["During 2024 the platform processed 1.2 million records."]) is None)
    check("temporal guard leaves absent figures to the numeric guard",
          temporal_violation("How many records in 2023?", "It processed 999 records.",
                             ["During 2024 the platform processed 1.2 million records."]) is None)

    # Q12: the right row reranks below a page with a more famous figure. Order
    # only -- nothing added, removed or rescored.
    row = make("row3", "Q3 2024, accuracy 99.0")
    page = make("page2", "Accuracy reached 99.4% during the year.", page=2)
    other = make("row4", "Q4 2024, accuracy 99.4")
    results = _results([page, row, other])
    reordered, detail = reorder_by_period_keys(
        results, "According to the table, what was the accuracy in Q3 2024?"
    )
    check("period targeting promotes the matching row",
          reordered[0].item.item_id == "row3", detail.message)
    check("period targeting keeps every item",
          sorted(r.item.item_id for r in reordered) == ["page2", "row3", "row4"])
    check("period targeting does not rescore",
          {r.item.item_id: round(r.rerank_score, 6) for r in reordered}
          == {r.item.item_id: round(r.rerank_score, 6) for r in results})
    untouched, noop = reorder_by_period_keys(results, "What is the top open risk?")
    check("no period named -> list returned unchanged", untouched is results and not noop.message)


# ---------------------------------------------------------------------------
# synthesis: evidence completion for broad questions
# ---------------------------------------------------------------------------

def test_synthesis() -> None:
    section("synthesis -- completion for multi-source questions")
    from synthesis import (
        complete_answer, figures_in, is_synthesis_query, shared_figures,
        strip_completion,
    )

    make = _items()
    check("figures keep units", [f.render() for f in figures_in("1.2 million at 99.4% and 87 %")]
          == ["1.2 million", "99.4%", "87%"])
    check("bare years are not figures", figures_in("launched in 2023 and grew in 2024") == [])
    check("separators survive for grouping",
          {f.value for f in figures_in("330,000 and 330000")} == {"330000"})
    check("ordinal markers are not figures",
          all(f.raw != "3" for f in figures_in("Q3 2024 accuracy")))

    check("synthesis trigger is narrow",
          is_synthesis_query("What evidence do we have about 2024?")
          and not is_synthesis_query("What was the accuracy in Q3 2024?"))

    evidence = _results([
        make("a", "processed 1.2 million records at 99.4% accuracy", source_id="s1"),
        make("b", "1.2 million records, accuracy 99.4%, budget 87%", source_id="s2"),
        make("c", "a figure only here: 42.5", source_id="s3"),
    ])
    names = {"s1": "project_report.pdf", "s2": "project_meeting.wav", "s3": "notes.docx"}
    shared = shared_figures(evidence, names)
    check("only corroborated figures are shared",
          {f.surface for f in shared} == {"1.2 million", "99.4%"}, str([f.surface for f in shared]))

    boiler = "The report covers the 2024 development cycle and its governance."
    completed, detail = complete_answer(boiler, evidence, names,
                                        "What evidence do we have about 2024?")
    check("missing corroborated figures are appended",
          "1.2 million" in completed and "99.4" in completed and detail.reason == "appended",
          detail.message)
    check("appended line names its sources", "project_report.pdf" in completed)
    check("appended line carries no clock digits", "00:" not in completed)
    again, second = complete_answer(completed, evidence, names,
                                    "What evidence do we have about 2024?")
    check("completion never repeats itself", second.reason == "complete" and again == completed)
    lookup, third = complete_answer(boiler, evidence, names, "What was the accuracy in Q3 2024?")
    check("lookup questions are untouched", lookup == boiler and third.reason == "no_marker")
    check("strip_completion removes only the appended line",
          strip_completion(completed) == boiler)
    check("single-source evidence is left alone",
          complete_answer(boiler, evidence[:1], names,
                          "What evidence do we have?")[1].reason == "single_source")


# ---------------------------------------------------------------------------
# repetition_guard and numeric_guard
# ---------------------------------------------------------------------------

def test_repetition_and_numeric() -> None:
    section("repetition_guard + numeric_guard -- looping output and false warnings")
    from numeric_guard import apply_guard, reset_guard_detail, strip_enumeration
    from repetition_guard import collapse_repetition

    loop = "\n\n".join(
        ["The paper lists these limitations:"]
        + [f"{i}. **Privacy Risk**: files may need to leave the local infrastructure."
           if i % 2 else f"{i}. **Explainability**: answers lack page-level attribution."
           for i in range(1, 12)]
    )
    collapsed, detail = collapse_repetition(loop)
    check("a sustained loop is cut", detail.reason == "collapsed" and len(collapsed) < len(loop),
          detail.message)
    check("the cut keeps the distinct items",
          "Explainability" in collapsed and "Privacy Risk" in collapsed)
    check("a single echoed sentence is not a loop",
          collapse_repetition("Budget was 87%. In summary, budget was 87%.")[1].reason == "")
    check("an ordinary numbered list is untouched",
          collapse_repetition("1. Alpha item one\n2. Beta item two\n3. Gamma item three")[0]
          == "1. Alpha item one\n2. Beta item two\n3. Gamma item three")

    check("list ordinals are stripped",
          strip_enumeration("10. item ten\n12) item twelve") == "item ten\nitem twelve")
    check("years and decimals survive stripping",
          "2024" in strip_enumeration("2024 was a record.")
          and "87.5" in strip_enumeration("87.5 percent of it"))

    reset_guard_detail()
    answer, flagged = apply_guard(
        "The platform processed 4.7 million records.",
        ["During 2024 the platform processed 1.2 million records."],
        ["project_report.pdf"],
    )
    check("numeric guard fires on an unsupported figure", flagged == ["4.7"], str(flagged))
    check("numeric guard warns visibly", "Evidence warning:" in answer)

    reset_guard_detail()
    with_locator = (
        "The platform processed 1.2 million records.\n\n"
        "Also stated in the evidence: 1.2 million (MCA_MP_58-mini-project.pdf, Page 3)."
    )
    answer2, flagged2 = apply_guard(
        with_locator,
        ["During 2024 the platform processed 1.2 million records."],
        ["project_report.pdf"],
    )
    # A file name carries digits; a locator is metadata, not a claim.
    check("locator digits are not treated as claims", flagged2 == [], str(flagged2))
    check("answer unchanged when nothing is unsupported", answer2 == with_locator)


# ---------------------------------------------------------------------------
# parents: small-to-big chunking
# ---------------------------------------------------------------------------

def test_parents() -> None:
    section("parents -- small-to-big retrieval")
    from parents import _block_text, assign_parents, expand_to_parents
    from schemas import Modality

    make = _items()

    rows = [make(f"r{i}", f"Q{i} 2024, spend {200000 + i}, accuracy 9{i}.0",
                 heading=("Metrics Table",)) for i in range(1, 5)]
    kids, parents = assign_parents(rows, max_chars=1800)
    check("a whole table becomes one parent", len(parents) == 1, str(len(parents)))
    check("the parent holds every row",
          all(f"Q{i} 2024" in parents[0].content for i in range(1, 5)))
    check("children point at their parent",
          all(kid.parent_id == parents[0].item_id for kid in kids))
    check("a parent is never embedded", parents[0].vector_id is None)

    mixed = [
        make("a", "risk one", heading=("Open Risks",)),
        make("b", "risk two", heading=("Open Risks",)),
        make("c", "row one", heading=("Table 1",)),
        make("d", "row two", heading=("Table 1",)),
    ]
    _kids2, parents2 = assign_parents(mixed, max_chars=1800)
    check("distinct sections stay distinct",
          sorted(p.location.section for p in parents2) == ["Open Risks", "Table 1"])

    big = [make(f"b{i}", "x" * 700, heading=("Long",)) for i in range(5)]
    _kids3, parents3 = assign_parents(big, max_chars=1800)
    check("an oversized run splits under the cap",
          len(parents3) >= 2 and all(len(p.content) <= 1800 for p in parents3))

    _kids4, parents4 = assign_parents([make("solo", "one chunk only", heading=("Solo",))])
    check("a lone chunk gets no duplicate parent", parents4 == [])

    audio = [
        make("s1", "first segment", start=0.0, end=21.0, modality=Modality.AUDIO),
        make("s2", "second segment", start=21.0, end=37.0, modality=Modality.AUDIO),
    ]
    _kids5, parents5 = assign_parents(audio, max_chars=1800)
    check("a transcript parent spans its segments",
          parents5[0].location.timestamp_start == 0.0
          and parents5[0].location.timestamp_end == 37.0)

    pages = [
        make("p1", "text on page two", heading=("Results",), page=2),
        make("p2", "text on page three", heading=("Results",), page=3),
    ]
    _kids6, parents6 = assign_parents(pages, max_chars=1800)
    check("a parent spanning pages claims no single page",
          parents6[0].location.page_number is None)

    # Query time: two rows of one table collapse into the table, order and best
    # score preserved, and the retrieved passage leads the block.
    kids, parents = assign_parents(rows, max_chars=1800)
    store = _FakeStore(parents)
    selected = _results([kids[1], kids[3], make("d1", "docx paragraph", heading=("Q4",),
                                                source_id="src-docx")])
    out, detail = expand_to_parents(store, selected, char_budget=6000)
    check("selected siblings collapse into one block", len(out) == 2, detail.message)
    check("the collapsed block is the whole table",
          all(f"Q{i} 2024" in out[0].item.content for i in range(1, 5)))
    check("the parent inherits the best child score",
          out[0].rerank_score == selected[0].rerank_score)
    check("the retrieved passage leads the block",
          out[0].item.content.startswith(kids[1].content))
    check("both children are recorded", out[0].item.metadata["expanded_from"] == 
          [kids[1].item_id, kids[3].item_id])
    check("the matched passages stay available to the guards",
          out[0].item.metadata["retrieved_texts"] == [kids[1].content, kids[3].content])
    check("a child with no parent passes through", out[1].item.item_id == "d1")

    noparent = _results([make("x", "no parent here")])
    same, noop = expand_to_parents(_FakeStore([]), noparent)
    check("no parents -> list returned unchanged", same is noparent and not noop.message)

    tight, fallback = expand_to_parents(store, selected, char_budget=1)
    check("at least one block always reaches the model", len(tight) >= 1)
    block = _block_text(parents[0], kids[0])
    check("the block labels the surrounding section", "Complete section" in block)


# ---------------------------------------------------------------------------
# observability: trace and manifest
# ---------------------------------------------------------------------------

def test_observability() -> None:
    section("observability -- stage trace and run manifest")
    from observability import (
        build_manifest, format_stage_trace, manifest_text, stage_trace,
    )

    timings = {"retrieval_ms": 18.0, "rerank_ms": 500.0, "generation_ms": 6000.0,
               "total_ms": 6600.0}
    counts = {"retrieved": 10, "rerank_pool": 10, "reranked": 10, "best_score": 4.7,
              "threshold": -6.0, "citations": 2, "outcome": "answered"}
    trace = stage_trace(timings, counts, detail="temporal guard: refused")
    stages = {entry["stage"]: entry for entry in trace}
    check("every stage is listed, run or not", len(trace) >= 15, str(len(trace)))
    check("a stage that ran is marked", stages["retrieval"]["ran"] is True)
    check("a stage that never ran is marked, not hidden",
          stages["entity_guard"]["ran"] is False and stages["entity_guard"]["note"] == "not reached")
    check("counts become notes", "10 candidate" in stages["retrieval"]["note"])
    check("the decisive check is carried", trace[-1]["stage"] == "decision")
    check("the trace renders as aligned text", "Hybrid retrieval" in format_stage_trace(trace))

    manifest = build_manifest(index_stats={"sources": 1, "items": 2, "parents": 0,
                                           "relationships": 0, "vectors": 2})
    for required in ("application", "platform", "models", "retrieval", "chunking",
                     "guards", "generation", "security", "evaluation", "index"):
        check(f"manifest carries {required}", required in manifest)
    check("manifest names the guard flags",
          {"entity_guard_enabled", "temporal_guard_enabled",
           "period_targeting_enabled", "repetition_guard_enabled",
           "synthesis_completion_enabled"} <= set(manifest["guards"]))
    check("manifest caps generation", manifest["generation"]["num_predict"] > 0)
    check("manifest renders for humans", "commit" in manifest_text(manifest))


# ---------------------------------------------------------------------------
# vector_store: the parents table round-trips and cleans up
# ---------------------------------------------------------------------------

def test_store() -> None:
    section("vector_store -- parents table round-trip and deletion")
    import numpy as np
    from parents import assign_parents
    from schemas import Source, SourceType
    from vector_store import VectorStore

    store = VectorStore(index_path=_TMP / "index" / "faiss.index",
                        db_path=_TMP / "index" / "store.db")
    source = Source(source_id="src-1", filename="table.csv", source_type=SourceType.CSV,
                    file_path=str(_TMP / "table.csv"))
    store.add_source(source)
    make = _items()
    kids = [make(f"k{i}", f"row {i} content", heading=("Table",), source_id="src-1")
            for i in range(3)]
    _linked, parents = assign_parents(kids, max_chars=1800)
    # Real dimensionality: delete_source rebuilds survivors into an index of
    # settings.embedding_dim, so a toy width would break the rebuild.
    from config import settings as _settings

    vectors = np.random.default_rng(0).random(
        (len(kids), _settings.embedding_dim), dtype=np.float32
    )
    store.add_items(kids, vectors)
    store.add_parents(parents)

    check("parents are stored", store.parent_count() == 1)
    check("stats report parents", store.stats().get("parents") == 1)
    fetched = store.get_parent(parents[0].item_id)
    check("a parent round-trips with its content",
          fetched is not None and fetched.content == parents[0].content)
    check("a parent round-trips without a vector", fetched.vector_id is None)
    check("children keep their parent_id in the database",
          all(item.parent_id == parents[0].item_id
              for item in store.get_items_by_source("src-1")))
    check("parents never appear in search",
          all(item.item_id != parents[0].item_id
              for item, _score in store.search(vectors[0], 10)))
    check("add_parents is idempotent", store.add_parents(parents) == 1
          and store.parent_count() == 1)

    store.delete_source("src-1")
    check("deleting a source deletes its parents", store.parent_count() == 0)
    check("deleting a source deletes its items", store.stats()["items"] == 0)
    store.close()


def main() -> int:
    print("Evidence AI offline self-test")
    print(f"throwaway index: {_TMP}")
    for test in (test_constraint_guard, test_synthesis, test_repetition_and_numeric,
                 test_parents, test_observability, test_store):
        test()

    failed = [entry for entry in CHECKS if not entry[1]]
    print("\n" + "=" * 64)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _ok, detail in failed:
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
