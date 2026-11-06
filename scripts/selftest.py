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
                     "ingestion", "guards", "generation", "security", "evaluation",
                     "index"):
        check(f"manifest carries {required}", required in manifest)
    check("manifest names the guard flags",
          {"entity_guard_enabled", "temporal_guard_enabled",
           "period_targeting_enabled", "repetition_guard_enabled",
           "synthesis_completion_enabled"} <= set(manifest["guards"]))
    check("manifest caps generation", manifest["generation"]["num_predict"] > 0)
    check("manifest renders for humans", "commit" in manifest_text(manifest))

    # The browser's drop zone reads this block instead of carrying its own list of
    # formats, so agreement with the router is the whole point: a drop zone that
    # accepts a file the router will refuse wastes the user's time twice.
    from config import settings
    from rag_pipeline import AUDIO_EXTENSIONS, DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS

    ingestion = manifest["ingestion"]
    check("manifest lists every format the router accepts",
          set(ingestion["documents"]) == set(DOCUMENT_EXTENSIONS)
          and set(ingestion["images"]) == set(IMAGE_EXTENSIONS)
          and set(ingestion["audio"]) == set(AUDIO_EXTENSIONS),
          sorted(ingestion["documents"]))
    check("manifest reports a usable format list",
          ingestion["formats"] >= 10
          and ingestion["formats"] == len(set(DOCUMENT_EXTENSIONS) | set(IMAGE_EXTENSIONS)
                                          | set(AUDIO_EXTENSIONS)),
          str(ingestion["formats"]))
    check("manifest caps the upload size the UI enforces",
          ingestion["max_file_bytes"] == settings.max_file_bytes > 0)
    check("manifest says whether PDF tables are lifted out",
          ingestion["pdf_tables"] is settings.pdf_table_extraction_enabled)
    check("manifest says whether contextual retrieval is on",
          ingestion["contextual_retrieval"] is settings.contextual_retrieval_enabled)


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


# ---------------------------------------------------------------------------
# Rate limiting: the arithmetic has to be right before it guards a real route,
# because a limiter that refuses legitimate demo questions is worse than none.
# Every timing here is simulated by backdating a bucket's clock rather than by
# sleeping, so the section stays instant.
# ---------------------------------------------------------------------------

class _FakeRequest:
    """The two attributes rate_limit.client_key reads, nothing more."""

    def __init__(self, host=None, forwarded=()):
        self.client = type("Client", (), {"host": host})() if host else None
        self._forwarded = list(forwarded)
        self.headers = self

    def getlist(self, name):
        return list(self._forwarded) if name.lower() == "x-forwarded-for" else []


def test_rate_limit() -> None:
    section("rate limiting")
    from config import settings
    from rate_limit import IDLE_SECONDS, RateLimiter, client_key

    saved = {name: getattr(settings, name) for name in (
        "rate_limit_enabled", "rate_limit_queries_per_minute", "rate_limit_queries_burst",
        "rate_limit_ingest_per_minute", "rate_limit_ingest_burst",
        "rate_limit_trust_forwarded_for")}
    try:
        settings.rate_limit_enabled = True
        settings.rate_limit_queries_per_minute = 6.0    # one token every 10 s
        settings.rate_limit_queries_burst = 3
        settings.rate_limit_ingest_per_minute = 60.0
        settings.rate_limit_ingest_burst = 20
        settings.rate_limit_trust_forwarded_for = False
        limiter = RateLimiter()

        first = [limiter.check("queries", "account:demo")[0] for _ in range(3)]
        allowed, retry_after = limiter.check("queries", "account:demo")
        check("a new client gets its full burst", all(first), str(first))
        check("the request after the burst is refused", allowed is False)
        check("a refusal says how long to wait",
              9.0 <= retry_after <= 10.5, f"{retry_after:.2f}s for a 6/min limit")

        # Refill is simulated, not slept for: move the bucket's clock back.
        limiter._buckets["queries"]["account:demo"].updated -= 25.0
        check("tokens refill over time", limiter.check("queries", "account:demo")[0] is True)

        other_allowed, _ = limiter.check("ingest", "account:demo")
        check("buckets are separate cost classes", other_allowed is True,
              "draining 'queries' must not touch 'ingest'")

        limiter.reset()
        pair = [limiter.check("queries", "account:bulk", cost=2)[0] for _ in range(2)]
        check("a multi-file request is charged per file",
              pair == [True, False], str(pair))
        limiter.reset()
        huge, _ = limiter.check("queries", "account:huge", cost=999)
        after_huge = limiter.check("queries", "account:huge")[0]
        check("one huge request is allowed but drains the bucket",
              huge is True and after_huge is False,
              "refusing a legitimate 200-file upload outright would be worse")

        settings.rate_limit_enabled = False
        hammer = [limiter.check("queries", "account:off")[0] for _ in range(50)]
        check("rate_limit_enabled=false disables the limiter", all(hammer))
        settings.rate_limit_enabled = True
        settings.rate_limit_queries_per_minute = 0.0
        check("a zero rate means off, not refuse-everything",
              limiter.check("queries", "account:zero")[0] is True)
        settings.rate_limit_queries_per_minute = 6.0

        # ---- who shares a bucket -----------------------------------------
        check("an authenticated client is keyed by account, not address",
              client_key(_FakeRequest("10.0.0.5"), {"username": "demo", "role": "admin"})
              == "account:demo")
        check("with auth off every caller is not one client",
              client_key(_FakeRequest("10.0.0.5"),
                         {"username": "api-auth-disabled", "via": "disabled"})
              == "address:10.0.0.5")
        check("X-Forwarded-For is ignored unless it is trusted",
              client_key(_FakeRequest("10.0.0.5", ["1.2.3.4"]),
                         {"via": "disabled"}) == "address:10.0.0.5")
        settings.rate_limit_trust_forwarded_for = True
        check("when trusted, the rightmost hop is the client",
              client_key(_FakeRequest("10.0.0.5", ["spoofed.by.client, 203.0.113.9"]),
                         {"via": "disabled"}) == "address:203.0.113.9",
              "the leftmost entry is whatever the client chose to send")
        settings.rate_limit_trust_forwarded_for = False
        check("an IPv6 client is bucketed by its /64",
              client_key(_FakeRequest("2001:db8:1:2:3:4:5:6"), {"via": "disabled"})
              == "address:2001:db8:1:2::/64")
        check("a request with no peer address still gets a key",
              client_key(_FakeRequest(None), {"via": "disabled"}) == "address:unknown")

        # ---- bookkeeping --------------------------------------------------
        limiter.reset()
        limiter.check("queries", "account:idle")
        limiter._buckets["queries"]["account:idle"].updated -= IDLE_SECONDS + 1
        limiter._sweep(__import__("time").monotonic())
        check("idle clients stop being tracked",
              "account:idle" not in limiter._buckets["queries"],
              "otherwise the table grows without bound")
        snapshot = limiter.snapshot()
        check("the snapshot reports what an operator needs",
              set(snapshot["buckets"]) == {"queries", "ingest"}
              and snapshot["buckets"]["queries"]["per_minute"] == 6.0
              and "process" in snapshot["scope"], str(snapshot["scope"]))
        limiter.reset()
        check("reset clears every bucket",
              all(table == {} for table in limiter._buckets.values()))
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)


# ---------------------------------------------------------------------------
# The configuration surface itself. The Windows kit ships `.env.example` as the
# only description of the knobs, and every README claim about "every setting is
# documented" is a claim about these two files agreeing. Checking it here is what
# stops that claim rotting the next time a field is added.
# ---------------------------------------------------------------------------

def test_configuration() -> None:
    section("configuration surface")
    from config import PROJECT_ROOT, Settings

    example = PROJECT_ROOT / ".env.example"
    raw = example.read_bytes()
    text = raw.decode("utf-8")
    upper = text.upper()

    fields = list(Settings.model_fields)
    undocumented = [name for name in fields if name.upper() not in upper]
    check("every setting is documented in .env.example",
          not undocumented, f"{len(fields)} fields, undocumented: {undocumented[:6]}")
    check("the documented settings are real fields, not leftovers",
          len(fields) >= 80, f"{len(fields)} fields")
    check(".env.example is CRLF throughout, as Windows Notepad needs",
          raw.count(b"\n") == raw.count(b"\r\n"),
          f"{raw.count(b'\r\n')} CRLF, {raw.count(b'\n') - raw.count(b'\r\n')} bare LF")

    defaults = Settings.model_fields
    check("rate limiting is on by default",
          defaults["rate_limit_enabled"].default is True)
    check("the default query limit leaves room for a live demo",
          float(defaults["rate_limit_queries_per_minute"].default) >= 12.0
          and int(defaults["rate_limit_queries_burst"].default) >= 4,
          f"{defaults['rate_limit_queries_per_minute'].default}/min, "
          f"burst {defaults['rate_limit_queries_burst'].default}")
    check("a forwarded-for header is not trusted by default",
          defaults["rate_limit_trust_forwarded_for"].default is False,
          "trusting a header any client can set turns the limiter into a suggestion")


# ---------------------------------------------------------------------------
# The chunk inspector. It is a debugging tool, but it is the tool that gets read
# when a chunking change is being judged, so a wrong location string or a silently
# dropped chunk here would mislead exactly the person trying to be careful.
# ---------------------------------------------------------------------------

def test_dump_chunks() -> None:
    section("chunk inspector")
    import importlib.util

    from schemas import ContentItem, Modality, SourceLocation

    spec = importlib.util.spec_from_file_location(
        "dump_chunks", ROOT / "scripts" / "dump_chunks.py"
    )
    dump = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dump)

    def item(item_id, content, *, page=None, section=None, start=None, end=None,
             modality=Modality.TEXT, **metadata):
        return ContentItem(
            item_id=item_id, source_id="src-1", modality=modality, content=content,
            location=SourceLocation(page_number=page, section=section,
                                    timestamp_start=start, timestamp_end=end),
            metadata=dict(metadata),
        )

    check("a page and section are both rendered",
          dump._location(item("a", "x", page=3, section="Budget")) == "p.3 · Budget")
    rows = dump._rows([item("a", "chunk text", page=3, section="Budget",
                            context="Q4 review, carrying the utilisation figure."),
                       item("b", "plain chunk", page=4)], 0, None)
    check("the inspector reports the blurb a chunk was indexed with",
          rows[0]["indexed_context"] == "Q4 review, carrying the utilisation figure."
          and rows[1]["indexed_context"] is None, str([r["indexed_context"] for r in rows]))
    check("a blurb is flagged but not counted as chunk text",
          "contextual" in rows[0]["flags"] and rows[0]["chars"] == len("chunk text"))
    check("an audio segment is rendered as a clock range",
          dump._location(item("b", "x", modality=Modality.AUDIO, start=0.0, end=21.4))
          == "00:00-00:21")
    check("a location that was never recorded is a dash, not an invention",
          dump._location(item("c", "x")) == "-",
          "a page number for audio, or a timestamp for a paragraph, would be made up")

    flagged = item("d", "x", ocr_confidence=41.0, ocr_unreliable=True, heading_only=True)
    marks = dump._flags(flagged)
    check("the flags that explain a strange chunk are surfaced",
          "ocr-unreliable" in marks and "ocr 41%" in marks and "heading-only" in marks, marks)
    check("a parent block says so", "parent" in dump._flags(item("e", "x", expanded_from=["c1"])))

    chunks = [item("1", "Budget utilisation closed at 87 percent", page=3),
              item("2", "Two\nload tests\nbreached the objective", page=4),
              item("3", "Latency under peak load remains the top risk", page=4)]
    rows = dump._rows(chunks, 280, None)
    check("every chunk is listed, in order", [r["n"] for r in rows] == [1, 2, 3])
    check("a chunk's length is its real length",
          rows[0]["chars"] == len("Budget utilisation closed at 87 percent"))
    check("--grep filters case-insensitively",
          [r["n"] for r in dump._rows(chunks, 280, "LATENCY")] == [3])
    check("newlines are flattened so a row stays one line",
          "\n" not in dump._rows(chunks, 280, None)[1]["text"])
    check("--chars 0 means the whole text",
          dump._rows(chunks, 0, None)[1]["text"]
          == "Two\nload tests\nbreached the objective")
    check("a truncated preview says it was truncated",
          dump._preview("a" * 400, 280).endswith("…")
          and len(dump._preview("a" * 400, 280)) == 282)

    stats = dump._stats([item("1", "a" * 100), item("2", "a" * 300),
                         item("3", "a" * 50, modality=Modality.IMAGE)])
    check("sizes are reported per modality, not pooled",
          set(stats) == {"text", "image"} and stats["text"]["chunks"] == 2)
    check("the median is the median",
          stats["text"]["chars_median"] == 200 and stats["text"]["chars_max"] == 300)


# ---------------------------------------------------------------------------
# Collections. The one that matters most is the last: deleting a collection must
# move its sources, never delete them. "Delete collection" reads like "delete the
# documents", and a UI that meant that would destroy an index.
# ---------------------------------------------------------------------------

def test_collections() -> None:
    section("collections")
    import numpy as np
    from schemas import Source, SourceType
    from vector_store import DEFAULT_COLLECTION, VectorStore

    base = _TMP / "collections"
    store = VectorStore(index_path=base / "faiss.index", db_path=base / "store.db")
    from config import settings as _settings

    make = _items()
    for number, name in ((1, "board.pdf"), (2, "notes.docx")):
        source = Source(source_id=f"src-{number}", filename=name,
                        source_type=SourceType.PDF, file_path=str(base / name),
                        collection=DEFAULT_COLLECTION)
        store.add_source(source)
        items = [make(f"i{number}", f"content {number}", source_id=f"src-{number}")]
        store.add_items(items, np.random.default_rng(number).random(
            (1, _settings.embedding_dim), dtype=np.float32))

    check("every source starts in the default collection",
          [row["collection"] for row in store.list_collections()] == [DEFAULT_COLLECTION])

    def declared_rows() -> list:
        """The declared-collections table read on its own.

        list_collections() merges it with the labels on source rows, so it cannot
        show whether the default collection was written there - and a write that is
        invisible through the API is exactly the kind that makes two logically
        identical corpora differ on disk.
        """
        return [row["name"] for row in
                store._conn.execute("SELECT name FROM collections").fetchall()]

    check("a freshly built index declares no collection, not even the default",
          declared_rows() == [], str(declared_rows()))
    check("declaring the default by name returns it and writes nothing",
          store.create_collection(f"  {DEFAULT_COLLECTION}  ") == DEFAULT_COLLECTION
          and declared_rows() == [], str(declared_rows()))

    declared = store.create_collection("  Q3   Board  Pack ")
    check("a name is normalized, not stored as typed", declared == "Q3 Board Pack", declared)
    names = {row["collection"]: row for row in store.list_collections()}
    check("an empty declared collection is still listed",
          names.get("Q3 Board Pack", {}).get("sources") == 0,
          "otherwise it vanishes between creating it and filling it")
    check("counts report sources and indexed items separately",
          names[DEFAULT_COLLECTION] == {"collection": DEFAULT_COLLECTION,
                                        "sources": 2, "items": 2}, names[DEFAULT_COLLECTION])

    for bad, why in (("", "empty"), ("   ", "whitespace only"),
                     ("a/b", "path separator"), ("a\\b", "backslash"),
                     ("a\nb", "control character")):
        try:
            store.validate_collection_name(bad)
            check(f"a name that is {why} is refused", False, f"{bad!r} was accepted")
        except ValueError:
            check(f"a name that is {why} is refused", True)
    check("an over-long name is capped, not refused",
          len(store.validate_collection_name("x" * 400)) == 60)

    check("existence is checked case-insensitively",
          store.collection_exists("q3 board pack") and store.collection_exists(DEFAULT_COLLECTION))
    check("a name nobody used does not exist", not store.collection_exists("nope"))

    moved = store.assign_sources(["src-1", "does-not-exist"], "Q3 Board Pack")
    check("assigning files sources under the collection", moved == 1, str(moved))
    check("an unknown source id is ignored, not fatal",
          store.get_source("src-1").collection == "Q3 Board Pack"
          and store.get_source("src-2").collection == DEFAULT_COLLECTION)
    check("assigning to a new name declares it",
          store.collection_exists("Q3 Board Pack"))

    store.assign_sources(["src-2"], "scratch")
    renamed = store.rename_collection("scratch", "archive")
    check("renaming moves the sources", renamed == 1
          and store.get_source("src-2").collection == "archive")
    check("renaming leaves the old name behind",
          not any(row["collection"] == "scratch" for row in store.list_collections()))
    check("renaming does not touch the items",
          len(store.get_items_by_source("src-2")) == 1)

    deleted = store.delete_collection("archive")
    check("deleting a collection returns its sources to the default",
          deleted == 1 and store.get_source("src-2").collection == DEFAULT_COLLECTION)
    check("deleting a collection deletes no source",
          len(store.list_sources()) == 2 and store.stats()["items"] == 2)
    check("deleting a collection deletes no item",
          len(store.get_items_by_source("src-2")) == 1)
    try:
        store.delete_collection(DEFAULT_COLLECTION)
        check("the default collection cannot be deleted", False, "it was deleted")
    except ValueError:
        check("the default collection cannot be deleted", True)

    check("the declared table holds only what somebody actually declared",
          declared_rows() == ["Q3 Board Pack"], str(declared_rows()))
    # src-2, not src-1: the check below this one reopens the index and expects
    # src-1 to still be grouped, so filing it home here would fail there instead,
    # in a place that looks unrelated to the write being tested.
    store.assign_sources(["src-2"], DEFAULT_COLLECTION)
    check("filing a source back under the default writes no declared row",
          declared_rows() == ["Q3 Board Pack"]
          and store.get_source("src-2").collection == DEFAULT_COLLECTION
          and store.get_source("src-1").collection == "Q3 Board Pack",
          str(declared_rows()))
    # A scratch collection, not the one above: a later check reopens this index and
    # expects "Q3 Board Pack" to still be declared, and a test that quietly removes
    # the state another test asserts on fails in a place that looks unrelated.
    store.create_collection("scratch")
    store.delete_collection("scratch")
    check("a collection declared and then ungrouped leaves no row behind",
          declared_rows() == ["Q3 Board Pack"] and len(store.list_sources()) == 2
          and store.stats()["items"] == 2, str(declared_rows()))

    reopened = VectorStore(index_path=base / "faiss.index", db_path=base / "store.db")
    check("a declared collection survives reopening the index",
          reopened.collection_exists("Q3 Board Pack")
          and reopened.get_source("src-1").collection == "Q3 Board Pack")
    store.close()
    reopened.close()


def test_pdf_tables() -> None:
    section("PDF tables -- the grid, and the page text it must not lose")
    import collections
    import re

    import documents
    from config import settings

    # --- the pieces, on inputs small enough to read -----------------------
    check("an all-empty column is dropped from a grid",
          documents._prune_grid([["a", "", "b"], ["c", "", "d"]]) == [["a", "b"], ["c", "d"]])
    check("an all-empty row is dropped from a grid",
          documents._prune_grid([["a", "b"], ["", ""], ["c", "d"]]) == [["a", "b"], ["c", "d"]])
    check("a cell's line breaks are flattened into one line",
          documents._prune_grid([["one\ntwo", "x"]]) == [["one two", "x"]])
    markdown = documents._grid_to_markdown([["Quarter", "Engineers"], ["Q1", "3"], ["Q2", "4"]])
    check("a grid renders as markdown with a header rule",
          markdown == "| Quarter | Engineers |\n|---|---|\n| Q1 | 3 |\n| Q2 | 4 |", markdown)
    check("a numbered caption is parsed with its number",
          documents._table_captions("Intro\nTable 3: Budget utilisation by quarter\nBody")
          == {3: "Table 3: Budget utilisation by quarter"})
    check("prose that merely starts with 'Table' is not a caption",
          documents._table_captions("Table of contents\nTable 1") == {})

    words = [(0, 0, 10, 10, "shared", 0, 0, 0), (20, 20, 30, 30, "outside", 0, 1, 0)]
    taken = [False, False]
    first = documents._claim(words, taken, (0, 0, 12, 12))
    second = documents._claim(words, taken, (0, 0, 40, 40))
    check("a word inside two overlapping cells is claimed exactly once",
          first == [0] and second == [1], f"{first} then {second}")
    documents._release(taken, first)
    check("a released word goes back to the prose", taken == [False, True], str(taken))

    # --- the same code against the shipped deck --------------------------
    deck = ROOT / "data" / "documents" / "MCA_MP_58-mini-project.pdf"
    report = ROOT / "data" / "documents" / "project_report.pdf"
    check("the demo deck is present to extract tables from", deck.exists(), str(deck))
    if not deck.exists():
        return

    def extract(path, **overrides):
        saved = {name: getattr(settings, name) for name in overrides}
        for name, value in overrides.items():
            setattr(settings, name, value)
        warnings: list[str] = []
        try:
            return documents._extract_pdf(path, warnings), warnings
        finally:
            for name, value in saved.items():
                setattr(settings, name, value)

    off, off_warnings = extract(deck, pdf_table_extraction_enabled=False)
    on, on_warnings = extract(deck, pdf_table_extraction_enabled=True)
    tables = [item for item in on if item[2].get("is_table")]

    # Only table complaints are asserted on. The deck also produces an OCR warning
    # for its last page -- a scanned slide with a low-confidence digit -- and that
    # one is real, expected and unrelated to any of this. An earlier version of this
    # check demanded no warnings at all and passed only because Tesseract happened to
    # be missing from the machine running it; on a machine with Tesseract installed,
    # which is every machine that shipped this kit, it failed.
    def table_warnings(warnings: list[str]) -> list[str]:
        return [entry for entry in warnings if "table" in entry.lower()]

    check("the shipped default lifts tables out of the pages that have them",
          settings.pdf_table_extraction_enabled is True
          and any(item[2].get("is_table") for item in on))
    check("turning the flag off indexes no table items at all",
          not any(item[2].get("is_table") for item in off))
    check("reading the deck's tables complains about nothing",
          table_warnings(on_warnings) == [] and table_warnings(off_warnings) == [],
          str(table_warnings(on_warnings) + table_warnings(off_warnings)))
    check("both of the deck's literature tables are found", len(tables) == 2,
          f"{len(tables)} found")
    check("a table item carries its page and its shape",
          all(item[1].page_number in (4, 5) and item[2]["table_rows"] >= 2
              and item[2]["table_cols"] >= 2 and item[2]["table_cells"] >= 3
              for item in tables),
          str([(i[1].page_number, i[2]) for i in tables]))
    check("the recorded shape agrees with the rendered grid",
          all(sum(1 for line in item[0].splitlines() if line.startswith("|"))
              == item[2]["table_rows"] + 1 for item in tables))

    # The reason the feature exists, measured on this corpus rather than asserted.
    page4_on = [item for item in on if item[1].page_number == 4]
    page4_off = [item for item in off if item[1].page_number == 4]
    check("a grid row keeps a paper beside its own finding",
          any("Sentence-BERT" in item[0] and "similarity" in item[0] for item in page4_on))
    check("plain extraction scatters that same row",
          not any("Sentence-BERT" in item[0] and "similarity" in item[0] for item in page4_off))

    # No word may be withheld: prose plus grids has to hold every word the page
    # has, which is the property the claim/release bookkeeping exists to give.
    import pymupdf

    page_words: collections.Counter = collections.Counter()
    with pymupdf.open(deck) as doc:
        for page in doc:
            for word in page.get_text("words") or []:
                page_words[str(word[4]).lower()] += 1
    indexed: collections.Counter = collections.Counter()
    for text, location, _meta in on:
        for chunk in (text, location.section or ""):
            for token in re.findall(r"[^\s|]+", chunk.lower()):
                indexed[token] += 1
    withheld = {word: count for word, count in page_words.items()
                if indexed.get(word, 0) < count}
    check("every word on the deck's pages still reaches the index", not withheld,
          f"{len(withheld)} withheld: {sorted(withheld)[:6]}")

    refused, _ = extract(deck, pdf_table_extraction_enabled=True, pdf_table_min_rows=99)
    check("a region too small to be a table is handed back to the prose",
          [item[0] for item in refused] == [item[0] for item in off])

    report_off, _ = extract(report, pdf_table_extraction_enabled=False)
    report_on, _ = extract(report, pdf_table_extraction_enabled=True)
    check("a PDF with no tables is untouched by the flag",
          [item[0] for item in report_on] == [item[0] for item in report_off]
          and not any(item[2].get("is_table") for item in report_on))


def test_contextual() -> None:
    section("contextual retrieval -- blurbs that never reach a screen")
    import numpy as np

    import contextual
    import embeddings
    import lexical
    from config import settings
    from schemas import ContentItem, Modality, SourceLocation
    from vector_store import VectorStore

    def make(content, item_id="it-1", source_id="src-1", **metadata):
        return ContentItem(
            item_id=item_id, source_id=source_id, modality=Modality.TEXT,
            content=content, location=SourceLocation(page_number=1),
            metadata=dict(metadata),
        )

    plain = make("Budget utilisation closed at 87 percent.")
    check("an item with no blurb indexes as itself",
          contextual.indexed_text(plain) == plain.content)
    blurbed = make("Budget utilisation closed at 87 percent.",
                   **{contextual.CONTEXT_KEY: "Q4 review of the Alpha programme."})
    check("a blurb is prepended to the indexed text, on its own line",
          contextual.indexed_text(blurbed)
          == "Q4 review of the Alpha programme.\nBudget utilisation closed at 87 percent.")
    check("the blurb is metadata, not content",
          blurbed.content == "Budget utilisation closed at 87 percent.")

    short = "a document shorter than the cap"
    check("a document within the cap is sent whole",
          contextual.truncate_document(short, 500) == short)
    cut = contextual.truncate_document("H" * 40 + "M" * 400 + "T" * 40, 60)
    check("a long document keeps its head and its tail, and says what is missing",
          cut.startswith("H") and cut.endswith("T") and "omitted" in cut, cut)

    prompt = contextual.build_prompt("THE DOCUMENT", "THE FRAGMENT", "report.pdf", 60, 12000)
    check("the prompt carries the document, the fragment and the filename",
          all(piece in prompt for piece in ("THE DOCUMENT", "THE FRAGMENT", "report.pdf")))
    check("the prompt forbids inventing facts and asks for no preamble",
          "not in the document" in prompt and "no preamble" in prompt)
    check("the prompt states the reply cap", "at most 60 tokens" in prompt)

    check("an empty reply is not a blurb", contextual.clean_blurb("   ", "chunk", 300) == "")
    check("a refusal is not a blurb",
          contextual.clean_blurb("I cannot help with that request.", "chunk", 300) == "")
    check("a model announcing itself is not a blurb",
          contextual.clean_blurb("As an AI language model, this fragment discusses...",
                                 "chunk", 300) == "")
    check("a reply that only repeats the chunk is not a blurb",
          contextual.clean_blurb("the budget was 87 percent",
                                 "The budget was 87 percent.", 300) == "")
    check("surrounding quotes are stripped, the sentence is kept",
          contextual.clean_blurb('"Q4 budget utilisation."', "87 percent", 300)
          == "Q4 budget utilisation.")
    capped = contextual.clean_blurb("word " * 80, "x", 40)
    check("an over-long blurb is cut at a word boundary and marked",
          len(capped) <= 44 and capped.endswith(" …") and "  " not in capped, capped)

    calls: list[str] = []

    def fake_generate(prompt, max_tokens):
        calls.append(prompt)
        return "Section 3 of the report, carrying the utilisation figure."

    items = [make("Budget utilisation closed at 87 percent.", "it-1"),
             make("Team size was six engineers.", "it-2")]
    before = [item.content for item in items]
    written, elapsed = contextual.annotate_items(items, "report.pdf", generate=fake_generate)
    check("a blurb is written for every chunk of a real document",
          written == 2 and elapsed >= 0.0
          and all((item.metadata or {}).get(contextual.CONTEXT_KEY) for item in items))
    check("annotating never rewrites the chunk", [item.content for item in items] == before)
    check("the model is shown the chunk it is situating",
          all("Budget utilisation" in calls[0] for _ in [0]) and "six engineers" in calls[1])

    junk = contextual.annotate_items([make("one", "it-3"), make("two", "it-4")], "r.pdf",
                                     generate=lambda prompt, tokens: "I cannot say.")
    warnings: list[str] = []
    contextual.annotate_items([make("one", "it-5"), make("two", "it-6")], "r.pdf",
                              generate=lambda prompt, tokens: "", warnings=warnings)
    check("a document whose blurbs are all junk says so, and indexes the chunks anyway",
          junk[0] == 0 and len(warnings) == 1 and "indexed as themselves" in warnings[0],
          str(warnings))

    def explode(prompt, tokens):
        raise RuntimeError("ollama is not running")

    survived: list[str] = []
    written, _ = contextual.annotate_items([make("one", "it-7"), make("two", "it-8")],
                                           "r.pdf", generate=explode, warnings=survived)
    check("a model that is down costs a warning naming the cause, not an ingest",
          written == 0 and len(survived) == 1 and "RuntimeError" in survived[0],
          str(survived))

    saved_min = settings.contextual_min_items
    try:
        settings.contextual_min_items = 3
        written, _ = contextual.annotate_items([make("only chunk", "it-9")], "one.pdf",
                                               generate=fake_generate)
        check("a source too small to situate is skipped without calling the model",
              written == 0 and calls[-1] != "only chunk")
    finally:
        settings.contextual_min_items = saved_min

    captured: list[str] = []
    real_embed = embeddings.embed_texts

    def capture(texts, batch_size=32):
        captured.extend(texts)
        return np.zeros((len(texts), settings.embedding_dim), dtype=np.float32)

    embeddings.embed_texts = capture
    try:
        embeddings.embed_items([make("Budget utilisation closed at 87 percent.", "it-10",
                                     **{contextual.CONTEXT_KEY: "Q4 review."})])
    finally:
        embeddings.embed_texts = real_embed
    check("the embedding is of the blurb plus the chunk, not the chunk alone",
          captured == ["Q4 review.\nBudget utilisation closed at 87 percent."], str(captured))
    check("the lexical leg indexes the same text as the embedding",
          lexical.indexed_text is contextual.indexed_text)

    base = _TMP / "contextual"
    base.mkdir(parents=True, exist_ok=True)
    store = VectorStore(index_path=base / "faiss.index", db_path=base / "store.db")
    from schemas import Source, SourceType

    store.add_source(Source(source_id="src-ctx", filename="report.pdf",
                            source_type=SourceType.PDF, file_path="report.pdf",
                            mime_type="application/pdf", file_size=10))
    # The item has to be filed under the source row that exists, or the reopened
    # store has nothing to hand back for it.
    stored = store.add_items(
        [make("Budget utilisation closed at 87 percent.", "it-11", source_id="src-ctx",
              **{contextual.CONTEXT_KEY: "Q4 review of the Alpha programme."})],
        np.zeros((1, settings.embedding_dim), dtype=np.float32),
    )
    check("the store accepted the annotated item", len(stored) == 1, str(stored))
    store.close()
    reopened = VectorStore(index_path=base / "faiss.index", db_path=base / "store.db")
    check("a blurb is persisted with the item and survives reopening the index",
          (reopened.get_items_by_source("src-ctx")[0].metadata or {}).get(contextual.CONTEXT_KEY)
          == "Q4 review of the Alpha programme.")
    check("the persisted chunk is still the extracted text",
          reopened.get_items_by_source("src-ctx")[0].content
          == "Budget utilisation closed at 87 percent.")
    reopened.close()


def test_experiment_harness() -> None:
    section("big-model experiment harness")
    import importlib.util

    from config import settings

    source = (ROOT / "scripts" / "experiment_big_model.py").read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "experiment_big_model", ROOT / "scripts" / "experiment_big_model.py"
    )
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    check("the harness imports and declares the three arms it documents",
          set(exp.ARMS) == {"reference", "contextual", "structure"}, str(sorted(exp.ARMS)))

    # An arm's delta is a set of environment variables. A key that is not a real
    # Settings field is ignored silently, and the arm then measures the feature's
    # absence and reports it as the feature's effect -- the same class of bug as a
    # flag reading a metadata key nobody writes.
    fields = set(settings.model_fields)
    unknown = {key for spec_arm in exp.ARMS.values() for key in spec_arm["delta"]
               if key.lower() not in fields}
    check("every arm's environment key is a real config.Settings field", not unknown,
          str(sorted(unknown)))

    # The two features that ship off by default are the reason the harness exists. If
    # either flag stops being covered, the documentation's "re-measure at 3B+" points
    # at an instrument that cannot measure it.
    all_keys = {key for arm in exp.ARMS.values() for key in arm["delta"]}
    check("both off-by-default features are covered by an arm",
          {"CONTEXTUAL_RETRIEVAL_ENABLED", "CHUNKING_MODE"} <= all_keys, str(sorted(all_keys)))

    # An arm may change the feature under test. It may not change the instrument: a
    # threshold, a top-k, a guard or a seed moved in the same breath as the feature
    # would make the comparison meaningless and still print a confident verdict.
    instrument = {"abstention_threshold", "abstention_threshold_image", "top_k",
                  "rerank_top_k", "hybrid_enabled", "mmr_enabled", "mmr_lambda",
                  "mmr_max_per_source", "coverage_guard_enabled", "entity_guard_enabled",
                  "temporal_guard_enabled", "period_targeting_enabled",
                  "numeric_guard_enabled", "repetition_guard_enabled",
                  "synthesis_completion_enabled", "ollama_temperature", "ollama_seed",
                  "embedding_model", "reranker_model", "chunk_size", "chunk_overlap"}
    touched = instrument & {key.lower() for key in all_keys}
    check("no arm moves the measuring instrument, only the feature under test",
          not touched, str(sorted(touched)))

    # Methodology: a feature is only comparable to the same model on the same corpus.
    check("the reference arm is the one compared against the 0.5b baseline",
          exp.ARMS["reference"]["compare_against"] == "baseline")
    check("every feature arm is compared against the reference arm, not the baseline",
          all(arm["compare_against"] == "reference"
              for name, arm in exp.ARMS.items() if name != "reference"))

    # Expectations are read against the persisted index, so their rule strings have to
    # parse; "==0" once produced int("=0") in an earlier draft of this script.
    rules = [rule for arm in exp.ARMS.values() for rule in arm["expect"].values()]
    check("every index expectation is a parseable ==N or >=N rule",
          all(rule.startswith(("==", ">=")) and rule[2:].lstrip("-").isdigit()
              for rule in rules), str(rules))
    check("a contextual arm that wrote no blurb is caught, not reported",
          len(exp.check_expectations("contextual", {"blurbs": 0, "tables": 3,
                                                    "parents": 0})) == 1)
    check("a contextual arm that wrote blurbs passes its expectations",
          not exp.check_expectations("contextual", {"blurbs": 31, "tables": 3,
                                                    "parents": 0}))
    check("a reference arm with a stray blurb is caught",
          len(exp.check_expectations("reference", {"blurbs": 1, "tables": 3,
                                                   "parents": 0})) == 1)
    check("a structure arm that built no parents is caught",
          len(exp.check_expectations("structure", {"blurbs": 0, "tables": 3,
                                                   "parents": 0})) == 1)

    # The verdict rule, on synthetic aggregates: this is the sentence a human will act
    # on, so it is tested rather than trusted.
    off = {"retrieval_precision": 0.35, "citation_accuracy": 0.56,
           "answer_correctness": 1.0, "groundedness": 0.94, "hallucination_rate": 0.06,
           "forbidden_leak_rate": 0.0, "retrieval_recall": 0.96, "mrr": 0.65,
           "abstention_accuracy": 1.0}
    tie = exp.compare_arms(off, dict(off), 0.0)
    check("an identical arm is a tie on every gated metric",
          not tie["worse"] and not tie["better"] and len(tie["unchanged"]) == len(off))
    check("a tie is not a win: the verdict is KEEP OFF and says why",
          exp.verdict_for(tie, 0, [], [], 0.0)[0] == "KEEP OFF"
          and "tie" in exp.verdict_for(tie, 0, [], [], 0.0)[1][0])
    up = dict(off, groundedness=1.0, hallucination_rate=0.0)
    better = exp.compare_arms(off, up, 0.0)
    check("groundedness up and hallucination down both count as better",
          {metric for metric, _, _ in better["better"]} == {"groundedness",
                                                            "hallucination_rate"}
          and not better["worse"])
    check("a clean win with a passing gate recommends turning the feature on",
          exp.verdict_for(better, 0, [], [], 0.0)[0] == "RECOMMEND ON")
    check("a failing gate keeps the feature off even when the metrics improved",
          exp.verdict_for(better, 1, [], [], 0.0)[0] == "KEEP OFF")
    down = dict(off, answer_correctness=0.67)
    worse = exp.compare_arms(off, down, 0.0)
    check("a lower-is-better metric that rises counts as worse, and vice versa",
          [metric for metric, _, _ in exp.compare_arms(off, dict(
              off, forbidden_leak_rate=0.2), 0.0)["worse"]] == ["forbidden_leak_rate"]
          and [metric for metric, _, _ in worse["worse"]] == ["answer_correctness"])
    check("a single regressed metric keeps the feature off",
          exp.verdict_for(worse, 0, [], [], 0.0)[0] == "KEEP OFF")
    check("a query that lost its PASS keeps the feature off",
          exp.verdict_for(better, 0, [], ["Q7"], 0.0)[0] == "KEEP OFF")
    check("a metric missing from one side is reported absent, never as zero",
          "answer_correctness" in exp.compare_arms(
              off, {key: value for key, value in off.items()
                    if key != "answer_correctness"}, 0.0)["absent"])

    # A full-mode report stores no per-query `status`: evaluation.py computes it for its
    # console line and never persists it. The harness re-derives it with that rule, so
    # the rule is what gets tested here -- a wrong derivation would silently turn every
    # PASS/CHECK count in a summary into zero.
    check("a correct, well-behaved answer is a PASS",
          exp.query_status({"answer_correctness": 1.0, "abstention_correct": True,
                            "forbidden_hits": []}) == "PASS")
    check("a wrong answer is a CHECK",
          exp.query_status({"answer_correctness": 0.4, "abstention_correct": True,
                            "forbidden_hits": []}) == "CHECK")
    check("a leaked forbidden figure is a CHECK even when the answer is right",
          exp.query_status({"answer_correctness": 1.0, "abstention_correct": True,
                            "forbidden_hits": ["1200000"]}) == "CHECK")
    check("a wrong abstention is a CHECK",
          exp.query_status({"answer_correctness": 1.0, "abstention_correct": False,
                            "forbidden_hits": []}) == "CHECK")
    check("a retrieval-only row's stored status is used as written",
          exp.query_status({"status": "UNDECIDED"}) == "UNDECIDED")
    check("a row with no answer and no stored status has no verdict, not a zero",
          exp.query_status({"recall": 1.0}) is None)

    # Per-query movement: at 0.5b the entire groundedness gain was one query and the
    # entire correctness loss was another, so the harness has to name them. These rows
    # carry no `status`, which is what a full report actually looks like.
    off_report = {"per_query": [
        {"id": "Q7", "query": "budget", "answer_correctness": 1.0, "citation_accuracy": 0.83,
         "abstention_correct": True, "forbidden_hits": []},
        {"id": "Q15", "query": "limitation", "answer_correctness": 1.0,
         "citation_accuracy": 1.0, "abstention_correct": True, "forbidden_hits": []}]}
    on_report = {"per_query": [
        {"id": "Q7", "query": "budget", "answer_correctness": 0.4, "citation_accuracy": 1.0,
         "abstention_correct": True, "forbidden_hits": []},
        {"id": "Q15", "query": "limitation", "answer_correctness": 1.0,
         "citation_accuracy": 1.0, "abstention_correct": True, "forbidden_hits": []}]}
    moves = exp.per_query_movements(off_report, on_report)
    check("only the query that actually moved is reported",
          [move["id"] for move in moves] == ["Q7"], str(moves))
    check("a movement carries both derived statuses and both values",
          moves[0]["before_status"] == "PASS" and moves[0]["after_status"] == "CHECK"
          and moves[0]["deltas"]["answer_correctness"] == (1.0, 0.4))
    check("a query that lost its PASS is named as dropped",
          [move["id"] for move in moves
           if move["before_status"] == "PASS"
           and move["after_status"] in ("CHECK", "FAIL", "UNDECIDED")] == ["Q7"])

    # The instrument must not be able to do the two things that would invalidate every
    # number it produces.
    check("the harness can never promote a run to the baseline",
          "save-baseline" not in source and "save_baseline" not in source)
    check("the harness states that it does not edit configuration",
          "never edits config.py" in source or "never edit config.py" in source)

    # Environment beats .env in config.py, and the whole harness rests on that: an arm's
    # flags are passed as environment variables so they die with the arm. Measured here
    # in a child process against the throwaway index selftest already exported.
    resolved = exp.resolve_settings({"CONTEXTUAL_RETRIEVAL_ENABLED": "true",
                                     "CHUNKING_MODE": "structure"})
    check("an arm's environment overrides the shipped defaults",
          resolved.get("contextual_retrieval_enabled") is True
          and resolved.get("chunking_mode") == "structure", str(resolved)[:200])
    plain = exp.resolve_settings({})
    check("without the arm's environment the shipped defaults come back",
          plain.get("contextual_retrieval_enabled") is False
          and plain.get("chunking_mode") == "fixed", str(plain)[:200])
    check("the harness can name every Settings field it validates arms against",
          len(exp.settings_field_names()) > 90, str(len(exp.settings_field_names())))


def test_kit_verifier() -> None:
    section("kit manifest and verifier")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_kit", ROOT / "scripts" / "verify_kit.py"
    )
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)

    html = ('<script type="module" crossorigin src="/assets/index-IryU-pKV.js"></script>'
            '<link rel="stylesheet" crossorigin href="/assets/index-hGHgsYra.css">'
            '<img src="https://example.com/logo.png">')
    check("the verifier finds the hashed assets a built page asks the browser for",
          vk.asset_references(html) == {"assets/index-IryU-pKV.js",
                                        "assets/index-hGHgsYra.css"},
          str(sorted(vk.asset_references(html))))
    check("it ignores assets served from somewhere else",
          not any(reference.startswith("http")
                  for reference in vk.asset_references(html)))

    names = [".env", ".env.example", "index/store.db", "history/chat_history.db",
             "auth/users.db", "webui/node_modules/react/index.js",
             "evaluation/reports/report_20261106_000901.json", "README.md"]
    leaked = vk.forbidden_entries(names)
    check("it flags a real .env but not the example a kit is supposed to ship",
          ".env" in leaked and ".env.example" not in leaked, str(leaked))
    check("it flags every generated database and the state directories around them",
          {"index/store.db", "history/chat_history.db", "auth/users.db"} <= set(leaked),
          str(leaked))
    check("it flags vendored dependency trees",
          "webui/node_modules/react/index.js" in leaked)
    check("it leaves the reports and guides a kit must carry alone",
          not ({"evaluation/reports/report_20261106_000901.json", "README.md"}
               & set(leaked)), str(leaked))

    text = "precision .3531 (report_20261106_000901) vs report_20261105_210146_retrieval"
    check("it reads both report spellings out of documentation",
          vk.cited_reports({"README.md": text})
          == {"report_20261106_000901", "report_20261105_210146_retrieval"},
          str(sorted(vk.cited_reports({"README.md": text}))))

    manifest = {"entries": {"a.py": {"sha256": "a" * 64, "bytes": 10},
                            "gone.py": {"sha256": "b" * 64, "bytes": 20},
                            "changed.py": {"sha256": "c" * 64, "bytes": 30}}}
    actual = {"a.py": ("a" * 64, 10), "changed.py": ("d" * 64, 31),
              "extra.py": ("e" * 64, 5)}
    diff = vk.compare_manifest(manifest, actual)
    check("a file the manifest lists but the archive lacks is reported missing",
          diff["missing"] == ["gone.py"], str(diff["missing"]))
    check("a file whose hash or size moved is a mismatch, not a missing file",
          [entry["name"] for entry in diff["mismatched"]] == ["changed.py"]
          and diff["mismatched"][0]["actual"] == ("d" * 64, 31))
    check("a file added after the build is reported as unaccounted",
          diff["unaccounted"] == ["extra.py"], str(diff["unaccounted"]))
    check("the manifest never counts itself as unaccounted",
          vk.compare_manifest({"entries": {}},
                              {vk.MANIFEST_NAME: ("f" * 64, 1)})["unaccounted"] == [])

    target = _TMP / "extract"
    check("an entry that would write outside the target directory is refused",
          vk.unsafe_destination("../escape.txt", target)
          and vk.unsafe_destination("/etc/passwd", target))
    check("an ordinary relative entry is allowed",
          not vk.unsafe_destination("scripts/selftest.py", target))

    check("a single wrapper folder is detected and stripped",
          vk.top_level(["Kit/README.md", "Kit/api.py"]) == "Kit"
          and vk.relative_names(["Kit/README.md", "Kit/api.py", "Kit/"], "Kit")
          == ["README.md", "api.py"])
    check("a flat archive is read as flat",
          vk.top_level(["README.md", "api.py"]) is None)
    check("a mixed archive is read as flat so no entry is silently skipped",
          vk.top_level(["README.md", "Kit/api.py"]) is None
          and vk.relative_names(["README.md", "Kit/api.py"], None)
          == ["Kit/api.py", "README.md"])

    check("the verifier requires the scripts its own documentation tells a reader to run",
          {"scripts/selftest.py", "scripts/apitest.py", "scripts/build_kit.py",
           "scripts/verify_kit.py", "scripts/experiment_big_model.py",
           "scripts/reindex.py", "scripts/ingest_demo.py"} <= set(vk.REQUIRED),
          str(sorted(vk.REQUIRED)))
    check("it requires the example environment file and never a real one",
          ".env.example" in vk.REQUIRED and ".env" not in vk.REQUIRED)
    builder = (ROOT / "scripts" / "build_kit.py").read_text(encoding="utf-8")
    missing_from_builder = [name for name in vk.REQUIRED if f'"{name}"' not in builder]
    check("every file the verifier demands is also a build-blocking requirement",
          not missing_from_builder, ", ".join(missing_from_builder))
    check("the manifest filename the verifier looks for is the one the builder writes",
          vk.MANIFEST_NAME == "KIT_MANIFEST.json" and vk.MANIFEST_NAME in builder)


def main() -> int:
    print("Evidence AI offline self-test")
    print(f"throwaway index: {_TMP}")
    for test in (test_constraint_guard, test_synthesis, test_repetition_and_numeric,
                 test_parents, test_observability, test_store, test_rate_limit,
                 test_configuration, test_dump_chunks, test_collections,
                 test_pdf_tables, test_contextual, test_experiment_harness,
                 test_kit_verifier):
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
