"""Evidence-completion aid for broad multi-source questions.

The measured failure this exists for
-----------------------------------
A deliberately broad question ("What evidence do we have about Project Alpha's
2024 development?") retrieves four sources that each state the same headline
figures. Retrieval for that query is good (recall 1.00, MRR 1.00, citation
0.75), yet answer correctness was 0.00: the model copied the first evidence
block -- the report's boilerplate cover paragraph -- and dropped "1.2 million"
and "99.4" entirely.

A prompt-side fix was built and measured first: a digest listing the figures the
evidence shares across sources, placed before the evidence and again next to the
request, with an explicit instruction to combine blocks. At qwen2.5:0.5b the
answer came back **byte-identical to the run with no digest at all**, and
byte-identical to the pre-change baseline. A 0.5b model does not follow that
kind of instruction, so the aid is applied to the answer instead of the prompt,
where it does not depend on the model complying.

What it does, and what it is not
--------------------------------
It is not model synthesis and it is not a disclaimer. After generation, if the
question asked for evidence across sources and the answer omits a figure that
two or more retrieved sources state independently, that figure is appended with
the files and locations that state it. Every appended character comes from
retrieved evidence, so nothing new enters the answer, and citations are built
afterwards so the appended figures are cited like the rest.

It is deliberately narrow:

* only synthesis phrasing triggers it (``SYNTHESIS_MARKERS``); on the 16-query
  evaluation dataset that is exactly one query, so lookup, adversarial and
  abstention behaviour is untouched;
* only figures corroborated by two or more distinct sources qualify -- a
  single-source figure is already visible in its own evidence block;
* at most three figures are appended, and a figure the answer already states in
  any form (``99.4`` counts for ``99.4%``) is never repeated.

The appended line is scored by the evaluation harness on purpose. Unlike the
numeric guard's "Evidence warning:" caveat, which ``strip_caveat`` removes
because a disclaimer is the opposite of an assertion, this line *is* an
assertion of grounded fact, so it should count.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Stable marker, mirroring numeric_guard.NUMERIC_GUARD_MARKER, so the UI can
# render the line differently and a reviewer can grep for it.
COMPLETION_MARKER = "Also stated in the evidence:"

# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

# Phrases that ask for an inventory of facts rather than one fact. Kept short
# and literal on purpose: a broad trigger would append figures to lookup
# answers, where they are noise at best and a near-miss temptation at worst.
SYNTHESIS_MARKERS = (
    "what evidence",
    "what do we know",
    "what do we have",
    "summarise",
    "summarize",
    "summary of",
    "overall",
    "compare",
    "across the sources",
    "cross-source",
    "in total",
    "how do the sources",
    "what are the key",
)


def is_synthesis_query(question: str) -> bool:
    """True when the question asks for facts spanning sources, not one fact."""
    lowered = (question or "").lower()
    return any(marker in lowered for marker in SYNTHESIS_MARKERS)


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

# A number with an optional unit. The lookbehind keeps "Q3", "p2" and the year
# inside "2024-11-05" from being read as standalone figures.
_FIGURE_RE = re.compile(
    r"(?<![A-Za-z0-9.,/-])"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>%|percent|million|billion|thousand|(?<![A-Za-z])k(?![A-Za-z]))?",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "million": "million",
    "billion": "billion",
    "thousand": "thousand",
    "k": "thousand",
}


@dataclass(frozen=True)
class Figure:
    """One numeric fact as written in the evidence."""

    raw: str        # exactly as written, separators kept: "330,000"
    unit: str       # "" | percent | million | billion | thousand

    @property
    def value(self) -> str:
        """Normalised for comparison only: "330,000" and "330000" are one fact."""
        return self.raw.replace(",", "")

    @property
    def key(self) -> tuple[str, str]:
        return (self.value, self.unit)

    def render(self) -> str:
        """Spell a figure the way the evidence does.

        Anything appended to an answer is copied by readers and by citation
        matching, so re-formatting "330,000" as "330000" would put a string in
        the answer that appears in no source.
        """
        if self.unit == "percent":
            return f"{self.raw}%"
        if self.unit:
            return f"{self.raw} {self.unit}"
        return self.raw


def figures_in(text: str) -> list[Figure]:
    """Every standalone figure in a string, in order of appearance.

    Bare four-digit numbers in the plausible year range are skipped unless they
    carry a unit, so "in 2024" does not become a figure while "2024%" would.
    """
    out: list[Figure] = []
    for match in _FIGURE_RE.finditer(text or ""):
        raw_unit = (match.group("unit") or "").lower()
        unit = _UNIT_ALIASES.get(raw_unit, "")
        raw = match.group("value")
        value = raw.replace(",", "")
        if not unit and re.fullmatch(r"\d{4}", value) and 1900 <= int(value) <= 2100:
            continue
        out.append(Figure(raw=raw, unit=unit))
    return out


# ---------------------------------------------------------------------------
# Shared figures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharedFigure:
    """A figure two or more distinct sources state."""

    surface: str              # most common spelling across the sources
    value: str                # normalised, for the "already in the answer" test
    labels: tuple[str, ...]   # "file, location" per source, in evidence order

    @property
    def sources(self) -> int:
        return len(self.labels)


def _label(result, filenames: dict[str, str]) -> str:
    """`file, location` for one evidence item.

    Deliberately not ``location.human()``: that renders an audio locator as a
    clock range, and "00:00-00:21" hands the numeric guard two figures ("00",
    "21") that exist in no evidence text -- measured, that alone dropped Q5's
    groundedness to 0.0 and attached a spurious "Evidence warning" to a correct
    answer. Page numbers and section names stay: they are how a reader finds the
    passage, and the citations panel still shows the exact audio offset.
    """
    item = getattr(result, "item", None)
    name = ""
    if item is not None:
        name = (
            filenames.get(getattr(item, "source_id", ""), "")
            or getattr(item, "filename", "")
            or ""
        )
    name = name or "source"
    location = getattr(item, "location", None) if item is not None else None
    parts: list[str] = []
    if location is not None:
        page = getattr(location, "page_number", None)
        if page is not None:
            parts.append(f"Page {page}")
        section = getattr(location, "section", None)
        if section:
            parts.append(section)
    return f"{name}, {' - '.join(parts)}" if parts else name


def strip_completion(answer: str) -> str:
    """Remove any trailing evidence-completion line, leaving the claim text.

    The completion line's parentheticals are source locators -- file names, page
    numbers, section titles -- not claims about the world, and a file name such
    as "MCA_MP_58-mini-project.pdf" carries digits that no evidence text states.
    Both the numeric guard and the evaluation harness therefore measure the
    generated answer; the figures the line adds are copied out of the evidence by
    construction, and ``answer_correctness`` still scores the full text.

    Call-order contract: strip the numeric guard's caveat first. The guard
    appends after this line, so ``strip_completion(strip_caveat(answer))`` is the
    order that always works; ``numeric_guard.apply_guard`` calls this before it
    appends anything, which is also safe.
    """
    if COMPLETION_MARKER not in (answer or ""):
        return answer
    lines = (answer or "").splitlines()
    while lines and (
        lines[-1].strip().startswith(COMPLETION_MARKER) or not lines[-1].strip()
    ):
        lines.pop()
    return "\n".join(lines).rstrip()


def shared_figures(
    evidence: list,
    filenames: dict[str, str],
    min_sources: int = 2,
) -> list[SharedFigure]:
    """Figures stated by at least `min_sources` distinct sources, best-corroborated
    first. Ties break on order of appearance so the list reads like the evidence.
    """
    seen: dict[tuple[str, str], dict[str, list]] = {}
    order: list[tuple[str, str]] = []
    for result in evidence or []:
        item = getattr(result, "item", None)
        if item is None:
            continue
        source_id = getattr(item, "source_id", "")
        label = _label(result, filenames)
        for figure in figures_in(getattr(item, "content", "") or ""):
            key = figure.key
            if key not in seen:
                seen[key] = {}
                order.append(key)
            bucket = seen[key]
            entry = bucket.setdefault(source_id, {"forms": [], "labels": []})
            entry["forms"].append(figure.render())
            if label not in entry["labels"]:
                entry["labels"].append(label)

    out: list[SharedFigure] = []
    for key in order:
        bucket = seen[key]
        if len(bucket) < min_sources:
            continue
        forms = [form for entry in bucket.values() for form in entry["forms"]]
        surface = Counter(forms).most_common(1)[0][0]
        labels: list[str] = []
        for entry in bucket.values():
            labels.extend(entry["labels"])
        out.append(SharedFigure(surface=surface, value=key[0], labels=tuple(labels)))
    out.sort(key=lambda figure: (-figure.sources,))
    return out


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

@dataclass
class CompletionDetail:
    """What the aid did, for the technical-detail trail."""

    appended: list[str] = field(default_factory=list)   # surfaces added
    present: list[str] = field(default_factory=list)    # surfaces already stated
    sources: int = 0
    reason: str = ""  # appended | complete | no_marker | single_source | no_shared_figure

    @property
    def message(self) -> str:
        if not self.reason or self.reason == "no_marker":
            return ""
        if self.reason == "single_source":
            return "synthesis aid: evidence came from one source, nothing to complete"
        if self.reason == "no_shared_figure":
            return (
                "synthesis aid: no figure is stated by more than one source, "
                "nothing to complete"
            )
        if self.reason == "complete":
            return (
                "synthesis aid: the answer already states every figure the "
                f"evidence corroborates ({', '.join(self.present)})"
            )
        return (
            f"synthesis aid: appended {', '.join(self.appended)} -- each stated "
            f"by two or more of the {self.sources} sources in evidence, but "
            f"missing from the generated answer"
        )


def complete_answer(
    answer: str,
    evidence: list,
    filenames: dict[str, str],
    question: str,
    max_figures: int = 3,
    max_labels: int = 2,
    min_sources: int = 2,
) -> tuple[str, CompletionDetail]:
    """Append corroborated figures the answer omitted. Never removes or rewrites.

    Returns the answer unchanged with a reason when there is nothing to add, so
    the caller can always record the decision.
    """
    if not is_synthesis_query(question):
        return answer, CompletionDetail(reason="no_marker")

    distinct = {
        getattr(getattr(r, "item", None), "source_id", None) for r in evidence or []
    }
    distinct.discard(None)
    if len(distinct) < min_sources:
        return answer, CompletionDetail(
            sources=len(distinct), reason="single_source"
        )

    shared = shared_figures(evidence, filenames, min_sources=min_sources)
    if not shared:
        return answer, CompletionDetail(
            sources=len(distinct), reason="no_shared_figure"
        )

    # A figure the answer already states counts as present whatever its surface:
    # "99.4", "99.4%" and "99.40" are the same fact, so only the value is
    # compared. This is what keeps the aid from repeating itself.
    stated = {figure.value for figure in figures_in(answer or "")}
    missing = [figure for figure in shared if figure.value not in stated]
    present = [figure.surface for figure in shared if figure.value in stated]

    if not missing:
        return answer, CompletionDetail(
            present=present, sources=len(distinct), reason="complete"
        )

    parts: list[str] = []
    for figure in missing[:max_figures]:
        where = "; ".join(figure.labels[:max_labels])
        extra = len(figure.labels) - min(max_labels, len(figure.labels))
        if extra > 0:
            where += f"; +{extra} more"
        parts.append(f"{figure.surface} ({where})")

    line = f"{COMPLETION_MARKER} {', '.join(parts)}."
    completed = f"{(answer or '').rstrip()}\n\n{line}"
    return completed, CompletionDetail(
        appended=[figure.surface for figure in missing[:max_figures]],
        present=present,
        sources=len(distinct),
        reason="appended",
    )
