"""
constraint_guard.py -- hold the answer to the constraints the question actually set.

The score gate and the coverage guard cannot see three failure modes that the
evaluation dataset measures, because in all three the retrieval scores look
healthy:

  * Near-miss period. "How many records were processed in 2023?" retrieves the
    2024 evidence at high confidence -- 2023 does appear in the corpus, as the
    launch year -- and the model answers with 2024's figure. The coverage guard
    is satisfied (the number 2023 exists in the evidence) and the answer is
    still wrong.
  * Near-miss row. "What was the accuracy in Q3 2024?" retrieves the row for
    Q4 2024 alongside the right one, and the more salient figure wins.
  * Wrong entity. "How many engineers were on the Project Beta team?" retrieves
    Project Alpha's six engineers. Nothing in the pipeline checks names, only
    numbers.

Three functions, each deliberately narrow:

  entity_mismatch()   pre-generation. The question names "Project Beta", the
                      evidence only ever says "Project Alpha" -> refuse before
                      spending an LLM call, and say which name the sources do
                      cover. It fires only when the same head noun appears with
                      a DIFFERENT name in the evidence, so an entity that is
                      simply absent falls through to the existing score and
                      coverage gates instead of being guessed about here.

  temporal_violation() post-generation. For each number the answer asserts, find
                      the evidence items that actually contain it, and require
                      that at least one of them mentions the period the question
                      asked about. If every supporting item talks about a
                      different period, the answer is off-topic however well
                      grounded it looks.

  reorder_by_period_keys() presentation only. When the question names a period,
                      evidence items containing that period are shown to the
                      model first. Scores are NOT touched: confidence, the
                      abstention threshold and MMR selection all run on the
                      original ranking, so calibration is unaffected. This
                      exists because a small model answers from EVIDENCE 1 --
                      measured, Row 3 (Q3 2024, 99.0) reranks at +2.76 while the
                      report page containing 99.4 reranks at +4.74, and the model
                      quotes the first block it is given.

All three are config-gated and all three say what they did, either in the
answer's caveat or in the response's technical detail. Nothing here is a
semantic model: these are string and set checks, which is why they can be
trusted to refuse an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A period is a year, optionally qualified by a quarter. Quarters are written
# Q1..Q4 in this corpus, but "first quarter" phrasing is common enough in
# transcripts to be worth normalising.
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
QUARTER = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
ORDINAL_QUARTER = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", re.IGNORECASE
)
_ORDINAL_TO_Q = {
    "first": "Q1", "1st": "Q1", "second": "Q2", "2nd": "Q2",
    "third": "Q3", "3rd": "Q3", "fourth": "Q4", "4th": "Q4",
}

# "Project Beta", "Team Alpha" -- a Title-Case head noun followed by a
# capitalised name. All-caps heads (NLP, OCR, US) are excluded on purpose: they
# are acronyms, not entity heads, and matching them produces nonsense pairs.
NAMED_ENTITY = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][A-Za-z]{2,})\b")

NUMBER_IN_TEXT = re.compile(r"\d+\.\d+|\d[\d,]*")


@dataclass
class EntityDetail:
    """The question asked about one named thing; the evidence talks about another."""
    asked: str
    found: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        others = ", ".join(self.found[:3])
        return (
            f"The indexed sources describe {others}, not {self.asked}, "
            f"so I will not answer for {self.asked}."
        )


@dataclass
class TemporalDetail:
    """The answer's figures come from evidence about a different period."""
    asked: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        asked = ", ".join(self.asked[:3]) or "the period you asked about"
        found = ", ".join(self.found[:3])
        figures = ", ".join(self.numbers[:4])
        return (
            f"The sources I found describe {found or 'a different period'}, "
            f"not {asked}, so the figures they give ({figures}) do not answer "
            f"the question and I will not guess."
        )


@dataclass
class ReorderDetail:
    """What the presentation reordering did, for the technical-detail trail."""
    keys: list[str] = field(default_factory=list)
    promoted: int = 0
    matched: int = 0
    reason: str = ""  # promoted | already_first | all_match | no_match | ""

    @property
    def message(self) -> str:
        if not self.keys or not self.reason:
            return ""
        named = ", ".join(self.keys)
        if self.reason == "promoted":
            return (
                f"period targeting: question names {named}; "
                f"{self.promoted} evidence item(s) mentioning it were shown first"
            )
        if self.reason == "already_first":
            return (
                f"period targeting: question names {named}; the matching "
                f"evidence was already first (order unchanged)"
            )
        if self.reason == "all_match":
            return (
                f"period targeting: question names {named}; all {self.matched} "
                f"evidence items mention it (order unchanged)"
            )
        return (
            f"period targeting: question names {named}; no retrieved evidence "
            f"mentions it (order unchanged)"
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def periods_in(text: str) -> set[str]:
    """Every period reference in a string, normalised.

    "Q3 2024" yields {"Q3 2024", "Q3", "2024"} so that a question phrased with
    the quarter and evidence phrased with the year alone can still be compared
    deliberately rather than by accident: the pair ("Q3", "2024") is what a
    caller checks, and the bare "2024" is what makes an item that only says
    "during 2024" recognisable.
    """
    if not text:
        return set()
    found: set[str] = set()
    years = {m.group(1) for m in YEAR.finditer(text)}
    found |= years

    quarters: set[str] = set()
    for match in QUARTER.finditer(text):
        quarters.add(f"Q{match.group(1)}")
    for match in ORDINAL_QUARTER.finditer(text):
        quarters.add(_ORDINAL_TO_Q[match.group(1).lower()])
    found |= quarters

    # A quarter next to a year is one specific period, so keep the pair too.
    # Proximity rather than a literal pattern, because the pair is written both
    # ways and both spellings: "Q3 2024", "2024 Q3", "third quarter of 2024".
    spans = [
        (m.start(), f"Q{m.group(1)}") for m in QUARTER.finditer(text)
    ] + [
        (m.start(), _ORDINAL_TO_Q[m.group(1).lower()])
        for m in ORDINAL_QUARTER.finditer(text)
    ]
    for quarter_position, quarter in spans:
        for year_match in YEAR.finditer(text):
            if abs(year_match.start() - quarter_position) <= 40:
                found.add(f"{quarter} {year_match.group(1)}")
    return found


def period_keys(text: str) -> list[str]:
    """The period references worth targeting on, most specific first.

    Only used when the question states a period, and only the most specific
    form is targeted: if the question says "Q3 2024", boosting every item that
    mentions 2024 would promote the whole corpus and change nothing.
    """
    periods = periods_in(text)
    paired = sorted(p for p in periods if " " in p)
    if paired:
        return paired
    quarters = sorted(p for p in periods if p.startswith("Q"))
    if quarters:
        return quarters
    return sorted(periods)


def _has_period(text: str, key: str) -> bool:
    """Does `text` mention this period key? Tolerates 'Q3 2024' written as
    'Q3-2024', 'Q3/2024' or 'third quarter of 2024'."""
    if not key:
        return False
    if " " in key:
        quarter, year = key.split(" ", 1)
        pattern = rf"\b{quarter}\s*[-/]?\s*{year}\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
        ordinal = next(
            (w for w, q in _ORDINAL_TO_Q.items() if q == quarter and w.isalpha()), None
        )
        if ordinal and re.search(
            rf"\b{ordinal}\s+quarter\b[^.]{{0,40}}\b{year}\b", text, re.IGNORECASE
        ):
            return True
        return False
    return bool(re.search(rf"\b{re.escape(key)}\b", text, re.IGNORECASE))


def numbers_in(text: str) -> set[str]:
    """Numeric tokens, normalised the way numeric_guard does it: commas removed
    so '1,200,000' and '1200000' are the same figure."""
    return {m.group(0).replace(",", "") for m in NUMBER_IN_TEXT.finditer(text or "")}


def named_entities(text: str) -> list[tuple[str, str]]:
    """(head, name) pairs the question asserts, e.g. ("Project", "Beta").

    A match at the very start of the string is dropped: that capital is a
    sentence boundary, not a proper noun, and "What did" is not an entity.
    """
    out: list[tuple[str, str]] = []
    for match in NAMED_ENTITY.finditer(text or ""):
        if match.start() == 0:
            continue
        head, name = match.group(1), match.group(2)
        if head.lower() in _ENTITY_STOPWORDS or name.lower() in _ENTITY_STOPWORDS:
            continue
        if (head, name) not in out:
            out.append((head, name))
    return out


# Words that are capitalised mid-sentence for reasons that have nothing to do
# with naming a thing, and which would otherwise produce junk entities.
_ENTITY_STOPWORDS = {
    "the", "and", "or", "but", "not", "with", "from", "into", "over", "under",
    "I", "II", "III", "IV", "Page", "Table", "Figure", "Section", "Chapter",
    "Evidence", "Note", "Warning", "Row", "Speaker", "Q1", "Q2", "Q3", "Q4",
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def entity_mismatch(question: str, evidence_texts: list[str]) -> EntityDetail | None:
    """Refuse when the question names something the evidence renames.

    Fires only on a near miss: the head noun must appear in the evidence with a
    different name. An entity that is simply absent is left to the score and
    coverage gates, because "absent" and "the retriever found nothing relevant"
    are already handled and guessing here would produce false refusals.
    """
    entities = named_entities(question)
    if not entities or not evidence_texts:
        return None
    corpus = "\n".join(evidence_texts)
    for head, name in entities:
        if re.search(rf"\b{re.escape(head)}\s+{re.escape(name)}\b", corpus, re.IGNORECASE):
            continue
        siblings = sorted({
            m.group(1) for m in re.finditer(
                rf"\b{re.escape(head)}\s+([A-Z][A-Za-z]{{2,}})\b", corpus
            )
            if m.group(1).lower() != name.lower()
            and m.group(1) not in _ENTITY_STOPWORDS
        })
        if siblings:
            return EntityDetail(asked=f"{head} {name}", found=siblings)
    return None


def temporal_violation(
    question: str,
    answer: str,
    evidence_texts: list[str],
) -> TemporalDetail | None:
    """Refuse when the answer's figures belong to a different period than asked.

    Two tests, applied only when the answer actually asserts a figure:

    1. The answer names a year and it is not the year asked about. "How many
       records were processed in 2023?" answered with "During 2024 the platform
       processed 1.2 million records" is off-topic however well grounded.
    2. No item that contains the answer's figures mentions the period asked
       about. This is per figure and per supporting item, so one correctly
       periodised source is enough: an annual report beside a quarterly table is
       not penalised for the annual row.

    An answer with no figures is left alone -- there is nothing to periodise --
    and so is a figure no evidence contains, which is the numeric guard's case
    (an invented number) rather than a period mismatch.
    """
    asked = periods_in(question)
    if not asked or not answer:
        return None
    # The most specific form of the constraint is the one that must be met: if
    # the question says "Q3 2024", evidence saying only "2024" does not meet it.
    required = set(period_keys(question)) or asked
    asked_years = {p for p in asked if YEAR.fullmatch(p)}

    tokens = numbers_in(answer)
    answer_years = {t for t in tokens if YEAR.fullmatch(t)}
    answer_figures = tokens - answer_years
    if not answer_figures:
        return None

    if asked_years and answer_years and not (asked_years & answer_years):
        return TemporalDetail(
            asked=sorted(required),
            found=sorted(answer_years),
            numbers=sorted(answer_figures),
        )

    supporting = [
        text for text in evidence_texts if numbers_in(text) & answer_figures
    ]
    if not supporting:
        return None
    if any(_has_period(text, key) for text in supporting for key in required):
        return None

    found = sorted({p for text in supporting for p in periods_in(text)} - asked)
    return TemporalDetail(
        asked=sorted(required), found=found or sorted(answer_years),
        numbers=sorted(answer_figures),
    )


def reorder_by_period_keys(results: list, question: str) -> tuple[list, ReorderDetail]:
    """Put evidence that mentions the asked period first. Order only.

    Takes and returns the same objects; nothing is added, removed or rescored.
    A stable partition keeps the reranker's judgement inside each group, so this
    can only promote an item the reranker already liked.
    """
    keys = period_keys(question)
    if not keys or not results:
        return results, ReorderDetail()

    def mentions(item_result) -> bool:
        content = getattr(getattr(item_result, "item", None), "content", "") or ""
        return any(_has_period(content, key) for key in keys)

    matching = [r for r in results if mentions(r)]
    if not matching:
        # Nothing to promote: leave the ranking exactly as the reranker produced
        # it. Reported as no_match so the audit trail does not claim a reorder.
        return results, ReorderDetail(keys=keys, reason="no_match")
    if len(matching) == len(results):
        return results, ReorderDetail(
            keys=keys, matched=len(matching), reason="all_match"
        )

    rest = [r for r in results if not mentions(r)]
    already_first = all(
        results[i] is matching[i] for i in range(len(matching))
    )
    if already_first:
        return results, ReorderDetail(
            keys=keys, matched=len(matching), reason="already_first"
        )
    return matching + rest, ReorderDetail(
        keys=keys, promoted=len(matching), matched=len(matching),
        reason="promoted",
    )
