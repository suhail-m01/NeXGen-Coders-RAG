"""
coverage_guard.py -- deterministic numeric/temporal coverage guard.

Named with the _guard suffix on purpose: a bare `coverage.py` shadows the PyPI
`coverage` package that numba (and therefore librosa) imports, which broke the
audio decode path.

The rerank-score gate cannot catch temporal traps: "What was the budget in
2040?" retrieves the 2024 budget chunks and scores them *inside* the range of
genuine questions (measured: +2.55 vs genuine -1.3..+9.7), because the
cross-encoder sees topical overlap and no reason to doubt. This guard adds the
missing axis: a number the question insists on must actually appear in the
evidence, or the system refuses instead of reusing a nearby number.

Default mode checks YEARS only (1800-2099). That is deliberate: years are the
trap class we measured, and year strings are unambiguous in both question and
evidence. `coverage_guard_all_numbers` additionally checks every numeric token
after normalising commas/percent signs - stricter, but able to refuse a
legitimate question whose answer is phrased differently ("1.2 million" vs
"1200000"), so it is off by default.

This is a minimal, deterministic instance of the per-part evidence coverage
idea: cheap, explainable, and wrong in the safe direction.
"""
from __future__ import annotations

import re

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _normalise(token: str) -> str:
    return token.replace(",", "").replace("%", "").strip()


def query_numbers(query: str, all_numbers: bool = False) -> list[str]:
    """Numeric tokens the question commits to, in order of appearance."""
    if all_numbers:
        tokens = NUMBER_RE.findall(query)
    else:
        tokens = YEAR_RE.findall(query)
    seen: list[str] = []
    for token in tokens:
        value = _normalise(token)
        if value and value not in seen:
            seen.append(value)
    return seen


def uncovered_numbers(
    query: str,
    evidence_texts: list[str],
    all_numbers: bool = False,
) -> list[str]:
    """Question numbers that appear nowhere in the evidence.

    An empty list means the guard has no objection. Evidence is searched as
    normalised text so '99.4%' in a document satisfies '99.4' in a question.
    """
    required = query_numbers(query, all_numbers=all_numbers)
    if not required:
        return []
    joined = _normalise(" ".join(evidence_texts or []))
    return [value for value in required if value not in joined]
