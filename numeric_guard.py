"""Numeric verification guard (post-generation).

The abstention gate decides whether to ask the model at all. This module checks
what came *back*: every figure in the answer must appear in the retrieved
evidence, otherwise the user gets a visible caveat naming it.

The failure mode it exists for is real and was measured: Tesseract misreads a
dashboard, the evidence contains "22 024", and the model answers "the team size
is 22,024" -- a confident, specific, wrong number. No retrieval metric catches
that, because retrieval was perfect.

Design choices, in the order they were argued:
  * It never deletes or refuses an answer. A false positive costs one sentence;
    silently dropping a correct figure costs the user's trust in the tool.
  * Comparison is on normalised values, not raw strings, so "32,000" matches
    "32000" and "99.40" matches "99.4". Reformatting is not hallucination.
  * Numbers the user typed in the question are always allowed: echoing "in
    2024" back is not a claim the evidence has to support.
  * Figures shorter than `numeric_guard_min_digits` are ignored. "3 items" and
    "page 5" are too easy to paraphrase legitimately to police.
"""

from __future__ import annotations

import logging
import re
import threading

from config import settings

logger = logging.getLogger("evidence_ai.numeric_guard")

# Comma-grouped first so "22,024" is one figure, then decimals, then plain runs
# of digits. Order matters: the regex engine takes the first alternative that
# matches at a position.
_FIGURE_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")

# Same pattern with an optional scale word, so "1.2 million" and "1200000" are
# recognised as one figure. `(?![a-z])` keeps units out: "500 KB" is not 500k.
_SCALE_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
    r"(?:\s*(thousand|million|billion|trillion|hundred|lakh|crore|bn|k)(?![a-z]))?",
    re.IGNORECASE,
)
_SCALE_FACTORS = {
    "hundred": 100,
    "thousand": 1_000,
    "k": 1_000,
    "lakh": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Stable marker so the UI and the evaluation harness can recognise the caveat.
NUMERIC_GUARD_MARKER = "Evidence warning:"

_local = threading.local()


def _digits_only(token: str) -> str:
    return re.sub(r"[^\d]", "", token)


def _normalize(token: str) -> str:
    """Canonical form of one figure: '22,024' -> '22024', '99.40' -> '99.4'."""
    cleaned = token.replace(",", "").replace("_", "").strip()
    if "." in cleaned:
        whole, _, fraction = cleaned.partition(".")
        fraction = fraction.rstrip("0")
        cleaned = f"{whole}.{fraction}" if fraction else whole
    # Keep '0' meaningful but drop OCR-style leading zeros ('024' -> '24').
    return cleaned.lstrip("0") or "0"


def _format_number(value: float) -> str:
    """1200000.0 -> '1200000', 1.25 -> '1.25' (no float artefacts)."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _candidates(token: str, scale: str) -> set[str]:
    """Every value one figure could legitimately stand for: its own normalised
    form, plus the scaled form when a scale word followed it."""
    values = {_normalize(token)}
    if scale:
        try:
            values.add(_normalize(_format_number(
                float(token.replace(",", "")) * _SCALE_FACTORS[scale]
            )))
        except (ValueError, KeyError):
            pass
    return values


def figures_in(text: str) -> list[str]:
    """Figures in `text`, in order of appearance, duplicates kept."""
    return _FIGURE_RE.findall(text or "")


def _values_in(text: str) -> set[str]:
    values: set[str] = set()
    for match in _SCALE_RE.finditer(text or ""):
        values |= _candidates(match.group(1), (match.group(2) or "").lower())
    return values


def unsupported_numbers(
    answer: str,
    evidence_texts: list[str],
    question: str = "",
) -> list[str]:
    """Figures asserted in `answer` that no evidence text (nor the question
    itself) supports. Returned in answer order, de-duplicated, raw form."""
    allowed = set()
    for text in list(evidence_texts) + [question]:
        allowed |= _values_in(text)

    seen: set[str] = set()
    unsupported: list[str] = []
    for match in _SCALE_RE.finditer(answer or ""):
        token = match.group(1)
        digits = _digits_only(token)
        if len(digits) < settings.numeric_guard_min_digits:
            continue
        candidates = _candidates(token, (match.group(2) or "").lower())
        if candidates & allowed or candidates & seen:
            continue
        seen |= candidates
        unsupported.append(token)
    return unsupported


def build_caveat(numbers: list[str], sources: list[str]) -> str:
    """The warning appended to the answer. Names the figures and the sources
    that were searched, so a reviewer knows what to re-check and where."""
    listed = ", ".join(numbers)
    plural = "s" if len(numbers) > 1 else ""
    where = f" (searched: {', '.join(sources)})" if sources else ""
    return (
        f"{NUMERIC_GUARD_MARKER} the figure{plural} {listed} could not be found in "
        f"the retrieved evidence{where}. Treat {listed} as unconfirmed and verify "
        f"it against the source."
    )


def strip_caveat(answer: str) -> str:
    """Remove any trailing guard caveat, leaving the claim text alone.

    The evaluation harness uses this so that a *disclaimer* is not scored as a
    hallucinated claim -- the harness must measure what the system asserts, and
    the caveat is the opposite of an assertion.
    """
    if NUMERIC_GUARD_MARKER not in (answer or ""):
        return answer
    lines = answer.splitlines()
    while lines and (
        lines[-1].strip().startswith(NUMERIC_GUARD_MARKER) or not lines[-1].strip()
    ):
        lines.pop()
    return "\n".join(lines).rstrip()


def _set_guard_detail(numbers: list[str], sources: list[str]) -> None:
    _local.numbers = numbers
    _local.sources = sources


def reset_guard_detail() -> None:
    """Clear the thread-local so a previous query's caveat can never be shown
    against the current one. Called at the start of every query."""
    _set_guard_detail([], [])


def last_guard_detail() -> tuple[list[str], list[str]]:
    """(unverified figures, searched sources) from the last answer on this
    thread, or ([], []). The UI reads this to badge the answer without the
    locked QueryResponse schema needing a new field."""
    return getattr(_local, "numbers", []), getattr(_local, "sources", [])


def apply_guard(
    answer: str,
    evidence_texts: list[str],
    sources: list[str],
    question: str = "",
) -> tuple[str, list[str]]:
    """Append a caveat if the answer asserts unsupported figures.

    Returns (answer, unsupported_figures). The answer is unchanged when the
    guard is disabled, nothing is unsupported, or the answer is empty.
    """
    _set_guard_detail([], [])
    if not settings.numeric_guard_enabled or not (answer or "").strip():
        return answer, []

    numbers = unsupported_numbers(answer, evidence_texts, question)
    if not numbers:
        return answer, []

    _set_guard_detail(numbers, sources)
    logger.info("numeric guard flagged %s figure(s): %s", len(numbers), numbers)
    return f"{answer.rstrip()}\n\n{build_caveat(numbers, sources)}", numbers
