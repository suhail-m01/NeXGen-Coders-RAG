"""Repetition collapse for looping generations.

Why this exists
---------------
Measured on the shipped model (qwen2.5:0.5b, temperature 0.0, seed 42): three
identical calls for the same question returned two different answers, and the
second one was a 3,114-character loop -- a numbered list whose items 2 and 3
repeat as items 4 and 5, 6 and 7, and so on until the token cap stops it:

    3. **Privacy Risk**: Sensitive organizational files may need to leave...
    4. **Limited Explainability**: Generated answers often lack accurate...
    5. **Privacy Risk**: Sensitive organizational files may need to leave...
    6. **Limited Explainability**: Generated answers often lack accurate...

Deterministic sampling does not prevent this; the same prompt can land in a loop
or not depending on prompt-cache and batching state inside the server. Two
consequences follow, and this module addresses the first:

* the user reads pages of duplicate text, and the loop's list ordinals (10, 11,
  12 ...) are then reported by the numeric guard as figures missing from the
  evidence -- a warning about numbering, not about facts. ``numeric_guard`` now
  ignores enumeration markers for that reason;
* generation runs until the context fills, which on two CPUs took more than five
  minutes. ``ollama_num_predict`` caps it.

Collapsing is done here rather than by raising ``repeat_penalty`` because a
sampling change re-rolls every answer in the system and would have to be
re-measured against the whole evaluation set. Cutting a loop is deterministic,
touches only pathological output, and cannot invent anything: every character it
keeps was already in the answer.

Rule
----
Paragraphs are compared after normalisation (case, markdown emphasis, list
ordinals). A paragraph of six or more words that repeats counts once; the answer
is cut at the *first* such repeat only when at least three repeats follow, so a
single echoed sentence -- ordinary prose -- is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A paragraph must be this long to count as a repeat: short lines ("Yes.", "87%")
# recur legitimately in lists and tables.
_MIN_WORDS = 6
# How many repeated paragraphs make it a loop rather than an echo.
_LOOP_BLOCKS = 3

_MARKDOWN_RE = re.compile(r"[*_`>#]+")
_ORDINAL_RE = re.compile(r"^\s*\d{1,3}\s*[.):\-]\s*")
_WS_RE = re.compile(r"\s+")


@dataclass
class RepetitionDetail:
    """What the collapse did, for the technical-detail trail."""

    dropped_blocks: int = 0
    dropped_chars: int = 0
    first_repeat: str = ""
    reason: str = ""  # collapsed | ""

    @property
    def message(self) -> str:
        if self.reason != "collapsed":
            return ""
        snippet = self.first_repeat[:60] + ("..." if len(self.first_repeat) > 60 else "")
        return (
            f"repetition guard: the model began repeating itself "
            f"({self.dropped_blocks} duplicate paragraph(s), {self.dropped_chars} "
            f"characters removed); the answer was cut at the first repeat: {snippet}"
        )


def _normalise(block: str) -> str:
    """Compare paragraphs by content, not by numbering or emphasis."""
    text = _ORDINAL_RE.sub("", block.strip())
    text = _MARKDOWN_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip().lower()


def _paragraphs(answer: str) -> list[tuple[int, str]]:
    """(start offset, text) for each non-empty line block, in order."""
    out: list[tuple[int, str]] = []
    for match in re.finditer(r"[^\n]+", answer or ""):
        text = match.group(0).strip()
        if text:
            out.append((match.start(), text))
    return out


def collapse_repetition(
    answer: str,
    min_words: int = _MIN_WORDS,
    loop_blocks: int = _LOOP_BLOCKS,
) -> tuple[str, RepetitionDetail]:
    """Cut a looping answer at its first duplicated paragraph.

    Returns the answer unchanged unless the repetition is sustained, so ordinary
    prose -- including one echoed sentence -- is never edited.
    """
    if not (answer or "").strip():
        return answer, RepetitionDetail()

    paragraphs = _paragraphs(answer)
    counts: dict[str, int] = {}
    first_repeat: int | None = None
    repeats = 0
    for index, (_start, text) in enumerate(paragraphs):
        key = _normalise(text)
        if len(key.split()) < min_words:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            repeats += 1
            if first_repeat is None:
                first_repeat = index

    if first_repeat is None or repeats < loop_blocks:
        return answer, RepetitionDetail()

    cut = paragraphs[first_repeat][0]
    kept = (answer or "")[:cut].rstrip()
    dropped_text = (answer or "")[cut:]
    dropped_blocks = sum(
        1 for _start, text in paragraphs[first_repeat:]
        if len(_normalise(text).split()) >= min_words
    )
    return kept, RepetitionDetail(
        dropped_blocks=dropped_blocks,
        dropped_chars=len(dropped_text),
        first_repeat=paragraphs[first_repeat][1],
        reason="collapsed",
    )
