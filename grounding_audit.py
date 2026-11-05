"""
Grounding audit utilities for Evidence AI.

This module does not generate answers or invent citations. It evaluates whether
the existing programmatic citation builder assigned at least one citation to
each answer sentence. Integrate the gate in rag_pipeline.py as documented.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_answer_sentences(answer: str) -> list[str]:
    """Keep sentence splitting consistent with citations.py."""
    return [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(answer.strip())
        if len(sentence.strip()) > 15
    ]


@dataclass(frozen=True)
class SentenceCoverage:
    sentence: str
    citation_ids: tuple[int, ...]
    supported: bool


@dataclass(frozen=True)
class GroundingLedger:
    status: str
    total_sentences: int
    supported_sentences: int
    unsupported_sentences: int
    coverage_ratio: float
    sentences: tuple[SentenceCoverage, ...]
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_grounding_ledger(
    answer: str,
    sentence_to_citation_ids: Mapping[str, Sequence[int]],
) -> GroundingLedger:
    """
    Build a transparent per-sentence coverage ledger.

    A sentence is counted as supported only when the citation builder supplied
    one or more citation IDs for that exact sentence. This is a traceability
    check, not a proof that semantic similarity equals factual entailment.
    """
    sentences = split_answer_sentences(answer)
    rows: list[SentenceCoverage] = []

    for sentence in sentences:
        ids = tuple(
            sorted({
                int(value)
                for value in sentence_to_citation_ids.get(sentence, ())
                if isinstance(value, int) and value > 0
            })
        )
        rows.append(
            SentenceCoverage(
                sentence=sentence,
                citation_ids=ids,
                supported=bool(ids),
            )
        )

    total = len(rows)
    supported = sum(1 for row in rows if row.supported)
    unsupported = total - supported
    ratio = round(supported / total, 4) if total else 0.0

    return GroundingLedger(
        status="complete" if total > 0 and unsupported == 0 else "incomplete",
        total_sentences=total,
        supported_sentences=supported,
        unsupported_sentences=unsupported,
        coverage_ratio=ratio,
        sentences=tuple(rows),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
