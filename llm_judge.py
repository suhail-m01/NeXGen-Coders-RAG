"""llm_judge.py -- LLM-as-judge scoring for the evaluation harness.

Reported ALONGSIDE the deterministic metrics and never instead of them. Keyword
coverage and the number check are reproducible and auditable; a judge is
neither, and it carries biases that have to be measured rather than assumed
away:

  * **Self-preference.** Offline, the judge is the same local model that wrote
    the answer.
  * **Position bias.** Small models pick the first option regardless of content.
    Measured on `qwen2.5:0.5b`: asked three yes/no questions about a correct
    answer, a wrong-figure answer and a non-answer, it replied "A" to all nine.
    Perfectly parseable, perfectly useless.

Position bias is the one this module defends against directly: every binary
question is asked twice with the option order swapped, and the two replies only
count when they select the same *statement*. A model that always says "A"
disagrees with itself on every question, so its scores come back as `None` and
the run reports a consistency rate near zero instead of a flattering 1.0.

Consequence: `judge_consistency_rate` is the number to read first. If it is low,
the model is too small to judge and the other judge columns are noise -- the
report says so explicitly rather than leaving the reader to notice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("evidence_ai.llm_judge")

_JUDGE_PREAMBLE = """You are a strict evaluator for a grounded question-answering system.
You judge ONLY against the evidence given below. You have no other knowledge.
You are NOT answering the question. You are grading the ANSWER that someone
else already wrote. Reply with one letter."""

# name -> (question, positive statement, negative statement)
_BINARY_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "grounded": (
        "Is every factual claim in the ANSWER supported by the EVIDENCE?",
        "every claim in the ANSWER is supported by the EVIDENCE",
        "some claim in the ANSWER is not in the EVIDENCE",
    ),
    "answers_question": (
        "Does the ANSWER directly address the QUESTION?",
        "the ANSWER addresses the QUESTION",
        "the ANSWER describes or copies the source instead of addressing the QUESTION",
    ),
    "conflicts": (
        "Does the ANSWER contradict the EVIDENCE anywhere?",
        "the ANSWER contradicts the EVIDENCE",
        "the ANSWER does not contradict the EVIDENCE",
    ),
}

_QUALITY_QUESTION = (
    "How well does the ANSWER answer the QUESTION using only the EVIDENCE?\n"
    "A = 1, wrong or unusable\n"
    "B = 2, mostly wrong\n"
    "C = 3, partly correct\n"
    "D = 4, correct but incomplete or clumsy\n"
    "E = 5, complete, correct and concise\n"
    "Reply with A, B, C, D or E."
)
_QUALITY_BY_LETTER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

_LETTER = re.compile(r"\b([A-E])\b")

# A reply this short is all the parser needs; the cap is what stops a small
# model from wandering off and answering the question instead of grading it.
_JUDGE_MAX_TOKENS = 6


@dataclass
class Judgement:
    """One answer's judge scores. `None` means no trustworthy verdict: either
    the reply was unparseable or the two orderings disagreed."""

    grounded: bool | None = None
    answers_question: bool | None = None
    conflicts: bool | None = None
    quality: int | None = None
    # Binary questions where both orderings agreed, out of those asked. This is
    # the health metric for the whole judge arm.
    agreements: int = 0
    disagreements: int = 0
    # Raw replies, kept so a surprising score can be audited against what the
    # judge actually said.
    raw: dict[str, str] = field(default_factory=dict)
    parse_failures: list[str] = field(default_factory=list)

    @property
    def consistency(self) -> float | None:
        asked = self.agreements + self.disagreements
        return round(self.agreements / asked, 4) if asked else None

    def as_record(self) -> dict:
        return {
            "judge_grounded": self.grounded,
            "judge_answers_question": self.answers_question,
            "judge_conflicts": self.conflicts,
            "judge_quality": self.quality,
            "judge_consistency": self.consistency,
            "judge_parse_failures": list(self.parse_failures),
            "judge_raw": dict(self.raw),
        }


def _parse_letter(reply: str) -> str | None:
    match = _LETTER.search(reply or "")
    return match.group(1).upper() if match else None


def _parse_quality(reply: str) -> int | None:
    return _QUALITY_BY_LETTER.get(_parse_letter(reply) or "")


def _ask(prompt: str) -> str:
    from llm import ollama_generate

    return (ollama_generate(prompt, max_tokens=_JUDGE_MAX_TOKENS) or "").strip()


def _build_prompt(question: str, answer: str, evidence_text: str, task: str) -> str:
    return (
        f"{_JUDGE_PREAMBLE}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        f"{task}\n"
    )


def _options_prompt(prompt_text: str, first: str, second: str) -> str:
    return f"{prompt_text}\nA = {first}\nB = {second}\nReply with A or B."


def _judge_binary(
    judgement: Judgement, name: str, question: str, answer: str, evidence: str
) -> None:
    """Ask one binary question in both option orders; keep it only if the same
    statement was chosen both times."""
    prompt_text, positive, negative = _BINARY_QUESTIONS[name]
    base = _build_prompt(question, answer, evidence, prompt_text)

    forward = _ask(_options_prompt(base, positive, negative))
    reverse = _ask(_options_prompt(base, negative, positive))
    judgement.raw[f"{name}_forward"] = forward[:40]
    judgement.raw[f"{name}_reverse"] = reverse[:40]

    first_letter, second_letter = _parse_letter(forward), _parse_letter(reverse)
    if first_letter not in ("A", "B") or second_letter not in ("A", "B"):
        judgement.parse_failures.append(name)
        judgement.disagreements += 1
        return

    # Which statement did each ordering select?
    forward_choice = positive if first_letter == "A" else negative
    reverse_choice = negative if second_letter == "A" else positive
    if forward_choice != reverse_choice:
        judgement.disagreements += 1
        judgement.parse_failures.append(f"{name}:order-bias")
        return

    judgement.agreements += 1
    setattr(judgement, name, forward_choice == positive)


def judge_answer(
    question: str,
    answer: str,
    evidence_text: str,
    *,
    max_evidence_chars: int = 4000,
) -> Judgement:
    """Score one answer. Never raises: a judge failure must not lose the
    deterministic results that were already computed."""
    judgement = Judgement()
    if not (answer or "").strip():
        judgement.parse_failures.append("empty answer")
        return judgement

    evidence = (evidence_text or "")[:max_evidence_chars]

    try:
        for name in _BINARY_QUESTIONS:
            _judge_binary(judgement, name, question, answer, evidence)

        if judgement.agreements == 0 and judgement.disagreements:
            # The judge disagreed with itself on every question, so it is not
            # reading the content. A quality score from it would be noise
            # dressed up as a measurement -- skip the call and say why.
            judgement.parse_failures.append("quality:skipped-order-bias")
            return judgement

        reply = _ask(_build_prompt(question, answer, evidence, _QUALITY_QUESTION))
        judgement.raw["quality"] = reply[:40]
        judgement.quality = _parse_quality(reply)
        if judgement.quality is None:
            judgement.parse_failures.append("quality")
    except Exception as exc:  # noqa: BLE001 -- judging is advisory
        logger.warning("LLM judge failed: %s", exc)
        judgement.parse_failures.append(f"error: {exc}")

    return judgement
