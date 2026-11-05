"""Confidence-aware OCR.

`pytesseract.image_to_string` throws away the only signal that tells us whether
the text can be trusted: a per-word confidence. For an evidence tool that is the
wrong trade -- a mangled dashboard digit read back as "team size is 22,024" is
worse than admitting the page could not be read.

Measured on the demo dashboard (`data/images/project_dashboard.png`), the two
token classes behave completely differently:

    prose tokens  ("PROJECT", "accuracy", "deployed")   86 - 94
    digit tokens  ("1.2", "520k", "22024", "99.4%")      0 - 75, mean 52.6

The digit that caused a confident hallucination -- "22024", which is the team
size "22" fused with the year "2024" -- scored 51. So digits get a higher bar
than prose: a misread letter still reads as noise, a misread digit silently
becomes a different fact.

Two thresholds, deliberately separate:
  * `ocr_min_word_confidence` -- prose words below this are dropped.
  * `ocr_min_digit_confidence` -- tokens containing a digit need this much.

Nothing here raises. OCR is a supporting signal: when it cannot be trusted we
say so, and ingestion continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger("evidence_ai.ocr")

# Stable marker so the UI and the evaluation harness can recognise (and strip)
# the caveat without guessing at wording.
LOW_CONFIDENCE_MARKER = "[low-confidence OCR]"


@dataclass
class OcrResult:
    """One OCR pass over one image or page."""

    text: str = ""
    # Mean Tesseract confidence of the KEPT words, 0-100.
    confidence: float = 0.0
    # Candidate words reported with a usable confidence, before filtering.
    words_total: int = 0
    words_kept: int = 0
    # Numeric tokens withheld because Tesseract was not confident in them,
    # out of all the numeric tokens it saw. These two are the counts that
    # matter for hallucination risk.
    digits_dropped: int = 0
    digits_total: int = 0
    # True when the plain grayscale pass was empty and Otsu binarisation had to
    # be used -- useful when diagnosing a low-contrast scan.
    used_fallback: bool = False
    # How far the image was upscaled before OCR (1 = untouched). Recorded
    # because it explains otherwise surprising confidence numbers.
    upscale: int = 1
    # The passage should not be quoted verbatim: either the surviving text is
    # mediocre overall, or numbers had to be withheld from it.
    unreliable: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def _to_gray(image):
    """Accept a PIL image or a numpy array; return a single-channel uint8 array."""
    import numpy as np

    if hasattr(image, "convert"):  # PIL.Image
        import cv2

        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        import cv2

        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return array


def _prepare(gray):
    """Upscale small renders toward ocr_target_longest_px.

    Returns (image, factor). Page segmentation stays on Tesseract's automatic
    default (--psm 3): measured on the dashboard it scored 90.5% versus 88.8%
    for sparse-text mode at the same scale, and automatic segmentation is the
    safer choice for dense scanned pages.
    """
    import cv2

    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= 0 or longest >= settings.ocr_target_longest_px:
        return gray, 1
    factor = round(settings.ocr_target_longest_px / longest)
    factor = max(2, min(factor, settings.ocr_max_upscale))
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC), factor


def _required_confidence(word: str) -> float:
    """Digits are held to a higher standard than prose. See module docstring."""
    if not settings.ocr_confidence_enabled:
        return -1.0
    if any(char.isdigit() for char in word):
        return settings.ocr_min_digit_confidence
    return settings.ocr_min_word_confidence


def _pass(gray) -> OcrResult:
    """One Tesseract `image_to_data` pass: filter weak words, keep reading order."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    lines: dict[tuple, list[str]] = {}
    words_total = 0
    words_kept = 0
    digits_total = 0
    digits_dropped = 0
    conf_sum = 0.0

    for i, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:  # Tesseract uses -1 for structural rows
            continue
        words_total += 1
        is_digit_token = any(char.isdigit() for char in word)
        if is_digit_token:
            digits_total += 1
        if conf < _required_confidence(word):
            if is_digit_token:
                digits_dropped += 1
            continue
        words_kept += 1
        conf_sum += conf
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    # Sorting on (block, paragraph, line) reproduces reading order for the
    # column-free layouts this project handles (screenshots, dashboards, scans).
    text = "\n".join(" ".join(lines[key]) for key in sorted(lines))
    return OcrResult(
        text=text.strip(),
        confidence=(conf_sum / words_kept) if words_kept else 0.0,
        words_total=words_total,
        words_kept=words_kept,
        digits_dropped=digits_dropped,
        digits_total=digits_total,
    )


def ocr_with_confidence(image) -> OcrResult:
    """OCR one image/page. Never raises: on any failure returns an empty result."""
    try:
        import cv2

        prepared, factor = _prepare(_to_gray(image))
        result = _pass(prepared)
        result.upscale = factor
        if not result.ok:
            _, otsu = cv2.threshold(
                prepared, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            fallback = _pass(otsu)
            fallback.used_fallback = True
            fallback.upscale = factor
            result = fallback
        if settings.ocr_confidence_enabled and result.ok:
            digits_lost = (
                result.digits_dropped / result.digits_total
                if result.digits_total
                else 0.0
            )
            result.unreliable = (
                result.confidence < settings.ocr_low_confidence_threshold
                or digits_lost >= settings.ocr_digit_drop_ratio
            )
        return result
    except Exception as exc:  # noqa: BLE001 -- OCR must never break ingestion
        logger.warning("Confidence-aware OCR failed: %s", exc)
        return OcrResult()


def low_confidence_note(result: OcrResult) -> str:
    """The caveat prepended to weak OCR text, or '' when the passage is fine.

    Aimed at two readers at once: the LLM (so it stops quoting figures it cannot
    actually see, and does not treat a withheld number as an absent one) and the
    human reviewing the evidence panel.
    """
    if not settings.ocr_confidence_enabled or not result.unreliable:
        return ""
    parts = [
        f"{LOW_CONFIDENCE_MARKER} mean word confidence {result.confidence:.0f}% "
        f"({result.words_kept}/{result.words_total} words readable)."
    ]
    if result.digits_dropped:
        parts.append(
            f"{result.digits_dropped} of {result.digits_total} numeric token(s) were "
            f"withheld as unreadable, so a missing figure below means 'not legible', "
            f"never 'not present'."
        )
    parts.append(
        "Do not quote exact figures from this passage without corroboration."
    )
    return " ".join(parts)


def annotate(text: str, result: OcrResult) -> str:
    """Prefix `text` with the low-confidence caveat when one applies."""
    note = low_confidence_note(result)
    return f"{note}\n{text}" if note else text


def warning_for(result: OcrResult) -> str:
    """One-line ingestion warning, or '' when the OCR was trustworthy."""
    if not settings.ocr_confidence_enabled or not result.unreliable:
        return ""
    dropped = (
        f", withheld {result.digits_dropped}/{result.digits_total} numeric tokens"
        if result.digits_dropped
        else ""
    )
    return (
        f"OCR confidence {result.confidence:.0f}% (kept {result.words_kept}/"
        f"{result.words_total} words{dropped}) -- figures in this text may be "
        f"misread or missing."
    )
