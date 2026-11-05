"""
diarization.py -- local speaker diarization for transcribed audio segments.

Answers "who said this" for meeting recordings without downloading any model:
each Whisper segment's waveform is summarised into an MFCC statistic vector
(mean + standard deviation of 13 mel-frequency cepstral coefficients), and the
vectors are clustered with agglomerative clustering. The speaker count is
chosen by silhouette score over 2..min(max_speakers, segments), which is the
same family of approach ORION uses (`src/processors/voice_processor.py`) but
without its torchaudio/speechbrain optional paths.

Everything is CPU-only and offline. When librosa or scikit-learn is missing the
module reports unavailable and the pipeline falls back to plain timestamps --
diarization must never be able to break ingestion.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

MIN_SEGMENT_SECONDS = 0.5


def available() -> bool:
    try:
        import librosa  # noqa: F401
        import sklearn.cluster  # noqa: F401
        return True
    except Exception:
        return False


def _segment_features(y: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        import librosa
    except Exception:
        return None
    if len(y) < int(sr * MIN_SEGMENT_SECONDS):
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.size == 0:
        return None
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def diarize_segments(
    path: Path,
    segments: list[dict],
) -> tuple[list[str | None], str | None]:
    """Label each segment with 'Speaker N', ordered by first appearance.

    Returns (labels, warning). Labels align 1:1 with `segments`; a segment too
    short or silent to characterise gets None and is rendered without a
    speaker prefix rather than guessed.
    """
    n = len(segments)
    if n == 0:
        return [], None
    if n == 1:
        return [["Speaker 1"] if segments[0] else None], None

    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [None] * n, f"Speaker diarization skipped: {exc.__class__.__name__}"

    try:
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
    except Exception as exc:
        return [None] * n, f"Speaker diarization skipped: could not decode audio ({exc})"

    features: list[np.ndarray | None] = []
    for segment in segments:
        start = int(float(segment["start"]) * sr)
        end = int(float(segment["end"]) * sr)
        clip = audio[max(0, start):max(0, end)]
        features.append(_segment_features(clip, sr))

    valid_idx = [i for i, f in enumerate(features) if f is not None]
    if len(valid_idx) < 2:
        return [None] * n, "Speaker diarization skipped: not enough voiced segments"

    matrix = np.vstack([features[i] for i in valid_idx])
    matrix = StandardScaler().fit_transform(matrix)

    # Silhouette needs at least k+1 samples for k clusters, so with two voiced
    # segments no split can be scored: report single speaker instead of raising.
    try:
        max_k = min(settings.diarization_max_speakers, len(valid_idx) - 1)
        best_k, best_score = 1, -1.0
        for k in range(2, max_k + 1):
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(matrix)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels))
            if score > best_score:
                best_k, best_score = k, score

        if best_k < 2:
            return (["Speaker 1"] * n,
                    "Single speaker detected - segments carry no speaker prefix.")

        final_labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(matrix)

        # Rename clusters to Speaker 1..K by first appearance in time order.
        order: dict[int, str] = {}
        for position, idx in enumerate(valid_idx):
            cluster = int(final_labels[position])
            if cluster not in order:
                order[cluster] = f"Speaker {len(order) + 1}"
        labels_out: list[str | None] = [None] * n
        for position, idx in enumerate(valid_idx):
            labels_out[idx] = order[int(final_labels[position])]

        logger.info(
            "Diarized %s into %d speakers (silhouette %.3f)",
            path.name, best_k, best_score,
        )
        return labels_out, None
    except Exception as exc:
        # Diarization is an enrichment, never a reason to lose a transcript.
        logger.warning("Diarization failed for %s: %s", path.name, exc)
        return [None] * n, f"Speaker diarization skipped: {exc.__class__.__name__}: {exc}"
