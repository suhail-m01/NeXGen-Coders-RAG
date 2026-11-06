"""
lexical.py -- BM25 lexical retrieval leg + reciprocal rank fusion.

Why this exists: dense retrieval alone misses queries whose signal is an exact
token -- record IDs, percentages, names, clause numbers -- because MiniLM
smooths them into neighbourhoods. A sparse BM25 leg recovers those, and
reciprocal rank fusion (RRF) merges the two rankings without needing the two
score scales to be comparable, which they are not (cosine vs BM25 logits).

Ported in spirit from SpectraRAG's `src/rag/hybrid.py` (RRF) and ORION's
`src/retrieval/hybrid.py` (BM25); neither ships it wired into an offline
pipeline the way this is: SpectraRAG needs cloud/GPU components around it and
ORION never connects its HybridRetriever to the served route.

The index is rebuilt lazily when the store changes, detected via a cheap
(count, max rowid) signature rather than re-hashing every document.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

from config import settings
from contextual import indexed_text
from schemas import ContentItem
from vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens, keeping inner . _ - so IDs like 'Q4-2024' survive."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class LexicalIndex:
    """In-memory BM25 over every indexed item.

    Deliberately simple and dependency-free: the corpus is the local index
    (thousands of chunks at most), so an exact in-memory inverted index is
    faster and more predictable than shipping a spare search engine.
    """

    def __init__(self, store: VectorStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1 = k1
        self.b = b
        self._signature: tuple | None = None
        self._items: list[ContentItem] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._dl: list[int] = []
        self._avgdl = 1.0
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------ build
    def _signature_now(self) -> tuple:
        row = self.store.items_signature()
        return tuple(row)

    def ensure_fresh(self) -> None:
        signature = self._signature_now()
        if signature == self._signature:
            return
        self._build()

    def _build(self) -> None:
        items = self.store.all_items()
        tf: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        dl: list[int] = []
        for item in items:
            # The same text the embedding sees, so a situating blurb helps the
            # lexical leg too -- and with no blurb this is item.content.
            tokens = tokenize(indexed_text(item))
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            tf.append(dict(counts))
            dl.append(len(tokens))
            for token in counts:
                df[token] += 1

        n = len(items)
        self._items = items
        self._tf = tf
        self._df = df
        self._dl = dl
        self._avgdl = (sum(dl) / n) if n else 1.0
        self._idf = {
            token: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }
        self._signature = self._signature_now()
        logger.info("BM25 lexical index built over %d items", n)

    # ----------------------------------------------------------- search
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (item_id, bm25_score) ordered by score desc."""
        self.ensure_fresh()
        if not self._items:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores: dict[int, float] = {}
        for token in tokens:
            idf = self._idf.get(token)
            if not idf:
                continue
            for doc_idx, counts in enumerate(self._tf):
                freq = counts.get(token)
                if not freq:
                    continue
                length = self._dl[doc_idx]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / self._avgdl
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (
                    freq * (self.k1 + 1) / denom
                )

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (self._items[idx].item_id, score)
            for idx, score in ordered[:top_k]
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse rankings by RRF: score(id) = sum over lists of 1 / (k + rank).

    Rank is 1-based inside each list. Ids absent from a list contribute
    nothing, which is exactly the behaviour wanted when one leg simply did
    not surface an item.
    """
    kk = k or settings.hybrid_rrf_k
    fused: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for position, item_id in enumerate(ranked, start=1):
            fused[item_id] += 1.0 / (kk + position)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
