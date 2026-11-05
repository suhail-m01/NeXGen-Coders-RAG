"""
retriever.py -- query -> RetrievalResult[].

    query text -> MiniLM embedding -> FAISS cosine search -> ranked results

All modalities compete in one ranked list. That is what makes a cross-modal
query genuinely cross-modal rather than a stitched-together per-modality
search.
"""
from __future__ import annotations

import logging

from config import settings
from embeddings import embed_query
from schemas import Modality, RetrievalResult
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore):
        from lexical import LexicalIndex

        self.store = store
        # One BM25 index for the life of the retriever, not one per query.
        # Constructing it per call threw away its own signature cache and made
        # every question pay an O(corpus) rebuild; ensure_fresh() inside search
        # already rebuilds when the underlying index changes.
        self._lexical = LexicalIndex(store)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        modality_filter: Modality | None = None,
        exclude_source_ids: set[str] | None = None,
        collection: str | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search over the whole index, or one collection of it.

        `modality_filter`, `exclude_source_ids` and `collection` over-fetch then
        filter, so a constrained search still returns up to top_k results rather
        than a truncated handful.

        `exclude_source_ids` is what makes "find sources related to this
        file" meaningful: without it a document query retrieves mostly its
        own chunks, which are trivially its own nearest neighbours.
        """
        k = top_k or settings.top_k
        if not query.strip():
            return []

        excluded = exclude_source_ids or set()
        # One query for the whole scope: filtering per hit would mean a SQLite
        # round trip per candidate.
        allowed_sources = (
            self.store.source_ids_in_collection(collection) if collection else None
        )
        fetch_k = k * 4 if (modality_filter or excluded or allowed_sources is not None) else k
        query_vector = embed_query(query)
        hits = self.store.search(query_vector, fetch_k)

        if settings.hybrid_enabled:
            # Hybrid leg: fuse the dense ranking with a BM25 lexical ranking.
            # The fused score is an RRF score (sum of 1/(k+rank)), NOT a cosine;
            # the two scales are incomparable and RRF deliberately ignores that.
            from lexical import reciprocal_rank_fusion

            lex_hits = self._lexical.search(
                query, max(fetch_k, settings.hybrid_lexical_k)
            )
            dense_ids = [item.item_id for item, _ in hits]
            fused = reciprocal_rank_fusion(
                [dense_ids, [item_id for item_id, _ in lex_hits]]
            )
            dense_by_id = {item.item_id: item for item, _ in hits}

            results: list[RetrievalResult] = []
            for item_id, score in fused:
                item = dense_by_id.get(item_id)
                if item is None:
                    item = self.store.get_item(item_id)
                if item is None:
                    continue
                if modality_filter and item.modality is not modality_filter:
                    continue
                if item.source_id in excluded:
                    continue
                if allowed_sources is not None and item.source_id not in allowed_sources:
                    continue
                results.append(
                    RetrievalResult(item=item, score=score, rank=len(results) + 1)
                )
                if len(results) >= k:
                    break
            return results

        results: list[RetrievalResult] = []
        for item, score in hits:
            if modality_filter and item.modality is not modality_filter:
                continue
            if item.source_id in excluded:
                continue
            if allowed_sources is not None and item.source_id not in allowed_sources:
                continue
            results.append(RetrievalResult(item=item, score=score, rank=len(results) + 1))
            if len(results) >= k:
                break
        return results
