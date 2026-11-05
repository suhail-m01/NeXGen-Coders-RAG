"""
mmr.py -- Maximal Marginal Relevance re-selection of reranked evidence.

The reranker ranks by relevance alone, so a question answered by three similar
paragraphs fills the context budget with near-duplicates while other relevant
material is cut off. MMR re-selects from an over-fetched rerank pool:

    MMR(d) = lambda * sim(d, query) - (1 - lambda) * max sim(d, selected)

The best-reranked candidate is always seeded first, so MMR can only ever
*diversify*, never outrank the reranker's top choice. Lambda 1.0 degenerates
to plain rerank order, which is what makes the knob safe to tune.

ORION ships this algorithm (`src/retrieval/mmr.py`) but never wires it into a
served path; here it is part of the query pipeline behind `mmr_enabled`.
"""
from __future__ import annotations

import logging

from config import settings
from embeddings import cosine_similarity, embed_texts
from schemas import RerankedResult

logger = logging.getLogger(__name__)


def mmr_select(
    query: str,
    candidates: list[RerankedResult],
    k: int,
    lambda_param: float | None = None,
    max_per_source: int | None = None,
) -> list[RerankedResult]:
    """Re-select up to `k` candidates for diversity. No-ops when unnecessary.

    `max_per_source` caps how many chunks one source may contribute. Embedding
    diversity alone is not enough: pages of the same report can be mutually
    dissimilar while still crowding out a different modality that the question
    actually needs, which is exactly the cross-modal recall failure this cap
    prevents. When the cap would starve the selection (fewer candidates than
    k across sources) it relaxes automatically.
    """
    lam = settings.mmr_lambda if lambda_param is None else lambda_param
    cap = settings.mmr_max_per_source if max_per_source is None else max_per_source
    if k <= 0 or len(candidates) <= k:
        return list(candidates)

    texts = [c.item.content for c in candidates]
    vectors = embed_texts([query] + texts)
    query_vector = vectors[0]
    candidate_vectors = vectors[1:]

    selected = [0]  # seed with the reranker's best; MMR may only diversify
    remaining = list(range(1, len(candidates)))
    source_counts: dict[str, int] = {candidates[0].item.source_id: 1}

    while len(selected) < k and remaining:
        eligible = [
            i for i in remaining
            if source_counts.get(candidates[i].item.source_id, 0) < cap
        ]
        # Relax the cap rather than return fewer than k candidates.
        pool_now = eligible or remaining
        best_index, best_score = None, float("-inf")
        for index in pool_now:
            relevance = cosine_similarity(query_vector, candidate_vectors[index])
            redundancy = max(
                cosine_similarity(candidate_vectors[index], candidate_vectors[j])
                for j in selected
            )
            score = lam * relevance - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        remaining.remove(best_index)
        source_id = candidates[best_index].item.source_id
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    reselected = [
        RerankedResult(
            item=candidates[i].item,
            retrieval_score=candidates[i].retrieval_score,
            rerank_score=candidates[i].rerank_score,
            rank=position,
        )
        for position, i in enumerate(selected, start=1)
    ]
    logger.info(
        "MMR selected %d of %d candidates (lambda=%.2f)",
        len(reselected), len(candidates), lam,
    )
    return reselected
