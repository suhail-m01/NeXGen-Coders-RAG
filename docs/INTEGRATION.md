# Integrate the Grounding Coverage Ledger

This is an **additive upgrade** for the current `NeXGen-Coders-RAG` repository.
It does not replace the existing ingestion, FAISS index, reranker, model
provider, or API contract.

## 1. Copy the files

Copy these paths from this package into the repository, preserving folders:

- `grounding_audit.py` -> repository root
- `tests/test_grounding_audit.py` -> `tests/`

## 2. Add the strict citation-coverage gate

Open `rag_pipeline.py`.

Find the existing block near the end of `RAGPipeline.query()`:

```python
# 8. Programmatic citations
mark = time.perf_counter()
citations, _sentence_map = build_citations(answer, final_evidence, filenames)
timings["citations_ms"] = (time.perf_counter() - mark) * 1000
```

Replace it with:

```python
# 8. Programmatic citations + sentence-level coverage audit
from grounding_audit import build_grounding_ledger

mark = time.perf_counter()
citations, _sentence_map = build_citations(answer, final_evidence, filenames)
ledger = build_grounding_ledger(answer, _sentence_map)
timings["citations_ms"] = (time.perf_counter() - mark) * 1000

# Fail closed: do not return a factual answer if any answer sentence
# has no citation mapping. The UI/API response schema remains unchanged.
if ledger.status != "complete":
    logger.warning(
        "Grounding coverage incomplete: %s/%s sentences supported",
        ledger.supported_sentences,
        ledger.total_sentences,
    )
    return QueryResponse(
        answer=ABSTENTION_MESSAGE,
        retrieved_items=reranked,
        relationships=relationships,
        abstained=True,
        confidence=confidence,
        error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
        query_representation=representation,
        latency_ms=timings,
    )

timings["total_ms"] = (time.perf_counter() - started) * 1000
```

The existing `citations.py` currently assigns the best-matching evidence item
when a sentence does not meet `CITATION_THRESHOLD`. That fallback can make an
unsupported sentence appear cited. For a genuinely strict gate, open
`citations.py` and remove the fallback:

```python
if not hits:
    hits = [int(np.argmax(similarities))]
```

Replace it with:

```python
if not hits:
    sentence_map[sentence] = []
    continue
```

Also keep the existing `supported.update(hits)` only after this check. This
means unsupported sentences receive no citation IDs, allowing the gate above
to abstain instead of attaching a weak citation.

## 3. Run tests

From the repository root in PowerShell:

```powershell
python -m pytest tests/test_grounding_audit.py -q
```

Then run the existing suite:

```powershell
python -m pytest -q
```

## 4. What this upgrade adds

- A sentence-level coverage ledger.
- Explicit supported / unsupported sentence counts.
- Coverage ratio for evaluation and debugging.
- A fail-closed integration option: if any answer sentence lacks a citation
  mapping, the pipeline abstains rather than presenting it as grounded.
- Unit tests that run without Ollama, Tesseract, Whisper, FAISS model downloads,
  or GPU access.

## Important limitation

This checks citation mapping, not factual entailment. The current citation
builder uses embedding similarity, which can associate semantically similar
but non-entailing evidence. A future stronger version should add claim-to-
evidence entailment checks and human-reviewed evaluation examples. Do not
describe this ledger as proof that every claim is true.
