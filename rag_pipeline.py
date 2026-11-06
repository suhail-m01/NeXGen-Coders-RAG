"""
rag_pipeline.py -- the single orchestrator.

Both the Streamlit UI and the FastAPI service call this module and nothing
else, so there is exactly one implementation of retrieval, reranking,
relationship expansion, abstention, generation, and citation logic.

Canonical query pipeline:

    query (text / image / audio / document)
      -> query representation
      -> embedding
      -> FAISS retrieval          top_k
      -> cross-encoder rerank     rerank_top_k
      -> ABSTENTION GATE          (before the LLM, so it cannot improvise)
      -> deterministic relationship expansion
      -> query-time RELATED_TO
      -> final evidence set       capped
      -> Qwen2.5-VL generation
      -> programmatic citations
      -> QueryResponse
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

from config import settings
from schemas import (
    Citation,
    ContentItem,
    ErrorCode,
    IngestionResult,
    InferenceMode,
    Modality,
    QueryModality,
    QueryRequest,
    QueryResponse,
    Relationship,
    RerankedResult,
    Source,
    SourceType,
)

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the available sources to "
    "answer this question."
)

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


# The answer field of a retrieval-only response. It is a label rather than an
# empty string so that a response produced without generation cannot be mistaken
# for one that generated nothing.
RETRIEVAL_ONLY_MARKER = "[retrieval-only] generation was not run"


def _safe_name(filename: str) -> str:
    """Strip any directory component so an upload cannot traverse paths."""
    return Path(filename).name.replace("\\", "_").replace("/", "_")


# Technical error detail for the most recent query on this thread. Kept
# outside QueryResponse so that schemas.py stays unchanged, and thread-local
# so concurrent API requests cannot read each other's detail. The UI renders
# it behind a "Technical details" expander; it is never the headline.
_local = threading.local()


def _set_technical_detail(detail: str) -> None:
    _local.technical_detail = detail


def last_technical_detail() -> str:
    """Technical detail from the last query on this thread, or ''."""
    return getattr(_local, "technical_detail", "")


def _count(**kwargs) -> None:
    """Record per-stage item counts for the trace (observability.stage_trace).

    Thread-local like the technical detail, so concurrent API requests cannot
    read each other's counts.
    """
    counts = getattr(_local, "stage_counts", None)
    if counts is None:
        counts = _local.stage_counts = {}
    counts.update(kwargs)


def last_stage_counts() -> dict:
    return getattr(_local, "stage_counts", {})


class RAGPipeline:
    """Owns the store and the stateless stages. Models load lazily on first
    use and are then reused for the life of the process."""

    def __init__(self, store=None):
        from retriever import Retriever
        from reranker import Reranker
        from vector_store import VectorStore

        self.store = store or VectorStore()
        self.retriever = Retriever(self.store)
        self.reranker = Reranker()

    # ---------------------------------------------------------------- health

    def health(self, backend=None) -> dict:
        """Readiness for the selected backend.

        Only the selected backend is probed. In online mode this never
        contacts Ollama, so the application is fully usable on a machine
        where Ollama is not installed.
        """
        from providers import Backend, get_provider, resolve_backend

        resolved = resolve_backend(backend)
        provider = get_provider(resolved)
        readiness = provider.readiness()

        tesseract_ok = Path(settings.tesseract_cmd).exists() or bool(
            shutil.which("tesseract")
        )
        health = {
            "status": "ok" if readiness.ready else "degraded",
            "backend": resolved.value,
            "inference": {
                "backend": resolved.value,
                "label": provider.label,
                "model": provider.model_name,
                "ready": readiness.ready,
                "message": readiness.message,
                "technical": readiness.technical,
                "can_autostart": readiness.can_autostart,
            },
            "tesseract": {
                "available": tesseract_ok,
                "detail": settings.tesseract_cmd if tesseract_ok
                else ErrorCode.OCR_NOT_AVAILABLE.value,
            },
            "libreoffice": {"available": Path(settings.soffice_cmd).exists()},
            "models": {
                "llm": provider.model_name,
                "embedding": settings.embedding_model,
                "reranker": settings.reranker_model,
                "whisper": settings.whisper_model_size,
            },
            "index": self.store.stats(),
        }
        # Preserved for existing callers and tests that read health["ollama"].
        # Populated without a probe when online, so online stays Ollama-free.
        if resolved is Backend.OFFLINE:
            health["ollama"] = {
                "available": readiness.ready,
                "detail": readiness.technical or readiness.message,
            }
        else:
            health["ollama"] = {
                "available": False,
                "detail": "Not required in online mode; not checked.",
            }
        return health

    # ------------------------------------------------------------- ingestion

    @staticmethod
    def route(path: Path):
        """Pick the ingestion module for a file extension."""
        suffix = path.suffix.lower()
        if suffix in DOCUMENT_EXTENSIONS:
            from documents import process_document
            return process_document
        if suffix in IMAGE_EXTENSIONS:
            from images import process_image
            return process_image
        if suffix in AUDIO_EXTENSIONS:
            from audio import process_audio
            return process_audio
        return None

    def store_upload(self, filename: str, data: bytes) -> Path:
        """Persist an uploaded file into data/ under a sanitised name."""
        safe = _safe_name(filename)
        suffix = Path(safe).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            target_dir = settings.images_dir
        elif suffix in AUDIO_EXTENSIONS:
            target_dir = settings.audio_dir
        else:
            target_dir = settings.documents_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / safe
        destination.write_bytes(data)
        return destination

    def stale_versions(self) -> dict[str, list]:
        """Filenames with more than one indexed version, for the UI and CLI to
        offer a cleanup."""
        return self.store.duplicate_filenames()

    def ingest_file(
        self,
        path_str: str,
        collection: str = "general",
        replace: bool = False,
    ) -> IngestionResult:
        """Ingest one file end to end: extract, embed, persist, relate.

        `collection` is applied here rather than inside each processor, so every
        modality gets the same grouping behaviour from one place.

        `replace` removes other indexed versions of the same filename first.
        Without it, updating a document leaves the previous version searchable
        and citable beside the new one -- both answer, and the stale one may
        rank higher.
        """
        path = Path(path_str).resolve()
        processor = self.route(path)
        if processor is None:
            return IngestionResult(
                source=Source(
                    source_id="unknown",
                    filename=path.name,
                    source_type=SourceType.UNKNOWN,
                    file_path=str(path),
                ),
                success=False,
                error=f"Unsupported file type '{path.suffix}'.",
                error_code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            )

        result = processor(str(path))
        if not result.success or not result.items:
            return result

        result.source.collection = collection.strip() or "general"

        if replace:
            for stale in self.store.sources_by_filename(result.source.filename):
                if stale.source_id == result.source.source_id:
                    continue  # byte-identical: nothing to replace
                removed = self.store.delete_source(stale.source_id)
                result.warnings.append(
                    f"Replaced the previous version of {stale.filename} "
                    f"({removed} item(s) removed from the index)."
                )

        return self._index_result(result)

    def delete_source(self, source_id: str) -> int:
        """Drop a source and its items from the index. The file on disk and any
        chat-history traces survive; see VectorStore.delete_source."""
        return self.store.delete_source(source_id)

    def _index_result(self, result: IngestionResult) -> IngestionResult:
        from embeddings import embed_items
        from relationships import build_same_source_relationships

        try:
            self.store.add_source(result.source)

            # Parents are built here rather than in each extractor so that every
            # modality gets the same treatment from one place: a document's
            # sections, a transcript's segments, a table's rows. Must run before
            # add_items, which persists each child's parent_id.
            if (
                settings.parent_expansion_enabled
                and settings.chunking_mode.strip().lower() == "structure"
            ):
                from parents import assign_parents

                result.items, parents = assign_parents(result.items)
                if parents:
                    self.store.add_parents(parents)
                    result.parents = parents

            if settings.contextual_retrieval_enabled:
                from contextual import annotate_items

                # Before embedding, and only here: the blurb is an ingest-time
                # annotation, persisted with the item's metadata so it survives a
                # restart instead of being recomputed per query. Turning the flag on
                # without reindexing does nothing to an existing corpus, because
                # there is nothing to prepend.
                annotate_items(
                    result.items,
                    result.source.filename,
                    warnings=result.warnings,
                )

            vectors = embed_items(result.items)
            stored = self.store.add_items(result.items, vectors)

            if stored:
                relationships = list(result.relationships)
                relationships.extend(build_same_source_relationships(stored))
                self.store.add_relationships(relationships)
                result.relationships = relationships
            else:
                result.warnings.append(
                    "Already indexed (identical content); no new vectors added."
                )
            result.items = stored or result.items
            return result
        except Exception as exc:
            logger.exception("Indexing failed")
            result.success = False
            result.error = f"{ErrorCode.INDEX_ERROR.value}: {exc}"
            result.error_code = ErrorCode.INDEX_ERROR
            return result

    def ingest_directory(self, directory: str) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and self.route(path) is not None:
                results.append(self.ingest_file(str(path)))
        return results

    # ------------------------------------------------- query representation

    def build_query_representation(self, request: QueryRequest) -> str:
        """Turn whatever the user supplied into the text that gets embedded.

        For image and audio queries the uploaded file is converted using the
        same local pipeline used at ingestion, so an image query and an
        indexed image live in the same representation space.
        """
        instruction = request.query.strip()

        if request.query_modality is QueryModality.TEXT or not request.query_file_path:
            return instruction

        path = Path(request.query_file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{ErrorCode.FILE_NOT_FOUND.value}: {path}")

        if request.query_modality is QueryModality.IMAGE:
            from images import build_representation, describe_image, run_ocr_detailed
            from ocr import annotate
            from PIL import Image

            with Image.open(path) as handle:
                ocr = run_ocr_detailed(handle.convert("RGB"))
            # Annotated exactly like indexed images, so query-time and
            # index-time OCR stay in the same representation space.
            ocr_text = annotate(ocr.text, ocr)
            try:
                description = describe_image(path)
            except Exception as exc:
                logger.warning("Vision failed for image query: %s", exc)
                description = ""
            representation = build_representation(description, ocr_text)

        elif request.query_modality is QueryModality.AUDIO:
            from audio import transcribe_to_text
            representation = transcribe_to_text(str(path))

        else:  # DOCUMENT
            from documents import process_document
            result = process_document(str(path))
            if not result.success:
                raise ValueError(result.error or "Could not read document query.")
            # Use the leading chunks rather than the whole file: sending an
            # entire document as a query embeds to noise and blows context.
            representation = " ".join(i.content for i in result.items[:5])

        return f"{instruction}\n\n{representation}".strip() if instruction else representation

    # ------------------------------------------------------------------ query

    def query(self, request: QueryRequest, backend=None) -> QueryResponse:
        """Run the shared RAG pipeline and generate with the selected backend.

        `backend` selects offline (local Ollama) or online (cloud API). It is
        a keyword argument rather than a QueryRequest field so that the
        locked schemas.py contract is unchanged; it defaults to
        settings.inference_backend.

        Everything before generation -- retrieval, reranking, relationship
        expansion, the abstention gate -- is identical on both backends and is
        shared verbatim with query_stream() through _ground().
        """
        grounded = self._ground(request, backend)
        if isinstance(grounded, QueryResponse):
            return self._observe(grounded)
        return self._observe(self._finish(grounded))

    def _observe(self, response):
        """Attach the stage trace to a response on its way out.

        Applied at the two funnels (query and query_stream) rather than at each
        return site, so an early abstention carries the same trace as a full
        answer -- and the stages that never ran are listed as not reached, which
        is the point of the artefact.
        """
        if not isinstance(response, QueryResponse):
            return response
        from observability import stage_trace

        counts = dict(last_stage_counts())
        if "outcome" not in counts:
            if response.abstained:
                code = response.error_code.value if response.error_code else "abstained"
                counts["outcome"] = f"abstained ({code})"
            else:
                counts["outcome"] = "answered"
        response.stages = stage_trace(
            response.latency_ms or {}, counts=counts, detail=last_technical_detail()
        )
        return response

    def query_stream(self, request: QueryRequest, backend=None):
        """query(), but yielding answer tokens as they are generated.

        Yields str chunks and finally the completed QueryResponse, so a UI can
        render progressively without running the pipeline twice. An early
        abstention yields only the QueryResponse - there are no tokens.
        """
        grounded = self._ground(request, backend)
        if isinstance(grounded, QueryResponse):
            yield self._observe(grounded)
            return
        yield from self._finish_stream(grounded)

    def query_retrieval_only(self, request: QueryRequest, backend=None) -> QueryResponse:
        """Steps 1-5b and no further: representation, retrieval, rerank, gates.

        Returns a QueryResponse carrying the retrieved items, relationships,
        confidence and timings, with `answer` set to RETRIEVAL_ONLY_MARKER. When a
        pre-generation check refuses the question, that refusal comes back exactly
        as a real query would return it -- same error code, same stage trace.

        This exists for the harness's fast gate (`evaluation.py --retrieval-only`),
        which checks retrieval quality on CPU in seconds with no Ollama and no GPU.
        The only honest way to build such a gate is to run the real code path up to
        where generation would start; a second, parallel retrieval implementation
        would drift, and a gate that measures something other than what ships is
        worse than no gate at all.

        The result is not an answer and must not be shown as one. It has no
        citations, because citations are built by matching answer text to evidence,
        and nothing downstream of generation has run: no numeric guard, no evidence
        completion, no temporal correction, no repetition collapse.
        """
        grounded = self._ground(request, backend)
        if isinstance(grounded, QueryResponse):
            return self._observe(grounded)

        state = grounded
        state["timings"]["total_ms"] = (time.perf_counter() - state["started"]) * 1000
        _count(outcome="retrieval only (generation not run)")
        return self._observe(QueryResponse(
            answer=RETRIEVAL_ONLY_MARKER,
            retrieved_items=state["reranked"],
            relationships=state["relationships"],
            abstained=False,
            confidence=state["confidence"],
            query_representation=state["representation"],
            latency_ms=state["timings"],
        ))

    def _ground(self, request: QueryRequest, backend=None):
        """Steps 1-5b: representation, retrieval, rerank, gates, evidence.

        Returns either a finished QueryResponse (invalid query, no evidence,
        abstention) or a state dict for _finish() to generate from.
        """
        from citations import build_citations
        from llm import pick_inspectable_image
        from numeric_guard import reset_guard_detail
        from providers import ProviderError, get_provider, supports_vision
        from relationships import compute_related_to, expand_evidence
        from reranker import score_to_confidence

        provider = get_provider(backend)
        _set_technical_detail("")  # never leak detail from a previous query
        reset_guard_detail()       # same for the numeric guard's caveat detail
        _local.stage_counts = {}   # same for the stage trace

        timings: dict[str, float] = {}
        started = time.perf_counter()

        def _refuse(**kwargs) -> QueryResponse:
            """An early exit, timed like a full answer.

            Every refusal path goes through here so the trace carries an end-to-end
            duration: how long the system took to decide it would not answer is
            exactly what an operator compares against a slow generation.
            """
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return QueryResponse(**kwargs)

        text = request.query.strip()
        if not text or len(text) > settings.max_query_chars:
            return _refuse(
                answer="Query must be between 1 and "
                       f"{settings.max_query_chars} characters.",
                abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )

        # 1. Query representation
        mark = time.perf_counter()
        try:
            representation = self.build_query_representation(request)
        except Exception as exc:
            return _refuse(
                answer=f"Could not read the supplied query file: {exc}",
                abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )
        if not representation:
            return _refuse(
                answer="Empty query.", abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )
        timings["representation_ms"] = (time.perf_counter() - mark) * 1000

        # 2. Retrieval. A file-based query excludes its own indexed source so
        # that "find sources related to this" returns other material rather
        # than the query file's own chunks.
        mark = time.perf_counter()
        excluded = self._self_source_ids(request)
        retrieved = self.retriever.retrieve(
            representation, top_k=request.top_k, exclude_source_ids=excluded,
            collection=request.collection,
        )
        timings["retrieval_ms"] = (time.perf_counter() - mark) * 1000
        _count(retrieved=len(retrieved))

        if not retrieved:
            return _refuse(
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=representation,
                latency_ms=timings,
            )

        # 3. Reranking. When MMR diversity is on, over-fetch so the diversity
        #    step chooses from a pool rather than reordering a fixed top-k.
        mark = time.perf_counter()
        pool_k = (
            max(request.rerank_top_k * 2, request.rerank_top_k + 2)
            if settings.mmr_enabled
            else request.rerank_top_k
        )
        reranked = self.reranker.rerank(representation, retrieved, top_k=pool_k)
        timings["rerank_ms"] = (time.perf_counter() - mark) * 1000

        best_score = reranked[0].rerank_score if reranked else -99.0
        confidence = score_to_confidence(best_score)
        _count(rerank_pool=len(retrieved), reranked=len(reranked), best_score=best_score)

        # 4. Abstention gate -- before any LLM call. OCR-derived image evidence
        #    scores on a lower scale than text evidence (measured gap of ~6
        #    logits), so a single global threshold would either refuse valid
        #    image evidence or accept text-scale noise; each modality gets its
        #    own calibrated threshold.
        top_modality = reranked[0].item.modality if reranked else None
        threshold = (
            settings.abstention_threshold_image
            if top_modality is Modality.IMAGE
            else settings.abstention_threshold
        )
        _count(threshold=threshold)
        if best_score < threshold:
            return _refuse(
                answer=ABSTENTION_MESSAGE,
                retrieved_items=reranked,
                abstained=True,
                confidence=confidence,
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=representation,
                latency_ms=timings,
            )

        # 4b. Diversity: swap near-duplicate evidence for distinct evidence.
        if settings.mmr_enabled and len(reranked) > request.rerank_top_k:
            from mmr import mmr_select

            mark = time.perf_counter()
            cap = (
                settings.mmr_max_per_source_structure
                if settings.chunking_mode.strip().lower() == "structure"
                else None
            )
            reranked = mmr_select(
                representation, reranked, request.rerank_top_k, max_per_source=cap
            )
            timings["mmr_ms"] = (time.perf_counter() - mark) * 1000
            _count(
                mmr_selected=len(reranked),
                mmr_cap=cap if cap is not None else settings.mmr_max_per_source,
            )

        # 4c. Period targeting. Order only, and deliberately after the
        #     abstention gate and MMR: confidence comes from the reranker's best
        #     score and MMR selects on score, so neither is disturbed. What
        #     changes is which block the model reads first, and a small model
        #     answers from EVIDENCE 1 -- measured, the row for Q3 2024 reranks at
        #     +2.76 while a page containing the more famous 99.4 reranks at
        #     +4.74, and the model quotes the page.
        if settings.period_targeting_enabled:
            from constraint_guard import reorder_by_period_keys

            mark = time.perf_counter()
            reranked, targeting = reorder_by_period_keys(reranked, request.query)
            timings["period_targeting_ms"] = (time.perf_counter() - mark) * 1000
            if targeting.message:
                _set_technical_detail(targeting.message)

        # 4d. Parent expansion. Retrieval stays fine-grained -- that is what
        #     raised precision and MRR when structure-aware chunking was measured
        #     -- but the model reads the complete section each selected chunk came
        #     from. Without it, a four-row table reaches the model as two rows and
        #     "which quarter used the most budget?" becomes a guess. Order and
        #     scores are inherited from the children, so the abstention decision
        #     above and the period targeting beside it still hold.
        if settings.parent_expansion_enabled:
            from parents import expand_to_parents

            mark = time.perf_counter()
            reranked, expansion = expand_to_parents(self.store, reranked)
            timings["parent_expansion_ms"] = (time.perf_counter() - mark) * 1000
            _count(
                evidence_blocks=len(reranked),
                parent_sections=sum(
                    1 for result in reranked
                    if result.item.metadata.get("is_parent")
                ),
            )
            if expansion.message:
                _set_technical_detail(expansion.message)

        # 5. Relationship expansion
        mark = time.perf_counter()
        expanded, deterministic = expand_evidence(self.store, reranked)
        semantic = compute_related_to(reranked)
        relationships = deterministic + semantic
        final_evidence = expanded[: settings.max_context_items
                                  + settings.max_relationship_expansion]
        timings["relationships_ms"] = (time.perf_counter() - mark) * 1000
        _count(final_evidence=len(final_evidence))

        # 5b. Deterministic coverage guard. The score gate cannot see temporal
        #     traps ("budget in 2040?" scores like a genuine question), but a
        #     number the question insists on must exist in the evidence.
        if settings.coverage_guard_enabled:
            from coverage_guard import uncovered_numbers

            mark = time.perf_counter()
            missing = uncovered_numbers(
                representation,
                [e.item.content for e in final_evidence],
                all_numbers=settings.coverage_guard_all_numbers,
            )
            timings["coverage_ms"] = (time.perf_counter() - mark) * 1000
            if missing:
                _set_technical_detail(
                    "coverage guard: question numbers absent from evidence: "
                    + ", ".join(missing)
                )
                return _refuse(
                    answer=(
                        "I found no evidence covering "
                        + ", ".join(missing)
                        + " in the available sources, so I will not guess."
                    ),
                    retrieved_items=reranked,
                    relationships=relationships,
                    abstained=True,
                    confidence=confidence,
                    error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                    query_representation=representation,
                    latency_ms=timings,
                )

        # 5c. Entity near-miss gate. The numeric guard checks figures against
        #     evidence; nothing checked names, so "how many engineers were on the
        #     Project Beta team" was answered with Alpha's six. Refusing here
        #     rather than after generation also saves the LLM call.
        if settings.entity_guard_enabled:
            from constraint_guard import entity_mismatch

            mark = time.perf_counter()
            mismatch = entity_mismatch(
                request.query, [e.item.content for e in final_evidence]
            )
            timings["entity_guard_ms"] = (time.perf_counter() - mark) * 1000
            if mismatch:
                _set_technical_detail(
                    f"entity guard: question asks about {mismatch.asked}; "
                    f"evidence only mentions {', '.join(mismatch.found)}"
                )
                return _refuse(
                    answer=(
                        f"I found no evidence about {mismatch.asked}. The sources "
                        f"indexed here describe {', '.join(mismatch.found[:3])}, "
                        f"so I will not answer for {mismatch.asked}."
                    ),
                    retrieved_items=reranked,
                    relationships=relationships,
                    abstained=True,
                    confidence=confidence,
                    error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                    query_representation=representation,
                    latency_ms=timings,
                )

        filenames = self.filenames_for(final_evidence)

        return {
            "request": request,
            "backend": backend,
            "representation": representation,
            "reranked": reranked,
            "confidence": confidence,
            "timings": timings,
            "started": started,
            "relationships": relationships,
            "final_evidence": final_evidence,
            "filenames": filenames,
            "inspect_image": (
                pick_inspectable_image(final_evidence)
                if (request.inference_mode is InferenceMode.LOCAL_VISION
                    and supports_vision(backend))
                else None
            ),
        }

    def _generate(self, state: dict, stream: bool):
        """Step 6. Answer text, or a chunk generator when streaming."""
        from providers import get_provider

        provider = get_provider(state["backend"])
        args = (
            state["representation"],
            state["final_evidence"],
            state["filenames"],
        )
        kwargs = {"inspect_image": state["inspect_image"]}
        if stream:
            return provider.generate_stream(*args, **kwargs)
        return provider.generate(*args, **kwargs)

    def _assemble(self, state: dict, answer: str, streamed: bool = False) -> QueryResponse:
        """Steps 7-8: model-signalled abstention, then programmatic citations."""
        from citations import build_citations

        timings = state["timings"]
        if "INSUFFICIENT_EVIDENCE" in answer.upper():
            return QueryResponse(
                answer=ABSTENTION_MESSAGE,
                retrieved_items=state["reranked"],
                relationships=state["relationships"],
                abstained=True,
                confidence=state["confidence"],
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=state["representation"],
                latency_ms=timings,
            )

        # Step 7 pre: collapse a looping generation before anything else reads
        #     the answer. Every stage downstream -- the temporal guard, the
        #     completion aid, citations and the numeric guard -- would otherwise
        #     process pages of duplicate text, and the loop's own list ordinals
        #     get reported as figures missing from the evidence.
        if settings.repetition_guard_enabled:
            from repetition_guard import collapse_repetition

            mark = time.perf_counter()
            answer, repetition = collapse_repetition(answer)
            timings["repetition_guard_ms"] = (time.perf_counter() - mark) * 1000
            if repetition.message:
                _set_technical_detail(repetition.message)

        # Step 7a: temporal verification. The answer can be perfectly grounded
        # and still be about the wrong period: every figure in it comes from
        # evidence that mentions a different year or quarter than the question
        # asked. Batch mode replaces the answer with a refusal; a stream has
        # already shown the text to the user, so it gets an explicit correction
        # appended instead of being silently swapped.
        temporal = None
        if settings.temporal_guard_enabled:
            from constraint_guard import temporal_violation

            # Judged against the passages that matched, not the whole context
            # block: parent expansion adds the surrounding section, and a section
            # spanning two years would make an answer about the wrong year look
            # supported. With no parents (fixed chunking) this is the item text.
            passages: list[str] = []
            for result in state["final_evidence"]:
                passages.extend(
                    result.item.metadata.get("retrieved_texts")
                    or [result.item.content]
                )
            mark = time.perf_counter()
            temporal = temporal_violation(
                state["request"].query, answer, passages
            )
            state["timings"]["temporal_guard_ms"] = (time.perf_counter() - mark) * 1000
            if temporal:
                _set_technical_detail(
                    "temporal guard: question asks about "
                    + ", ".join(temporal.asked)
                    + "; the evidence supporting "
                    + ", ".join(temporal.numbers)
                    + " describes "
                    + (", ".join(temporal.found) or "another period")
                )
            if temporal and not streamed:
                state["timings"]["total_ms"] = (
                    time.perf_counter() - state["started"]
                ) * 1000
                return QueryResponse(
                    answer=(
                        f"The sources describe {', '.join(temporal.found[:3]) or 'a different period'}, "
                        f"not the {', '.join(temporal.asked[:2])} you asked about, "
                        f"so their figures ({', '.join(temporal.numbers[:3])}) do not "
                        f"answer this question and I will not guess."
                    ),
                    retrieved_items=state["reranked"],
                    relationships=state["relationships"],
                    abstained=True,
                    confidence=state["confidence"],
                    error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                    query_representation=state["representation"],
                    latency_ms=state["timings"],
                )

        # Step 7b: evidence completion. A broad question retrieves several
        #     sources that state the same figures, and a small model answers from
        #     the first block. Measured: for such a question the answer came back
        #     byte-identical with and without a prompt-side digest of the shared
        #     figures, so the aid is applied to the answer instead, where it does
        #     not depend on the model following an instruction. Appends only
        #     figures two or more sources state independently, each labelled with
        #     the file and location; skipped when the temporal guard fired,
        #     because that answer is already flagged as the wrong period.
        if settings.synthesis_completion_enabled and not temporal:
            from synthesis import complete_answer

            mark = time.perf_counter()
            answer, completion = complete_answer(
                answer,
                state["final_evidence"],
                state["filenames"],
                state["request"].query,
            )
            timings["synthesis_ms"] = (time.perf_counter() - mark) * 1000
            if completion.message:
                _set_technical_detail(completion.message)

        mark = time.perf_counter()
        citations, _sentence_map = build_citations(
            answer, state["final_evidence"], state["filenames"]
        )
        timings["citations_ms"] = (time.perf_counter() - mark) * 1000
        _count(citations=len(citations))

        # Step 7b: numeric verification guard. It runs after citations and its
        # caveat is appended afterwards, so a disclaimer can neither attract a
        # citation of its own nor be scored as a claim by the harness.
        from numeric_guard import apply_guard

        answer, _unverified = apply_guard(
            answer,
            [result.item.content for result in state["final_evidence"]],
            sorted(set(state["filenames"].values())),
            question=state["request"].query,
        )

        if temporal and streamed:
            # Already on screen: correct it in place rather than pretend it was
            # never said.
            answer = (
                f"{answer}\n\nCorrection: those figures describe "
                f"{', '.join(temporal.found[:3]) or 'a different period'}, not the "
                f"{', '.join(temporal.asked[:2])} asked about."
            )

        timings["total_ms"] = (time.perf_counter() - state["started"]) * 1000

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_items=state["reranked"],
            relationships=state["relationships"],
            abstained=False,
            confidence=state["confidence"],
            query_representation=state["representation"],
            latency_ms=timings,
        )

    def _provider_error_response(self, state: dict, exc) -> QueryResponse:
        # The headline stays user-facing. The raw detail travels through
        # last_technical_detail() rather than QueryResponse, so the locked
        # schemas.py contract needs no new field.
        _set_technical_detail(exc.technical)
        return QueryResponse(
            answer=exc.user_message,
            retrieved_items=state["reranked"],
            relationships=state["relationships"],
            abstained=True,
            confidence=state["confidence"],
            error_code=exc.code,
            query_representation=state["representation"],
            latency_ms=state["timings"],
        )

    def _finish(self, state: dict) -> QueryResponse:
        """Batch completion: generate, then assemble. Never a generator."""
        from providers import ProviderError

        mark = time.perf_counter()
        try:
            answer = self._generate(state, stream=False)
        except ProviderError as exc:
            return self._provider_error_response(state, exc)
        state["timings"]["generation_ms"] = (time.perf_counter() - mark) * 1000
        return self._assemble(state, answer)

    def _finish_stream(self, state: dict):
        """Streaming completion: yield chunks, then the QueryResponse."""
        from providers import ProviderError

        chunks: list[str] = []
        mark = time.perf_counter()
        try:
            for chunk in self._generate(state, stream=True):
                chunks.append(chunk)
                yield chunk
        except ProviderError as exc:
            yield self._observe(self._provider_error_response(state, exc))
            return
        state["timings"]["generation_ms"] = (time.perf_counter() - mark) * 1000
        yield self._observe(self._assemble(state, "".join(chunks), streamed=True))

    # ----------------------------------------------------------------- lookup

    def _self_source_ids(self, request: QueryRequest) -> set[str]:
        """Source IDs to exclude from a file-based query's own results.

        Because source_id is a content hash, an uploaded copy of an already
        indexed file resolves to the same ID and is correctly excluded.
        """
        if not request.query_file_path or request.query_modality is QueryModality.TEXT:
            return set()
        try:
            from schemas import content_hash

            path = Path(request.query_file_path)
            if not path.exists():
                return set()
            return {content_hash(path.read_bytes())}
        except Exception as exc:
            logger.warning("Could not hash query file: %s", exc)
            return set()

    def filenames_for(self, evidence: list[RerankedResult]) -> dict[str, str]:
        names: dict[str, str] = {}
        for result in evidence:
            source_id = result.item.source_id
            if source_id not in names:
                source = self.store.get_source(source_id)
                names[source_id] = source.filename if source else "unknown"
        return names

    def get_source_detail(self, source_id: str) -> dict | None:
        source = self.store.get_source(source_id)
        if source is None:
            return None
        items = self.store.get_items_by_source(source_id)
        return {
            "source": source.model_dump(),
            "item_count": len(items),
            "items": [item.model_dump() for item in items],
        }


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Process-wide singleton so models and the index load only once."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
