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
            return grounded
        return self._finish(grounded)

    def query_stream(self, request: QueryRequest, backend=None):
        """query(), but yielding answer tokens as they are generated.

        Yields str chunks and finally the completed QueryResponse, so a UI can
        render progressively without running the pipeline twice. An early
        abstention yields only the QueryResponse - there are no tokens.
        """
        grounded = self._ground(request, backend)
        if isinstance(grounded, QueryResponse):
            yield grounded
            return
        yield from self._finish_stream(grounded)

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

        timings: dict[str, float] = {}
        started = time.perf_counter()

        text = request.query.strip()
        if not text or len(text) > settings.max_query_chars:
            return QueryResponse(
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
            return QueryResponse(
                answer=f"Could not read the supplied query file: {exc}",
                abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )
        if not representation:
            return QueryResponse(
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

        if not retrieved:
            return QueryResponse(
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
        if best_score < threshold:
            return QueryResponse(
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
            reranked = mmr_select(
                representation, reranked, request.rerank_top_k
            )
            timings["mmr_ms"] = (time.perf_counter() - mark) * 1000

        # 5. Relationship expansion
        mark = time.perf_counter()
        expanded, deterministic = expand_evidence(self.store, reranked)
        semantic = compute_related_to(reranked)
        relationships = deterministic + semantic
        final_evidence = expanded[: settings.max_context_items
                                  + settings.max_relationship_expansion]
        timings["relationships_ms"] = (time.perf_counter() - mark) * 1000

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
                return QueryResponse(
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

    def _assemble(self, state: dict, answer: str) -> QueryResponse:
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

        mark = time.perf_counter()
        citations, _sentence_map = build_citations(
            answer, state["final_evidence"], state["filenames"]
        )
        timings["citations_ms"] = (time.perf_counter() - mark) * 1000

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
            yield self._provider_error_response(state, exc)
            return
        state["timings"]["generation_ms"] = (time.perf_counter() - mark) * 1000
        yield self._assemble(state, "".join(chunks))

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
