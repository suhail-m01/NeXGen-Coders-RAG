"""
api.py -- FastAPI surface over rag_pipeline.

    GET  /health                 runtime and index status
    POST /ingest                 upload and index one or more files
    POST /query                  ask a question (text / image / audio / document)
    POST /query_file             ask using an uploaded image / audio / document
    GET  /sources                list indexed sources
    GET  /source/{source_id}     inspect an indexed source and its items
    GET  /source/{source_id}/file  download the original file
    DELETE /source/{source_id}   remove a source from the index (admin)

Every endpoint delegates to the same RAGPipeline the Streamlit UI uses, so
there is one implementation of the RAG logic, not two.

Every endpoint except /health requires a bearer token:

    Authorization: Bearer eai_...

Tokens are minted per account (scripts\\create_user.py --create-token) and carry
that account's role, so a viewer token cannot ingest and an analyst token cannot
delete. Set API_AUTH_ENABLED=false in .env to turn this off -- which leaves the
API open on whatever interface it is bound to.

Run:  .venv\\Scripts\\uvicorn api:app --port 8000
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from rag_pipeline import get_pipeline
from schemas import (
    ErrorCode,
    InferenceMode,
    QueryModality,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evidence AI",
    description="Offline-runtime multimodal RAG over documents, images and audio.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Local dev frontend, plus your deployed Vercel URL once you have one.
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require(permission: str):
    """FastAPI dependency: resolve a bearer token and check one permission.

    401 for a missing, unknown or revoked token; 403 for a valid token whose
    role is not allowed. The two are kept distinct so a caller can tell "your
    credential is wrong" from "your role cannot do this".
    """
    from auth import can, get_user_store

    def dependency(authorization: str | None = Header(default=None)) -> dict:
        if not settings.api_auth_enabled:
            return {"username": "api-auth-disabled", "role": "admin", "via": "disabled"}
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail={"error": "Missing bearer token.",
                        "hint": "Authorization: Bearer eai_... "
                                "(scripts\\create_user.py --create-token)"},
            )
        account = get_user_store().authenticate_token(authorization.split(" ", 1)[1].strip())
        if account is None:
            raise HTTPException(
                status_code=401,
                detail={"error": "Unknown, revoked or expired token."},
            )
        if not can(account["role"], permission):
            raise HTTPException(
                status_code=403,
                detail={"error": f"Role '{account['role']}' may not {permission}.",
                        "username": account["username"]},
            )
        return account

    return dependency


@app.get("/health")
def health(backend: str | None = None) -> dict:
    """Runtime status for the selected inference backend.

    `backend` is "offline" (local Ollama) or "online" (cloud API); it
    defaults to settings.inference_backend. Online mode is not probed
    against Ollama, so this endpoint works on a host with no Ollama.
    """
    try:
        return get_pipeline().health(backend)
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form("general"),
    replace: bool = Form(False),
    account: dict = Depends(require("ingest")),
) -> dict:
    """Upload and index files. Returns a per-file result rather than failing
    the whole batch when one file is bad.

    `replace=true` removes older indexed versions of the same filename. That is
    a deletion, so it needs the delete permission on top of ingest.
    """
    from auth import can

    pipeline = get_pipeline()
    if replace and not can(account["role"], "delete"):
        raise HTTPException(
            status_code=403,
            detail={"error": f"Role '{account['role']}' may ingest but not replace "
                             "existing sources (needs the delete permission)."},
        )
    results = []

    for upload in files:
        try:
            data = await upload.read()
            if len(data) > settings.max_file_bytes:
                results.append({
                    "filename": upload.filename,
                    "success": False,
                    "error_code": ErrorCode.FILE_TOO_LARGE.value,
                    "error": f"Exceeds {settings.max_file_bytes} bytes.",
                })
                continue

            stored_path = pipeline.store_upload(upload.filename or "upload", data)
            result = pipeline.ingest_file(
                str(stored_path), collection=collection, replace=replace
            )
            results.append({
                "filename": result.source.filename,
                "source_id": result.source.source_id,
                "source_type": result.source.source_type.value,
                "items_indexed": len(result.items),
                "relationships": len(result.relationships),
                "success": result.success,
                "error": result.error,
                "error_code": result.error_code.value if result.error_code else None,
                "warnings": result.warnings,
            })
        except Exception as exc:
            logger.exception("Ingest failed for %s", upload.filename)
            results.append({
                "filename": upload.filename,
                "success": False,
                "error": str(exc),
            })

    return {
        "results": results,
        "indexed": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "index": pipeline.store.stats(),
    }


@app.post("/query")
def query(
    request: QueryRequest,
    backend: str | None = None,
    account: dict = Depends(require("ask")),
) -> dict:
    """Text query against the index.

    `backend` ("offline" | "online") selects the inference provider and
    defaults to settings.inference_backend. It is a query parameter rather
    than a QueryRequest field so the locked schemas.py contract is unchanged.

    The response is the QueryResponse contract plus a sibling
    `technical_detail` string, which carries provider error detail for a
    "Technical details" view without altering QueryResponse itself.
    """
    from rag_pipeline import last_technical_detail

    response = get_pipeline().query(request, backend=backend)
    return {
        **response.model_dump(mode="json"),
        "technical_detail": last_technical_detail(),
    }


@app.post("/query_file", response_model=QueryResponse)
async def query_file(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    query_modality: QueryModality = Form(QueryModality.IMAGE),
    inference_mode: InferenceMode = Form(InferenceMode.LOCAL),
    top_k: int = Form(settings.top_k),
    rerank_top_k: int = Form(settings.rerank_top_k),
    backend: str | None = Form(None),
    collection: str | None = Form(None),
    account: dict = Depends(require("ask")),
) -> QueryResponse:
    """Query using an uploaded image, audio clip, or document as the query.

    `backend` ("offline" | "online") selects the inference provider.
    """
    suffix = Path(file.filename or "query").suffix or ".bin"
    data = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = handle.name

    try:
        return get_pipeline().query(QueryRequest(
            query=instruction or f"Find evidence related to this {query_modality.value}.",
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            inference_mode=inference_mode,
            query_modality=query_modality,
            query_file_path=temp_path,
            collection=collection,
        ), backend=backend)
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.get("/sources")
def list_sources(
    collection: str | None = None,
    account: dict = Depends(require("ask")),
) -> dict:
    sources = get_pipeline().store.list_sources(collection)
    return {
        "count": len(sources),
        "sources": [s.model_dump() for s in sources],
        "collections": get_pipeline().store.list_collections(),
    }


@app.get("/source/{source_id}")
def get_source(source_id: str, account: dict = Depends(require("ask"))) -> dict:
    detail = get_pipeline().get_source_detail(source_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": ErrorCode.SOURCE_NOT_FOUND.value,
                    "source_id": source_id},
        )
    return detail


@app.get("/source/{source_id}/file")
def get_source_file(
    source_id: str, account: dict = Depends(require("ask"))
) -> FileResponse:
    """Serve the original file. Constrained to data/ so a crafted source_id
    cannot read arbitrary paths off the filesystem."""
    pipeline = get_pipeline()
    source = pipeline.store.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404,
                            detail={"error_code": ErrorCode.SOURCE_NOT_FOUND.value})

    path = Path(source.file_path).resolve()
    data_root = settings.data_dir.resolve()
    if not path.is_file() or data_root not in path.parents:
        raise HTTPException(
            status_code=403,
            detail={"error": "Source file is outside the managed data directory."},
        )
    return FileResponse(path, filename=source.filename)


@app.delete("/source/{source_id}")
def delete_source(source_id: str, account: dict = Depends(require("delete"))) -> dict:
    """Remove a source and its items from the index. The original file and any
    chat-history traces survive; see VectorStore.delete_source."""
    pipeline = get_pipeline()
    source = pipeline.store.get_source(source_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": ErrorCode.SOURCE_NOT_FOUND.value,
                    "source_id": source_id},
        )
    removed = pipeline.delete_source(source_id)
    logger.info("%s deleted source %s (%d items)", account["username"], source_id, removed)
    return {
        "deleted": source.filename,
        "source_id": source_id,
        "items_removed": removed,
        "index": pipeline.store.stats(),
        "note": "The original file on disk and any saved chat history were left untouched.",
    }
