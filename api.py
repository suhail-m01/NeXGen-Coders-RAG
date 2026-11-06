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

import json
import logging
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from rag_pipeline import get_pipeline
from rate_limit import client_key, get_limiter
from vector_store import DEFAULT_COLLECTION
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


def rate_limited(bucket: str, permission: str = "ask"):
    """FastAPI dependency: authenticate, then take a token from one rate bucket.

    Returns the same account dict `require(permission)` does, so a route swaps
    `Depends(require("ask"))` for `Depends(rate_limited("queries", "ask"))` and
    gains the limit without touching anything else.

    Authentication runs first, which is deliberate: an unauthenticated caller gets
    401, never 429, so this endpoint cannot be used to discover which routes are
    rate limited without holding a credential. The refusal is 429 with a
    `Retry-After` header and a body that says what to do instead, because a bare
    429 tells an operator nothing except that something is refusing them.
    """
    authorized = require(permission)

    async def dependency(
        request: Request, account: dict = Depends(authorized)
    ) -> dict:
        limiter = get_limiter()
        cost = 1
        if bucket == "ingest":
            # One token per uploaded file, capped at the burst size in the limiter:
            # a request that carries ten documents costs ten times a question, not
            # the same as one.
            form = await request.form()
            cost = len(form.getlist("files")) or 1
        allowed, retry_after = limiter.check(bucket, client_key(request, account), cost=cost)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": f"Too many {bucket} requests from this client.",
                    "retry_after_seconds": round(retry_after, 2),
                    "bucket": bucket,
                    "charged": cost,
                    "hint": "These routes run local models and each call costs seconds of "
                            "compute, so the limit protects the machine rather than a quota. "
                            "Wait the retry period, or run bulk work through "
                            "scripts/reindex.py, which indexes from disk and never touches HTTP.",
                },
                headers={"Retry-After": str(max(1, int(round(retry_after))))},
            )
        return account

    return dependency


# Process start, for uptime in /health. The manifest itself is built once: it
# describes configuration and code version, neither of which changes at runtime,
# and building it runs two git subprocesses that a health poll should not repeat.
_STARTED = time.time()
_MANIFEST: dict | None = None


def _manifest(with_index: bool = True) -> dict:
    """The cached run manifest, plus current index counts when asked."""
    global _MANIFEST
    if _MANIFEST is None:
        from observability import build_manifest

        _MANIFEST = build_manifest()
    if not with_index:
        return _MANIFEST
    out = dict(_MANIFEST)
    try:
        out["index"] = get_pipeline().store.stats()
    except Exception as exc:
        out["index"] = {"error": str(exc)}
    return out


@app.get("/manifest")
def manifest(account: dict = Depends(require("ask"))) -> dict:
    """Everything needed to reproduce a result: commit, models, every retrieval,
    chunking, guard and generation setting, auth mode, index counts, dataset
    version. The same dict is embedded in each evaluation report, so a number in
    a report can be traced back to the build that produced it.
    """
    return _manifest()


@app.get("/health")
def health(backend: str | None = None) -> dict:
    """Runtime status for the selected inference backend.

    `backend` is "offline" (local Ollama) or "online" (cloud API); it
    defaults to settings.inference_backend. Online mode is not probed
    against Ollama, so this endpoint works on a host with no Ollama.

    Beyond dependency readiness this reports the build (commit, models, guard
    flags), host headroom (CPU, RAM, disk, uptime) and index size, because
    "the service answered 200" is not the same as "the service can still write
    to its index". Nothing here loads a model or calls the network beyond the
    backend probe the pipeline already performs.
    """
    try:
        data = get_pipeline().health(backend)
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(exc)}

    manifest = _manifest(with_index=False)
    platform_info = manifest["platform"]
    notes: list[str] = []
    disk_free = platform_info.get("disk_free_mb")
    if disk_free is not None and disk_free < 200:
        notes.append(
            f"Only {disk_free:.0f} MB of free disk: ingestion and index writes "
            "may fail."
        )
    ram = platform_info.get("ram_mb") or {}
    if ram.get("available") is not None and ram["available"] < 512:
        notes.append(
            f"Only {ram['available']:.0f} MB of RAM available: local model "
            "loading may be killed by the OS."
        )
    if data.get("status") == "ok" and notes:
        data["status"] = "degraded"
    data["runtime"] = {
        "git_commit": manifest["application"]["git_commit"],
        "git_dirty": manifest["application"]["git_dirty"],
        "python": platform_info.get("python"),
        "platform": platform_info.get("system"),
        "cpu_count": platform_info.get("cpu_count"),
        "ram_mb": ram,
        "disk_free_mb": disk_free,
        "uptime_s": round(time.time() - _STARTED, 1),
        "auth_enabled": settings.api_auth_enabled,
        "rate_limit": get_limiter().snapshot(),
        "chunking_mode": settings.chunking_mode,
        "dataset_version": manifest["evaluation"]["dataset_version"],
        "guards": manifest["guards"],
        "generation": manifest["generation"],
        "notes": notes,
    }
    return data


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form("general"),
    replace: bool = Form(False),
    account: dict = Depends(rate_limited("ingest", "ingest")),
) -> dict:
    """Upload and index files. Returns a per-file result rather than failing
    the whole batch when one file is bad.

    `replace=true` removes older indexed versions of the same filename. That is
    a deletion, so it needs the delete permission on top of ingest.

    Rate limited at `rate_limit_ingest_per_minute`, charged one token per file in
    the request, because embedding is the expensive part and a single call can
    carry many documents.
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


def _record_turn(request, response, backend, session_id: str | None) -> str | None:
    """Append one question/answer pair to chat history, creating a session if needed.

    Returns the session id the turn was written to, or None when recording was not
    requested. This mirrors `app.save_turn` exactly -- same trace, same unverified
    figures -- so an answer recorded through the API replays in the Streamlit UI
    and one recorded in the UI replays through `/history/sessions/{id}`.

    Recording is opt-in per request. An evaluation run, a curl probe or a monitor
    that does not pass `session_id` leaves no trace behind, which is what keeps
    the history a record of questions somebody actually asked.
    """
    if not session_id:
        return None
    from chat_history import build_trace, get_history
    from numeric_guard import last_guard_detail

    pipeline = get_pipeline()
    history = get_history()
    known = history.get_session(session_id)
    if known is None:
        # An id the client believes in but the database does not (a deleted
        # session, a copy-paste) starts a fresh one rather than failing the query.
        session_id = history.create_session(
            backend=getattr(backend, "value", str(backend)) if backend else None
        )
    history.add_user_message(session_id, request.query)
    flagged, _searched = last_guard_detail()
    history.add_assistant_message(
        session_id,
        response,
        trace=build_trace(request, response, pipeline.filenames_for(response.retrieved_items)),
        unverified=flagged,
    )
    return session_id


@app.post("/query")
def query(
    request: QueryRequest,
    backend: str | None = None,
    session_id: str | None = None,
    account: dict = Depends(rate_limited("queries", "ask")),
) -> dict:
    """Text query against the index.

    `backend` ("offline" | "online") selects the inference provider and
    defaults to settings.inference_backend. It is a query parameter rather
    than a QueryRequest field so the locked schemas.py contract is unchanged,
    and `session_id` follows the same pattern: pass one and the turn is written
    to chat history with its trace, pass nothing and it is not recorded.

    The response is the QueryResponse contract plus two siblings:
    `technical_detail`, carrying provider error detail for a "Technical details"
    view, and `session_id`, the session this turn belongs to. Neither alters
    QueryResponse itself.

    Rate limited (429 with `Retry-After`) at `rate_limit_queries_per_minute`,
    because every call loads and runs local models.
    """
    from rag_pipeline import last_technical_detail

    response = get_pipeline().query(request, backend=backend)
    return {
        **response.model_dump(mode="json"),
        "technical_detail": last_technical_detail(),
        "session_id": _record_turn(request, response, backend, session_id),
    }


@app.post("/query_stream", response_model=None)
def query_stream(
    request: QueryRequest,
    backend: str | None = None,
    session_id: str | None = None,
    account: dict = Depends(rate_limited("queries", "ask")),
):
    """The same answer as `/query`, delivered as newline-delimited JSON as it is made.

    One `{"type": "token"}` line per generated token, then a single
    `{"type": "result"}` line carrying exactly the body `/query` would have
    returned - citations, retrieved items, stage trace, guards, technical
    detail, session id. A client that reads only the last line gets the `/query`
    contract unchanged, so this route adds time-to-first-token without adding a
    second idea of what an answer is.

    Three rules it does not get to break, all inherited from the pipeline's
    generator rather than reimplemented here:

      * Abstention still happens before generation. A refused question yields no
        token lines at all and arrives in about half a second - streaming must
        not turn a fast, decisive refusal into a slow one.
      * A stream cannot un-say a token. When a post-generation guard fires the
        correction is appended and `abstained` stays False; the result line
        carries the corrected text and the tokens already sent are history.
      * Nothing is buffered. The generator goes straight to StreamingResponse, so
        the first token leaves as soon as the model produces it.

    Rate limited exactly like `/query`, and the turn is recorded to history once,
    from the final QueryResponse rather than from the tokens - a stream that
    recorded as it went would leave a partial turn behind if the client
    disconnected halfway.
    """
    from rag_pipeline import last_technical_detail

    pipeline = get_pipeline()

    def lines():
        final = None
        for item in pipeline.query_stream(request, backend=backend):
            if isinstance(item, str):
                if item:
                    yield json.dumps({"type": "token", "content": item}) + "\n"
            else:
                final = item
        payload = {
            **(final.model_dump(mode="json") if final is not None else {}),
            "technical_detail": last_technical_detail(),
            "session_id": (_record_turn(request, final, backend, session_id)
                           if final is not None else None),
        }
        yield json.dumps({"type": "result", "content": payload}) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/models")
def models(account: dict = Depends(require("ask"))) -> dict:
    """What the inference server has installed, next to what this build configured.

    `/health` reports `degraded` when the configured model is not the one
    serving, and a failed query carries `technical_detail: "Model ... is not
    installed"` - but neither said what *was* available, which is the one thing
    the person reading it needs. That gap cost real time during the 2026-11-06
    fresh-install rehearsal: the answer was a refusal with an error code, and
    finding out that the models directory had been emptied took a separate call.

    Read-only and diagnostic on purpose. It does not select a model: the
    configuration of record is `.env`, and the run manifest exists so a result
    can be attributed to the build that produced it. A dropdown here would let a
    click change that build mid-demonstration.
    """
    configured = settings.ollama_model
    host = str(settings.ollama_host or "").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except Exception as exc:  # unreachable server, bad JSON, no such host
        return {
            "reachable": False,
            "host": host,
            "configured": configured,
            "configured_present": False,
            "installed": [],
            "names": [],
            "error": f"{type(exc).__name__}: {exc}",
            "hint": f"Ollama is not answering at {host}. Start it, or point "
                    "OLLAMA_HOST at the server that is running.",
        }
    installed = [
        {
            "name": entry.get("name"),
            "size_bytes": entry.get("size"),
            "modified_at": entry.get("modified_at"),
            "family": (entry.get("details") or {}).get("family"),
            "parameter_size": (entry.get("details") or {}).get("parameter_size"),
            "quantization": (entry.get("details") or {}).get("quantization_level"),
        }
        for entry in (payload.get("models") or [])
    ]
    names = [str(entry.get("name")) for entry in installed]
    present = configured in names
    return {
        "reachable": True,
        "host": host,
        "configured": configured,
        "configured_present": present,
        "installed": installed,
        "names": names,
        "error": None,
        "hint": None if present else (
            f"'{configured}' is not installed here. Either pull it "
            f"(ollama pull {configured}) or set OLLAMA_MODEL to one that is "
            f"present: {', '.join(names) if names else 'none'}."
        ),
    }


@app.post("/query_file", response_model=None)
async def query_file(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    query_modality: QueryModality = Form(QueryModality.IMAGE),
    inference_mode: InferenceMode = Form(InferenceMode.LOCAL),
    top_k: int = Form(settings.top_k),
    rerank_top_k: int = Form(settings.rerank_top_k),
    backend: str | None = Form(None),
    collection: str | None = Form(None),
    session_id: str | None = Form(None),
    account: dict = Depends(rate_limited("queries", "ask")),
) -> dict:
    """Query using an uploaded image, audio clip, or document as the query.

    `backend` ("offline" | "online") selects the inference provider, and
    `session_id` records the turn to chat history as it does for `/query`.

    The response model was opened up (it was `QueryResponse`) so this can return
    the same two sibling keys `/query` does. Every QueryResponse field is still
    present and unchanged; the additions are `technical_detail` and `session_id`.
    """
    suffix = Path(file.filename or "query").suffix or ".bin"
    data = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = handle.name

    from rag_pipeline import last_technical_detail

    try:
        request = QueryRequest(
            query=instruction or f"Find evidence related to this {query_modality.value}.",
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            inference_mode=inference_mode,
            query_modality=query_modality,
            query_file_path=temp_path,
            collection=collection,
        )
        response = get_pipeline().query(request, backend=backend)
        return {
            **response.model_dump(mode="json"),
            "technical_detail": last_technical_detail(),
            "session_id": _record_turn(request, response, backend, session_id),
        }
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


# ---------------------------------------------------------------------------
# Browser session: the React UI in webui/
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


# Serialises the first-run bootstrap. See POST /bootstrap for why one lock is
# enough here and where it stops being enough.
_BOOTSTRAP_LOCK = threading.Lock()


def _permissions_for(role) -> list[str]:
    from auth import PERMISSIONS, Role

    value = role.value if hasattr(role, "value") else str(role)
    try:
        return sorted(PERMISSIONS[Role(value)])
    except ValueError:
        return []


@app.post("/login")
def login(payload: LoginRequest) -> dict:
    """Exchange a username and password for a bearer token.

    The API is token-based and a CLI token is minted by
    `scripts/create_user.py --create-token`, but a browser cannot be handed one
    out of band, so the React UI signs in here. Lockout still applies -- five
    failures lock the account for five minutes -- and the token is labelled
    `web-ui`, so an administrator can tell a browser session from a script token
    in `list_tokens` and revoke one without the other.

    When `api_auth_enabled` is false there are no accounts to check, so the
    response carries no token and the UI works anonymously with full permissions,
    exactly as every other endpoint does in that mode.
    """
    from auth import get_user_store

    if not settings.api_auth_enabled:
        return {
            "token": None,
            "username": "auth-disabled",
            "role": "admin",
            "permissions": _permissions_for("admin"),
            "auth_enabled": False,
        }

    store = get_user_store()
    account = store.authenticate(payload.username, payload.password)
    if account is None:
        # authenticate() answers None both for a wrong password and for an
        # account locked by earlier failures. Those get different statuses: 401
        # keeps guessing uninformative, while 423 tells someone who mistyped five
        # times to stop rather than extend their own lock. The distinction costs
        # an attacker five logged attempts per username to learn that the
        # username exists, which is the trade the Streamlit login makes too.
        locked = store.locked_until(payload.username)
        if locked:
            raise HTTPException(
                status_code=423,
                detail={
                    "error": "This account is temporarily locked after repeated "
                             "failed sign-ins.",
                    "locked_until": locked,
                    "hint": f"Wait {settings.auth_lockout_minutes} minutes, or have "
                            "an administrator clear it by resetting the password: "
                            "python scripts/create_user.py <name> <new-password> "
                            "--set-password",
                },
            )
        raise HTTPException(
            status_code=401,
            detail={"error": "Incorrect username or password."},
        )

    role = account["role"]
    return {
        "token": store.create_token(account["username"], label="web-ui"),
        "username": account["username"],
        "role": role.value if hasattr(role, "value") else str(role),
        "permissions": _permissions_for(role),
        "auth_enabled": True,
    }


@app.get("/bootstrap")
def bootstrap_status() -> dict:
    """Whether the service is still waiting for its first administrator.

    A fresh install has no accounts, so nobody can sign in and nobody can be
    granted the `manage_users` permission that creates one. GET /bootstrap says
    whether that state still holds; POST /bootstrap acts on it and closes the
    moment any account exists.
    """
    from auth import get_user_store

    if not settings.api_auth_enabled:
        return {"needs_bootstrap": False, "auth_enabled": False, "accounts": 0}
    accounts = len(get_user_store().list_users())
    return {"needs_bootstrap": accounts == 0, "auth_enabled": True, "accounts": accounts}


@app.post("/bootstrap", status_code=201)
def bootstrap(payload: LoginRequest) -> dict:
    """Create the first administrator, without a credential, exactly once.

    This is the only unauthenticated endpoint that can create an account, and it
    is safe for the same reason the Streamlit first-run form is: it refuses as
    soon as any account exists, so it is a one-shot door that closes behind the
    first administrator. Afterwards accounts are created by someone holding the
    `manage_users` permission (`scripts/create_user.py`).

    The lock serialises the check and the insert within this process. Running
    several workers or several instances against one database would need the
    check inside a database transaction instead; for a first run on one machine,
    create the account headlessly if you would rather not rely on this at all.
    """
    from auth import AuthError, Role, get_user_store

    if not settings.api_auth_enabled:
        raise HTTPException(
            status_code=409,
            detail={"error": "API authentication is disabled, so there are no "
                             "accounts to create and none are needed."},
        )

    store = get_user_store()
    with _BOOTSTRAP_LOCK:
        if store.list_users():
            raise HTTPException(
                status_code=409,
                detail={"error": "An account already exists, so first-run "
                                 "bootstrap is closed.",
                        "hint": "Sign in, or add accounts with "
                                "scripts/create_user.py (needs manage_users)."},
            )
        try:
            created = store.create_user(
                payload.username.strip(), payload.password, Role.ADMIN
            )
        except AuthError as exc:
            # Password policy, or a name that is not acceptable.
            raise HTTPException(status_code=400, detail={"error": str(exc)})

    logger.warning(
        "First administrator '%s' created through /bootstrap; the endpoint is "
        "now closed.", created["username"],
    )
    return {
        "token": store.create_token(created["username"], label="web-ui"),
        "username": created["username"],
        "role": created["role"],
        "permissions": _permissions_for(created["role"]),
        "auth_enabled": True,
        "created": True,
    }


@app.get("/me")
def me(account: dict = Depends(require("ask"))) -> dict:
    """Who the presented token belongs to, and what that role may do.

    The UI reads this on load rather than trusting a role stored in the browser,
    so revoking a token or changing a role takes effect on the next request.
    """
    role = account["role"]
    return {
        "username": account["username"],
        "role": role.value if hasattr(role, "value") else str(role),
        "permissions": _permissions_for(role),
        "auth_enabled": settings.api_auth_enabled,
    }


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)) -> dict:
    """Revoke the presented token, so signing out destroys the credential rather
    than only forgetting it in the browser."""
    if not settings.api_auth_enabled or not authorization:
        return {"revoked": False}
    if not authorization.lower().startswith("bearer "):
        return {"revoked": False}
    from auth import TOKEN_VISIBLE_CHARS, get_user_store

    token = authorization.split(" ", 1)[1].strip()
    if len(token) < TOKEN_VISIBLE_CHARS:
        return {"revoked": False}
    try:
        get_user_store().revoke_token(token[:TOKEN_VISIBLE_CHARS])
        return {"revoked": True}
    except Exception as exc:
        logger.warning("Logout could not revoke the token: %s", exc)
        return {"revoked": False, "detail": str(exc)}


@app.get("/sources/stale")
def stale_sources(account: dict = Depends(require("ask"))) -> dict:
    """Filenames indexed more than once, with their source IDs newest last.

    Uploading a changed copy of a file adds a second version instead of replacing
    the first unless `replace=true` is passed to /ingest, so both can answer a
    question and the older one can be the one quoted. This is the surface that
    says which files are in that state; DELETE /source/{source_id} retires one.
    """
    return {
        "stale_versions": get_pipeline().stale_versions(),
        "hint": "Re-ingest with replace=true, or DELETE /source/{source_id} for "
                "the version you no longer want quoted.",
    }


# ---------------------------------------------------------------------------
# Chat history: one audit trail, reachable from both UIs
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    title: str | None = None


def _history():
    from chat_history import get_history

    return get_history()


def _session_or_404(session_id: str) -> dict:
    session = _history().get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail={"error": f"No session '{session_id}'."}
        )
    return session


# Permission policy mirrors the Streamlit history panel exactly: reading and
# deleting a session needs only `ask`, exporting one needs `export_history`
# (admin), because an export hands a whole conversation to someone else.
#
# Honest limitation, unchanged from the Streamlit UI: sessions carry no owner
# column, so "your own session" means "any session". A viewer can delete a
# session another role created. Per-account ownership is a schema change, not a
# permission check, and is not pretended at here.


@app.get("/history/sessions")
def list_history_sessions(
    limit: int = 50, account: dict = Depends(require("ask"))
) -> dict:
    """Recent sessions, newest first, plus counts for a sidebar."""
    history = _history()
    return {
        "sessions": history.list_sessions(max(1, min(int(limit or 50), 200))),
        "stats": history.stats(),
    }


@app.post("/history/sessions", status_code=201)
def create_history_session(
    payload: SessionRequest, account: dict = Depends(require("ask"))
) -> dict:
    """Start an empty session, so a browser can hold a conversation id before
    the first question is asked."""
    history = _history()
    title = (payload.title or "").strip()[:120] or "New session"
    session_id = history.create_session(title=title)
    return {
        "session_id": session_id,
        "session": history.get_session(session_id),
        "messages": [],
    }


@app.get("/history/sessions/{session_id}")
def read_history_session(
    session_id: str, account: dict = Depends(require("ask"))
) -> dict:
    """One session with every message, including the stored trace and citations.

    A message's `trace` holds the pipeline stages as they were recorded, so an
    answer can be re-inspected without re-running it -- including after the source
    it cited has been deleted, since filenames were resolved at record time.
    """
    session = _session_or_404(session_id)
    return {"session": session, "messages": _history().messages(session_id)}


@app.patch("/history/sessions/{session_id}")
def rename_history_session(
    session_id: str,
    payload: SessionRequest,
    account: dict = Depends(require("ask")),
) -> dict:
    """Rename a session. Titles default to the first question, which is not
    always the name a reviewer wants to find it under later."""
    _session_or_404(session_id)
    title = (payload.title or "").strip()[:120]
    if not title:
        raise HTTPException(
            status_code=400, detail={"error": "A title is required to rename a session."}
        )
    history = _history()
    history.rename_session(session_id, title)
    return {"session": history.get_session(session_id)}


@app.delete("/history/sessions/{session_id}")
def delete_history_session(
    session_id: str, account: dict = Depends(require("ask"))
) -> dict:
    """Delete a session and its messages. Indexed sources are untouched."""
    _session_or_404(session_id)
    history = _history()
    history.delete_session(session_id)
    return {"deleted": session_id, "stats": history.stats()}


@app.get("/history/sessions/{session_id}/export")
def export_history_session(
    session_id: str,
    format: str = "markdown",
    account: dict = Depends(require("export_history")),
) -> Response:
    """A whole session as Markdown or CSV, for a reviewer who will not sign in.

    Needs `export_history` (admin), the same permission the Streamlit download
    buttons check. The Markdown export carries each answer with its citations,
    abstention, unverified figures and the stage trace as recorded.
    """
    _session_or_404(session_id)
    history = _history()
    wanted = (format or "markdown").strip().lower()
    if wanted in ("markdown", "md"):
        text = history.export_markdown(session_id)
        media, extension = "text/markdown; charset=utf-8", "md"
    elif wanted == "csv":
        text = history.export_csv(session_id)
        media, extension = "text/csv; charset=utf-8", "csv"
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown export format '{format}'.",
                    "accepted": ["markdown", "csv"]},
        )
    return Response(
        content=text,
        media_type=media,
        headers={
            "Content-Disposition":
                f'attachment; filename="evidence_session_{session_id[:8]}.{extension}"'
        },
    )


# ---------------------------------------------------------------------------
# Collections: the grouping a query can be scoped to
# ---------------------------------------------------------------------------

class CollectionRequest(BaseModel):
    name: str


class CollectionAssignRequest(BaseModel):
    source_ids: list[str]


def _clean_collection_name(name: str) -> str:
    """Normalize a collection name, or 400 with the reason it was refused."""
    from vector_store import VectorStore

    try:
        return VectorStore.validate_collection_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc


# Collections are the one resource here where a DELETE does not delete anything.
# That is worth stating at the endpoint rather than leaving a caller to discover
# it: "delete collection" reads like "delete the documents in it", and this moves
# them to `general` instead. Nothing is re-embedded either -- a collection is a
# label on a source row, so renaming or removing one cannot damage the index.


@app.get("/collections")
def list_collections(account: dict = Depends(require("ask"))) -> dict:
    """Every collection with its source and indexed-item counts.

    Includes collections that were declared but are still empty, so one created a
    moment ago is still there before anything is filed under it.
    """
    return {
        "collections": get_pipeline().store.list_collections(),
        "default": DEFAULT_COLLECTION,
    }


@app.post("/collections", status_code=201)
def create_collection(
    payload: CollectionRequest,
    account: dict = Depends(require("manage_collections")),
) -> dict:
    """Declare a collection. It stays empty until sources are filed under it.

    409 if the name is taken, compared case-insensitively: `Board` beside `board`
    in one scope dropdown is a trap rather than a feature.
    """
    store = get_pipeline().store
    name = _clean_collection_name(payload.name)
    if store.collection_exists(name):
        raise HTTPException(
            status_code=409,
            detail={"error": f"A collection named '{name}' already exists.",
                    "hint": "Names are compared case-insensitively."},
        )
    store.create_collection(name)
    return {"collection": name, "sources": 0, "items": 0,
            "collections": store.list_collections()}


@app.patch("/collections/{name}")
def rename_collection(
    name: str,
    payload: CollectionRequest,
    account: dict = Depends(require("manage_collections")),
) -> dict:
    """Rename a collection, moving its sources with it.

    The sources keep their items, vectors and relationships. 400 for the default
    collection, which ingestion files into and deletion moves sources to, so it
    has to keep its name.
    """
    store = get_pipeline().store
    if name == DEFAULT_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail={"error": f"The '{DEFAULT_COLLECTION}' collection cannot be renamed.",
                    "hint": "Ingestion defaults to it and a deleted collection moves "
                            "its sources there."},
        )
    if not store.collection_exists(name):
        raise HTTPException(status_code=404,
                            detail={"error": f"No collection '{name}'."})
    new_name = _clean_collection_name(payload.name)
    if new_name.lower() != name.lower() and store.collection_exists(new_name):
        raise HTTPException(
            status_code=409,
            detail={"error": f"A collection named '{new_name}' already exists."},
        )
    moved = store.rename_collection(name, new_name)
    return {"collection": new_name, "renamed_from": name, "sources_moved": moved,
            "collections": store.list_collections()}


@app.delete("/collections/{name}")
def delete_collection(
    name: str, account: dict = Depends(require("manage_collections"))
) -> dict:
    """Ungroup a collection: its sources move to `general` and nothing is deleted.

    Sources, indexed items, vectors and the files on disk all survive. A query
    scoped to the deleted name simply stops matching anything.
    """
    store = get_pipeline().store
    if name == DEFAULT_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail={"error": f"The '{DEFAULT_COLLECTION}' collection cannot be deleted.",
                    "hint": "Deleting a collection moves its sources there, so it "
                            "cannot itself go."},
        )
    if not store.collection_exists(name):
        raise HTTPException(status_code=404,
                            detail={"error": f"No collection '{name}'."})
    moved = store.delete_collection(name)
    return {"deleted": name, "sources_moved": moved, "moved_to": DEFAULT_COLLECTION,
            "sources_deleted": 0, "collections": store.list_collections()}


@app.post("/collections/{name}/sources")
def assign_sources(
    name: str,
    payload: CollectionAssignRequest,
    account: dict = Depends(require("manage_collections")),
) -> dict:
    """File existing sources under a collection.

    The collection is declared if it did not exist, which is what ingestion
    already does when it is given a new name; the default collection is implied by
    its sources and is not declared. Unknown source ids are reported rather than
    failing the batch, so a browser holding a stale source list still moves
    everything that is really there.
    """
    store = get_pipeline().store
    collection = _clean_collection_name(name)
    missing = [sid for sid in payload.source_ids if store.get_source(sid) is None]
    moved = store.assign_sources(payload.source_ids, collection)
    return {
        "collection": collection,
        "requested": len(payload.source_ids),
        "moved": moved,
        "not_found": missing,
        "collections": store.list_collections(),
    }


# ---------------------------------------------------------------------------
# Static React UI
# ---------------------------------------------------------------------------

# The UI in webui/ builds to plain static files and is served by this process, so
# a deployment exposes one port and the browser talks to nothing else -- no CORS,
# no second server, and no localhost URL baked into the bundle. Registered last:
# FastAPI matches the explicit routes above before a mount.
_WEBUI_DIST = Path(__file__).resolve().parent / "webui" / "dist"

if (_WEBUI_DIST / "index.html").exists():
    app.mount(
        "/ui", StaticFiles(directory=str(_WEBUI_DIST), html=True), name="ui"
    )


@app.get("/", include_in_schema=False, response_model=None)
def root() -> dict | RedirectResponse:
    """Send a browser to the UI, or say how to build it."""
    if (_WEBUI_DIST / "index.html").exists():
        return RedirectResponse("/ui/")
    return {
        "service": "Evidence AI API",
        "ui": "not built",
        "build": "cd webui && npm install && npm run build",
        "docs": "/docs",
        "health": "/health",
    }
