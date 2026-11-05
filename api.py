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
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
