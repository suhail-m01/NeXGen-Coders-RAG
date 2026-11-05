"""
scripts/verify_offline.py -- prove the runtime really needs no internet.

This is a real test, not a claim. It installs a socket guard that raises on any
connection or DNS lookup to a non-loopback address, forces HuggingFace into
offline mode, and then runs actual work through the pipeline: an embedding, a
reranker score and a full question-answering query. Ollama on 127.0.0.1 is
deliberately allowed, because "offline" here means "no internet", not "no
local processes".

If anything reaches outward, the guard records it and the script fails, so a
passing run is evidence rather than optimism.

Run from the project root, after scripts\\prepare_models.py:
    .venv\\Scripts\\python scripts\\verify_offline.py
    .venv\\Scripts\\python scripts\\verify_offline.py --query "What is the project status?"
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
blocked_attempts: list[str] = []

_real_getaddrinfo = socket.getaddrinfo
_real_create_connection = socket.create_connection
_real_socket_connect = socket.socket.connect
_real_socket_connect_ex = socket.socket.connect_ex


def _is_loopback_ip(text: str) -> bool:
    try:
        return ipaddress.ip_address(text.split("%")[0]).is_loopback
    except ValueError:
        return False


def _host_is_local(host) -> bool:
    if host is None:
        return True
    text = str(host).strip("[]")
    return text in LOOPBACK_HOSTS or _is_loopback_ip(text)


def guarded_getaddrinfo(host, port, *args, **kwargs):
    if not _host_is_local(host):
        blocked_attempts.append(f"DNS lookup for {host}")
        raise OSError(f"[offline-verify] blocked DNS lookup for {host!r}")
    return _real_getaddrinfo(host, port, *args, **kwargs)


def guarded_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if not _host_is_local(host):
        blocked_attempts.append(f"connection to {host}")
        raise OSError(f"[offline-verify] blocked connection to {host!r}")
    return _real_create_connection(address, *args, **kwargs)


def guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if not _host_is_local(host):
        blocked_attempts.append(f"socket connect to {host}")
        raise OSError(f"[offline-verify] blocked socket connect to {host!r}")
    return _real_socket_connect(self, address)


def guarded_connect_ex(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if not _host_is_local(host):
        blocked_attempts.append(f"socket connect_ex to {host}")
        raise OSError(f"[offline-verify] blocked socket connect_ex to {host!r}")
    return _real_socket_connect_ex(self, address)


def install_guard() -> None:
    socket.getaddrinfo = guarded_getaddrinfo
    socket.create_connection = guarded_create_connection
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    # Force HuggingFace to use its local cache and never call home.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _say(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def _ok(text: str) -> None:
    print(f"  [OK]   {text}", flush=True)


def _bad(text: str) -> None:
    print(f"  [FAIL] {text}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query",
                        default="What is the current status of the project?",
                        help="question to answer while the network is blocked")
    parser.add_argument("--skip-query", action="store_true",
                        help="only test embeddings and reranking, skip the LLM")
    args = parser.parse_args()

    print("Installing the offline socket guard (loopback still allowed for Ollama).")
    install_guard()

    failures: list[str] = []

    _say("Embeddings, offline")
    try:
        from embeddings import embed_query

        start = time.time()
        vector = embed_query("offline verification")
        _ok(f"{vector.shape[0]}-d vector in {time.time() - start:.1f}s, "
            f"no network used")
    except Exception as exc:
        failures.append(f"embeddings: {exc}")
        _bad(f"{exc.__class__.__name__}: {exc}")
        if not blocked_attempts:
            _bad("Models are probably not cached yet. Run scripts\\prepare_models.py "
                 "while still connected.")

    _say("Reranking, offline")
    try:
        from reranker import get_model

        start = time.time()
        score = float(get_model().predict(
            [("what is the budget?", "the budget is 87 percent used")]
        )[0])
        _ok(f"cross-encoder logit {score:.3f} in {time.time() - start:.1f}s")
    except Exception as exc:
        failures.append(f"reranker: {exc}")
        _bad(f"{exc.__class__.__name__}: {exc}")

    if not args.skip_query:
        _say("Full question answering, offline")
        try:
            from rag_pipeline import get_pipeline
            from schemas import QueryRequest

            pipeline = get_pipeline()
            health = pipeline.health()
            index = health.get("index", {})
            print(f"  index: sources={index.get('sources')} items={index.get('items')}")
            if not index.get("items"):
                print("  [WARN] the index is empty - ingest files first "
                      "(scripts\\ingest_demo.py). The query will abstain.")

            start = time.time()
            response = pipeline.query(QueryRequest(query=args.query))
            elapsed = time.time() - start
            print(f"  Q: {args.query}")
            answer = (response.answer or "").strip().replace("\n", " ")
            print(f"  A: {answer[:260]}")
            print(f"  abstained={response.abstained} "
                  f"confidence={response.confidence} "
                  f"latency={elapsed:.1f}s")
            for citation in (response.citations or [])[:3]:
                location = citation.location
                renderer = getattr(location, "human", None)
                where = ""
                if callable(renderer):
                    try:
                        where = str(renderer()).strip()
                    except Exception:
                        where = ""
                if not where:
                    page = getattr(location, "page_number", None)
                    start = getattr(location, "timestamp_start", None)
                    end = getattr(location, "timestamp_end", None)
                    if page is not None:
                        where = f"p.{page}"
                    elif start is not None:
                        where = f"{start:.1f}-{end:.1f}s" if end is not None else f"{start:.1f}s"
                print(f"  cite: {citation.filename} {where}")
            _ok("answered using only local models and the local Ollama server")
        except Exception as exc:
            failures.append(f"query: {exc}")
            _bad(f"{exc.__class__.__name__}: {exc}")

    _say("Outbound access attempted during this run")
    if blocked_attempts:
        for attempt in dict.fromkeys(blocked_attempts):
            _bad(attempt)
        failures.append(f"{len(blocked_attempts)} blocked outbound attempt(s)")
    else:
        _ok("none - every socket stayed on loopback")

    if failures:
        print(f"\nOffline verification FAILED ({len(failures)} problem(s)).")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nOffline verification PASSED. The runtime works with no internet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
