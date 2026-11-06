#!/usr/bin/env python3
"""End-to-end process flow: one scripted user walking the whole application.

Unlike apitest.py, which probes endpoints in isolation, this script follows a
single session through the journey a demonstrator actually takes - sign in,
look at the overview feeds, ask a streamed question, get a refusal, read the
conversation back, export it, regroup sources into a collection, upload a new
document, find it, delete it again, sign out - and asserts the observable
result of every step, including which authentication transport carried it.

Authentication is exercised on all four transports the server accepts
(Authorization header, X-EAI-Token header, session cookie, access_token query
parameter) because restrictive proxies and sandboxed embed viewers are known
to drop the first one, and the UI silently falls through to the others.

Usage:
    python scripts/e2e_flow.py [--base http://127.0.0.1:8000]
                               [--username tester --password Tester!234]
                               [--json /path/to/report.json]

Exit code 0 when every step passed, 1 otherwise. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PASS, FAIL = "PASS", "FAIL"


class Runner:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.results: list[dict] = []
        self.token: str | None = None
        self.cookie: str | None = None

    # ---- transport ------------------------------------------------------
    def call(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        raw: bytes | None = None,
        content_type: str | None = None,
        transport: str = "header",
        headers: dict | None = None,
        stream: bool = False,
    ):
        """Return (status, parsed-or-text payload, response headers)."""
        url = self.base + path
        hdrs = dict(headers or {})
        if transport == "header" and self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        elif transport == "x-eai-token" and self.token:
            hdrs["X-EAI-Token"] = self.token
        elif transport == "cookie" and self.token:
            hdrs["Cookie"] = f"eai_session={self.token}"
        elif transport == "query" and self.token:
            url += ("&" if "?" in url else "?") + "access_token=" + self.token
        if body is not None:
            raw = json.dumps(body).encode()
            content_type = "application/json"
        if content_type:
            hdrs["Content-Type"] = content_type
        request = urllib.request.Request(url, data=raw, headers=hdrs, method=method)
        try:
            # A streamed response must outlive this frame: closing it here
            # (as a `with` block would) leaves the caller an empty body.
            response = urllib.request.urlopen(request, timeout=300)
            if stream:
                return response.status, response, dict(response.headers)
            with response:
                text = response.read().decode("utf-8", "replace")
            return response.status, _maybe_json(text), dict(response.headers)
        except urllib.error.HTTPError as err:
            text = err.read().decode("utf-8", "replace")
            return err.code, _maybe_json(text), dict(err.headers)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"step": name, "result": PASS if ok else FAIL, "detail": detail})
        print(f"{PASS if ok else FAIL}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
        return ok


def _maybe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return text


def multipart(fields: dict, files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----e2e" + uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
    for key, filename, data in files:
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n"
        ).encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="tester")
    parser.add_argument("--password", default="Tester!234")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    run = Runner(args.base)
    started = time.time()
    codeword = f"zebra-lantern-{uuid.uuid4().hex[:6]}"

    # 1 - first-run door ---------------------------------------------------
    status, payload, _ = run.call("/bootstrap")
    run.check("GET /bootstrap answers and reports accounts", status == 200 and isinstance(payload, dict)
              and payload.get("needs_bootstrap") is False, f"{status} {payload}")

    # 2 - sign-in ----------------------------------------------------------
    status, payload, _ = run.call("/login", "POST", {"username": args.username, "password": "wrong-pass"})
    run.check("wrong password is refused with 401", status == 401, f"{status}")
    status, payload, headers = run.call("/login", "POST", {"username": args.username, "password": args.password})
    token = payload.get("token") if isinstance(payload, dict) else None
    set_cookies = [v for k, v in headers.items() if k.lower() == "set-cookie" and "eai_session" in v]
    run.check("login mints a token", status == 200 and bool(token), f"{status} {payload}")
    run.check("login also sets the HttpOnly session cookie", bool(set_cookies) and "httponly" in set_cookies[0].lower(),
              f"{set_cookies}")
    run.token = token

    # 3 - every auth transport ---------------------------------------------
    for transport in ("header", "x-eai-token", "cookie", "query"):
        status, payload, _ = run.call("/me", transport=transport)
        run.check(f"/me authenticates via the {transport} transport",
                  status == 200 and isinstance(payload, dict) and payload.get("username") == args.username,
                  f"{status} {payload}")
    status, payload, _ = run.call("/me", transport="none")
    run.check("no credential at all is still 401", status == 401, f"{status}")

    # 4 - the feeds the Overview tab reads ----------------------------------
    status, health, _ = run.call("/health")
    run.check("health feed", status == 200 and isinstance(health, dict) and "index" in health, f"{status}")
    status, models, _ = run.call("/models")
    run.check("language-model diagnostics: reachable and configured model present",
              status == 200 and models.get("reachable") is True and models.get("configured_present") is True,
              f"{status} {models}")
    status, collections, _ = run.call("/collections")
    run.check("collections feed", status == 200 and isinstance(collections, dict), f"{status}")
    status, sessions, _ = run.call("/history/sessions?limit=50")
    run.check("history feed", status == 200 and isinstance(sessions, dict), f"{status}")

    # 5 - streamed ask: refusal first, then an answered question -------------
    status, payload, _ = run.call("/history/sessions", "POST", {"title": "e2e flow"})
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    run.check("conversation session created", status in (200, 201) and bool(session_id), f"{status} {payload}")

    status, response, _ = run.call(f"/query_stream?session_id={session_id}", "POST",
                                   {"query": "What was the budget in 2040?", "query_modality": "text"},
                                   stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    run.check("refusal streams zero tokens and abstains before generation",
              status == 200 and tokens == [] and isinstance(result, dict)
              and result.get("abstained") is True and result.get("error_code") == "INSUFFICIENT_EVIDENCE",
              f"{status} tokens={len(tokens)} result={type(result).__name__}")

    status, response, _ = run.call(f"/query_stream?session_id={session_id}", "POST",
                                   {"query": "What percentage of the budget was utilised?",
                                    "query_modality": "text"},
                                   stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    streamed = "".join(tokens)
    ok_stream = (
        status == 200 and isinstance(result, dict) and tokens
        and result.get("answer", "").startswith(streamed)
        and len(result.get("citations") or []) >= 1
        and len(result.get("stages") or []) >= 15
    )
    run.check("answered question streams tokens that are a prefix of the final answer, with citations",
              ok_stream, f"{status} tokens={len(tokens)} citations={len((result or {}).get('citations') or [])}")
    run.check("the answered turn reports 87 %", "87" in (result or {}).get("answer", ""),
              (result or {}).get("answer", "")[:80])

    # 6 - history: the conversation comes back, and exports ------------------
    status, payload, _ = run.call(f"/history/sessions/{session_id}")
    messages = (payload or {}).get("messages") if isinstance(payload, dict) else None
    roles = [m.get("role") for m in messages] if isinstance(messages, list) else None
    # A turn is a pair: the question as asked and the answer as given.
    run.check("session reads back both turns as question/answer pairs",
              status == 200 and roles == ["user", "assistant", "user", "assistant"],
              f"{status} roles={roles}")
    for fmt in ("md", "csv"):
        status, payload, _ = run.call(f"/history/sessions/{session_id}/export?format={fmt}")
        run.check(f"session exports as {fmt}", status == 200 and isinstance(payload, str) and len(payload) > 50,
                  f"{status}")

    # 7 - sources ------------------------------------------------------------
    status, payload, _ = run.call("/sources")
    sources = (payload or {}).get("sources") if isinstance(payload, dict) else None
    before = len(sources or [])
    run.check("source list reads", status == 200 and before >= 6, f"{status} count={before}")
    first_id = (sources or [{}])[0].get("source_id")
    status, payload, _ = run.call(f"/source/{first_id}")
    run.check("source detail reads", status == 200 and isinstance(payload, dict), f"{status}")

    # 8 - collection lifecycle ------------------------------------------------
    name = f"e2e-{uuid.uuid4().hex[:6]}"
    status, payload, _ = run.call("/collections", "POST", {"name": name})
    run.check("collection created", status in (200, 201), f"{status} {payload}")
    status, payload, _ = run.call(f"/collections/{name}/sources", "POST", {"source_ids": [first_id]})
    status2, sources2, _ = run.call("/sources")
    moved = [s for s in (sources2 or {}).get("sources", []) if s.get("source_id") == first_id]
    run.check("source moved into the collection",
              status == 200 and moved and moved[0].get("collection") == name, f"{status} {moved}")
    renamed = name + "-b"
    status, payload, _ = run.call(f"/collections/{name}", "PATCH", {"name": renamed})
    run.check("collection renamed", status == 200, f"{status} {payload}")
    status, payload, _ = run.call(f"/collections/{renamed}", "DELETE")
    status2, sources2, _ = run.call("/sources")
    back = [s for s in (sources2 or {}).get("sources", []) if s.get("source_id") == first_id]
    run.check("deleting the collection returns the source to general without deleting it",
              status == 200 and back and back[0].get("collection") == "general", f"{status} {back}")

    # 9 - ingest lifecycle: upload, find, delete ------------------------------
    note = f"The maintenance codeword for the night shift is {codeword}.\n".encode()
    body, ctype = multipart({"collection": "general", "replace": "false"},
                            [("files", "e2e_note.txt", note)])
    status, payload, _ = run.call("/ingest", "POST", raw=body, content_type=ctype)
    run.check("upload ingests", status == 200 and isinstance(payload, dict), f"{status} {payload}")
    status, payload, _ = run.call("/sources")
    new = [s for s in (payload or {}).get("sources", []) if s.get("filename") == "e2e_note.txt"]
    run.check("uploaded document appears in the source list", status == 200 and len(new) == 1,
              f"{status} matches={len(new)}")
    status, result, _ = run.call("/query", "POST",
                                 {"query": "What is the maintenance codeword for the night shift?",
                                  "query_modality": "text"})
    answer = (result or {}).get("answer", "")
    cited = [c.get("filename") for c in (result or {}).get("citations") or []]
    run.check("the uploaded document is retrievable and cited",
              status == 200 and codeword in answer and "e2e_note.txt" in cited,
              f"{status} answer={answer[:60]} cited={cited}")
    if new:
        status, payload, _ = run.call(f"/source/{new[0]['source_id']}", "DELETE")
        status2, payload2, _ = run.call("/sources")
        after = (payload2 or {}).get("count")
        run.check("deleting the source rebuilds the index to its previous size",
                  status == 200 and after == before, f"{status} after={after} before={before}")

    # 9b - attachments: a question asked WITH a file, ChatGPT-style ------------
    # The attachment is ephemeral evidence: extracted with the ingestion
    # processors, reranked against the question, guarded and cited like any
    # other evidence, and never indexed. These steps prove each of those
    # claims against the running service.
    note = b"Night handover.\nThe handover codeword for the night shift is tangerine-wharf-9.\n"
    body, ctype = multipart(
        {"query": "What is the handover codeword for the night shift?"},
        [("files", "handover_note.txt", note)])
    status, response, _ = run.call("/query_attachment", "POST", raw=body,
                                   content_type=ctype, stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    cited = [c.get("filename") for c in (result or {}).get("citations") or []]
    run.check("a text attachment is answered from and cited by its upload name",
              status == 200 and tokens and "tangerine-wharf-9" in (result or {}).get("answer", "")
              and "handover_note.txt" in cited
              and (result or {}).get("answer", "").startswith("".join(tokens)),
              f"{status} cited={cited}")

    fixture = Path(__file__).resolve().parent.parent / "evaluation" / "fixtures" / "attach_vault_note.png"
    body, ctype = multipart({"query": "What is the vault codeword?"},
                            [("files", "vault_note.png", fixture.read_bytes())])
    status, response, _ = run.call("/query_attachment", "POST", raw=body,
                                   content_type=ctype, stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    cited = [c.get("filename") for c in (result or {}).get("citations") or []]
    run.check("an image attachment is OCR-read, answered from and cited",
              status == 200 and "indigo harbor melon" in (result or {}).get("answer", "")
              and "vault_note.png" in cited,
              f"{status} cited={cited} answer={(result or {}).get('answer', '')[:60]}")

    body, ctype = multipart({"query": "What was the handover budget in 2040?"},
                            [("files", "handover_note.txt", note)])
    status, response, _ = run.call("/query_attachment", "POST", raw=body,
                                   content_type=ctype, stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    run.check("the coverage guard refuses a number the attachment does not contain",
              status == 200 and tokens == [] and (result or {}).get("abstained") is True,
              f"{status} tokens={len(tokens)}")

    # "Explain this document" is the request the paperclip exists for. A
    # cross-encoder relevance score is meaningless for it (measured: the old
    # score gate refused every such request at confidence 0.001), so an
    # attachment turn must answer from the attachment - and cite nothing else,
    # because weak corpus pages in the context get woven into the summary.
    body, ctype = multipart({"query": "explain this document"},
                            [("files", "handover_note.txt", note)])
    status, response, _ = run.call("/query_attachment", "POST", raw=body,
                                   content_type=ctype, stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    cited = sorted({c.get("filename") for c in (result or {}).get("citations") or []})
    run.check("'explain this document' is answered from the attachment alone",
              status == 200 and tokens and (result or {}).get("abstained") is False
              and cited == ["handover_note.txt"],
              f"{status} cited={cited} abstained={(result or {}).get('abstained')}")

    # The mirror image: with an attachment present, a question the CORPUS
    # answers must still resolve to the corpus, because relevance - not the
    # paperclip - orders the evidence (measured: attachment-first ordering
    # answered 50 % from an attached scheme of examination instead of 87 %).
    body, ctype = multipart({"query": "What percentage of the budget was utilised?"},
                            [("files", "vault_note.png", fixture.read_bytes())])
    status, response, _ = run.call("/query_attachment", "POST", raw=body,
                                   content_type=ctype, stream=True)
    tokens, result = [], None
    if status == 200:
        for line in response:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "result":
                result = event.get("content")
    response.close()
    cited = [c.get("filename") for c in (result or {}).get("citations") or []]
    run.check("a corpus question asked with an attachment still resolves to the corpus",
              status == 200 and "87" in (result or {}).get("answer", "")
              and "project_notes.docx" in cited,
              f"{status} answer={(result or {}).get('answer', '')[:40]} cited={cited}")

    # A well-formed upload with no credential: authentication must refuse it
    # before any extraction work happens. (A *malformed* multipart body is a
    # 400 from the form parser before the auth dependency runs - that order is
    # FastAPI's, and it leaks nothing, so it is not what this step asserts.)
    body, ctype = multipart({"query": "hi"}, [("files", "note.txt", b"x")])
    status, payload, _ = run.call("/query_attachment", "POST", raw=body,
                                  content_type=ctype, transport="none")
    run.check("an attachment without a credential is 401", status == 401, f"{status}")
    status, payload, _ = run.call("/query_attachment", "POST",
                                  raw=multipart({"query": "hi"}, [])[0],
                                  content_type=multipart({}, [])[1])
    run.check("an attachment question with no file is 400", status == 400, f"{status}")
    body, ctype = multipart({"query": "hi"}, [("files", "payload.exe", b"MZ")])
    status, payload, _ = run.call("/query_attachment", "POST", raw=body, content_type=ctype)
    run.check("an unsupported attachment is 400 naming the file",
              status == 400 and "payload.exe" in str(payload), f"{status} {payload}")
    status, payload, _ = run.call("/sources")
    run.check("attachments never join the index",
              status == 200 and (payload or {}).get("count") == before,
              f"{status} count={(payload or {}).get('count')} before={before}")

    # 10 - static UI ------------------------------------------------------------
    status, payload, headers = run.call("/ui/", transport="none")
    run.check("the React UI is served at /ui/", status == 200 and "<div" in str(payload).lower(), f"{status}")
    # urllib follows redirects, which would turn the 307 into a 200 and hide a
    # broken hop: read the status line off the socket instead.
    import http.client
    from urllib.parse import urlsplit
    parts = urlsplit(run.base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=30)
    conn.request("GET", "/", headers={"Accept": "text/html"})
    raw_status = conn.getresponse().status
    conn.close()
    run.check("root redirects into the UI without following it", raw_status in (307, 308), f"{raw_status}")

    # 11 - sign out destroys the credential -------------------------------------
    status, payload, headers = run.call("/logout", "POST", {})
    cleared = [v for k, v in headers.items() if k.lower() == "set-cookie" and "eai_session" in v]
    run.check("logout clears the session cookie", bool(cleared), f"{cleared}")
    status, payload, _ = run.call("/me")
    run.check("the revoked token no longer authenticates", status == 401, f"{status}")

    elapsed = round(time.time() - started, 1)
    failed = [r for r in run.results if r["result"] == FAIL]
    print(f"\n{len(run.results) - len(failed)}/{len(run.results)} steps passed in {elapsed}s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"base": args.base, "elapsed_s": elapsed,
                       "passed": len(run.results) - len(failed),
                       "total": len(run.results), "steps": run.results}, handle, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
