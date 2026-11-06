"""Test the browser-facing API surface without starting a server.

    python scripts/apitest.py                # 51 checks, no LLM, no network
    python scripts/apitest.py --with-query   # +5 more: the live /query contract (needs Ollama)
    python scripts/apitest.py -v             # print every check

This covers the part of the API that a browser actually uses, which is the part
`scripts/selftest.py` cannot reach: the static React bundle and the relative asset
paths under `/ui`, the first-run bootstrap, the session endpoints, the role and
lockout behaviour, chat history over HTTP including its export permissions, the
429 path of the rate limiter, the stale-version endpoint, and the manifest. It runs through FastAPI's TestClient, so
nothing binds a port and no Ollama call is made unless `--with-query` is passed.

One thing to know before reading the code: `config.auth_db` is derived from
`DATA_DIR`, so there is no way to point this at a throwaway account database while
still testing against the real index. It therefore creates one account, and removes
it in a `finally` block, and asserts at the end that the database holds exactly as
many accounts and tokens as it did at the start. If that assertion ever fails, this
script left credentials behind and should be treated as broken. The same is
asserted of the chat-history database, since the history checks create sessions
and are supposed to delete them.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []
CHECKS = 0
VERBOSE = False


def check(label: str, ok: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    print(("PASS  " if ok else "FAIL  ") + label + (f"  | {detail}" if detail != "" and VERBOSE else ""))
    if not ok:
        FAILURES.append(label)
        if detail != "" and not VERBOSE:
            print(f"        {str(detail)[:200]}")


def db_counts(auth_db: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(auth_db))
    try:
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        tokens = conn.execute("SELECT COUNT(*) FROM api_tokens").fetchone()[0]
    finally:
        conn.close()
    return users, tokens


def index_state(index_dir) -> dict:
    """What the built index contains, opened read-only.

    This suite moves a real source into a temporary collection and moves it back again,
    which is a write to the corpus database. The content has to come back exactly as it
    was found, and until this check existed nothing said so: apitest asserted that the
    account and history databases were left alone and stayed silent about the one
    database a demonstration actually depends on. Read-only mode matters -- a check that
    writes to the thing it is verifying cannot catch its own damage.
    Compared across processes this has to survive JSON, where a tuple becomes a
    list, so every sequence it returns is a list: a snapshot dumped before a demo
    and read back afterwards must not report a change that is only a container
    type. That is not a hypothetical - it produced exactly that false positive
    while this check was being written.
    """
    import sqlite3
    from pathlib import Path

    db = Path(index_dir) / "store.db"
    if not db.exists():
        return {"present": False}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        def count(table: str):
            try:
                return con.execute(f"select count(*) from {table}").fetchone()[0]
            except sqlite3.Error:
                return None

        return {
            "present": True,
            "sources": count("sources"),
            "items": count("items"),
            "relationships": count("relationships"),
            "parents": count("parents"),
            "collections": count("collections"),
            "assignments": [list(row) for row in con.execute(
                "select source_id, filename, collection from sources order by source_id"
            ).fetchall()],
        }
    finally:
        con.close()


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-query", action="store_true",
                        help="also run one live query (needs Ollama and a built index)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.verbose

    from fastapi.testclient import TestClient  # noqa: F401  (httpx-backed)

    import api
    from auth import Role, get_user_store
    from config import settings

    client = TestClient(api.app)
    store = get_user_store()
    before = db_counts(settings.auth_db)
    index_before = index_state(settings.index_dir)
    from chat_history import get_history

    history = get_history()
    history_before = history.stats()
    made_sessions: list[str] = []
    username, password = "apitest", "Ap1-test-pass!"
    first_admin = "apitest-first"
    token: str | None = None

    # ---- static React UI ---------------------------------------------------
    response = client.get("/", follow_redirects=False)
    check("GET / redirects to /ui/", response.status_code in (302, 307, 308)
          and "/ui/" in response.headers.get("location", ""), response.headers.get("location"))

    page = client.get("/ui/")
    html = page.text
    check("GET /ui/ serves the built app", page.status_code == 200 and '<div id="root">' in html,
          f"{page.status_code}, {len(html)} bytes")
    assets = [t for t in html.split('"') if t.startswith("./assets/") and t.endswith(".js")]
    check("index.html references a hashed bundle", bool(assets), assets[:1])
    if assets:
        bundle = client.get("/ui/" + assets[0][2:])
        check("bundle serves under /ui/ (relative base works)",
              bundle.status_code == 200 and len(bundle.text) > 10000, len(bundle.text))
    styles = [t for t in html.split('"') if t.endswith(".css")]
    if styles:
        sheet = client.get("/ui/" + styles[0][2:])
        check("stylesheet serves", sheet.status_code == 200 and "--accent" in sheet.text,
              sheet.status_code)
    check("unknown /ui path is a 404, not a crash",
          client.get("/ui/no/such/route").status_code == 404)

    # ---- authentication ----------------------------------------------------
    check("GET /me without a token is 401", client.get("/me").status_code == 401)
    check("GET /me with an unknown token is 401",
          client.get("/me", headers={"Authorization": "Bearer eai_deadbeefdeadbeef"}).status_code == 401)
    bad = client.post("/login", json={"username": "nobody", "password": "wrong"})
    check("POST /login with unknown credentials is 401", bad.status_code == 401, bad.json())
    check("GET /health needs no token", client.get("/health").status_code == 200)
    check("GET /bootstrap needs no token", client.get("/bootstrap").status_code == 200)

    try:
        # ---- first-run bootstrap ----------------------------------------
        # Only meaningful on an empty account database, which is how a fresh
        # install starts. On a machine that already has accounts the door is
        # correctly closed, and that is what gets asserted instead.
        status = client.get("/bootstrap").json()
        if before == (0, 0):
            check("a fresh install reports that it needs a first administrator",
                  status.get("needs_bootstrap") is True, status)
            weak = client.post("/bootstrap", json={"username": first_admin, "password": "short"})
            check("bootstrap enforces the password policy (400, door still open)",
                  weak.status_code == 400, (weak.json() or {}).get("detail"))
            created = client.post("/bootstrap", json={"username": first_admin, "password": password})
            body = created.json() if created.status_code == 201 else {}
            check("POST /bootstrap creates the first administrator and signs them in",
                  created.status_code == 201 and body.get("role") == "admin"
                  and str(body.get("token", "")).startswith("eai_"), body.get("role"))
            check("the bootstrap door is closed once an account exists",
                  client.get("/bootstrap").json().get("needs_bootstrap") is False
                  and client.post("/bootstrap",
                                  json={"username": "second", "password": password}).status_code == 409)
        else:
            check("an install with accounts keeps the bootstrap door closed",
                  status.get("needs_bootstrap") is False
                  and client.post("/bootstrap",
                                  json={"username": first_admin, "password": password}).status_code == 409,
                  status)


        store.create_user(username, password, Role.ADMIN)
        login = client.post("/login", json={"username": username, "password": password})
        body = login.json()
        token = body.get("token")
        check("POST /login returns a bearer token",
              login.status_code == 200 and isinstance(token, str) and token.startswith("eai_"),
              login.status_code)
        check("login reports the role and its permissions",
              body.get("role") == "admin" and "delete" in (body.get("permissions") or []),
              body.get("permissions"))
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/me", headers=headers)
        check("GET /me resolves the token to the account",
              me.status_code == 200 and me.json().get("username") == username, me.json())

        sources = client.get("/sources", headers=headers)
        check("GET /sources answers with the corpus and its collections",
              sources.status_code == 200 and isinstance(sources.json().get("sources"), list),
              sources.json().get("count"))
        stale = client.get("/sources/stale", headers=headers)
        check("GET /sources/stale answers",
              stale.status_code == 200 and isinstance(stale.json().get("stale_versions"), dict),
              list((stale.json().get("stale_versions") or {}).keys()))

        manifest = client.get("/manifest", headers=headers)
        payload = manifest.json() if manifest.status_code == 200 else {}
        check("GET /manifest returns the build manifest",
              manifest.status_code == 200 and {"application", "guards", "models"} <= set(payload),
              sorted(payload))
        # The React drop zone takes its accepted formats from here, so the payload
        # has to carry the router's own tables rather than a copy of them.
        ingestion = payload.get("ingestion") or {}
        accepted = {ext for group in ("documents", "images", "audio")
                    for ext in ingestion.get(group) or []}
        check("GET /manifest carries the accepted ingestion formats",
              {".pdf", ".docx", ".csv", ".png", ".wav"} <= accepted
              and ingestion.get("max_file_bytes", 0) > 0,
              sorted(accepted))

        health = client.get("/health").json()
        runtime = health.get("runtime") or {}
        check("GET /health carries the runtime block and the index",
              bool(runtime.get("git_commit")) and "index" in health and "guards" in runtime,
              {"commit": runtime.get("git_commit"), "status": health.get("status")})

        # ---- chat history over HTTP -------------------------------------
        listed = client.get("/history/sessions", headers=headers)
        check("GET /history/sessions lists sessions with counts",
              listed.status_code == 200 and isinstance(listed.json().get("sessions"), list)
              and "stats" in listed.json(), listed.json().get("stats"))

        created = client.post("/history/sessions", headers=headers,
                              json={"title": "apitest session"})
        new_id = (created.json() or {}).get("session_id")
        if new_id:
            made_sessions.append(new_id)
        check("POST /history/sessions creates an empty session (201)",
              created.status_code == 201 and bool(new_id), created.status_code)

        read = client.get(f"/history/sessions/{new_id}", headers=headers)
        payload = read.json() if read.status_code == 200 else {}
        check("GET /history/sessions/{id} returns the session and its messages",
              read.status_code == 200 and payload.get("messages") == []
              and (payload.get("session") or {}).get("title") == "apitest session",
              (payload.get("session") or {}).get("title"))

        renamed = client.patch(f"/history/sessions/{new_id}", headers=headers,
                               json={"title": "renamed by apitest"})
        check("PATCH renames a session",
              renamed.status_code == 200
              and (renamed.json().get("session") or {}).get("title") == "renamed by apitest")
        blank = client.patch(f"/history/sessions/{new_id}", headers=headers, json={"title": "   "})
        check("PATCH refuses a blank title (400)", blank.status_code == 400, blank.status_code)
        check("an unknown session is a 404, not a 500",
              client.get("/history/sessions/does-not-exist", headers=headers).status_code == 404)

        # ---- rate limiting on the model-running routes --------------------
        # Buckets are drained through the limiter itself so the HTTP call that
        # follows is refused before any model is loaded. That is what keeps this
        # section in the offline suite: a 429 has to be provable without spending
        # fifteen seconds of generation on it.
        from rate_limit import get_limiter

        limiter = get_limiter()
        runtime = (client.get("/health").json() or {}).get("runtime") or {}
        snapshot = runtime.get("rate_limit") or {}
        check("GET /health reports the rate limit in force",
              snapshot.get("enabled") is True
              and set(snapshot.get("buckets") or {}) == {"queries", "ingest"}
              and "process" in (snapshot.get("scope") or ""), snapshot.get("scope"))

        saved_limits = (
            settings.rate_limit_queries_per_minute, settings.rate_limit_queries_burst,
            settings.rate_limit_ingest_per_minute, settings.rate_limit_ingest_burst,
        )
        try:
            settings.rate_limit_queries_per_minute = 6.0
            settings.rate_limit_queries_burst = 1
            settings.rate_limit_ingest_per_minute = 6.0
            settings.rate_limit_ingest_burst = 1
            limiter.reset()
            limiter.check("queries", f"account:{username}")   # spend the single token
            throttled = client.post("/query", headers=headers,
                                    json={"query": "throttled", "query_modality": "text"})
            detail = (throttled.json() or {}).get("detail") or {} if throttled.status_code == 429 else {}
            check("a client past its burst gets 429 before any model runs",
                  throttled.status_code == 429 and detail.get("bucket") == "queries"
                  and detail.get("retry_after_seconds", 0) > 0, throttled.status_code)
            check("the refusal carries a Retry-After header a client can obey",
                  (throttled.headers.get("retry-after") or "").isdigit(),
                  throttled.headers.get("retry-after"))
            check("the refusal says what to do instead",
                  "reindex" in (detail.get("hint") or ""), (detail.get("hint") or "")[:60])
            check("authentication is checked before the limit (401, not 429)",
                  client.post("/query", json={"query": "x", "query_modality": "text"}
                              ).status_code == 401)
            limiter.check("ingest", f"account:{username}")
            upload = client.post(
                "/ingest", headers=headers,
                files=[("files", ("throttled.txt", b"hello", "text/plain"))])
            check("/ingest is limited as well, on its own bucket",
                  upload.status_code == 429
                  and ((upload.json() or {}).get("detail") or {}).get("bucket") == "ingest",
                  upload.status_code)
            check("an unlimited route still answers while the buckets are empty",
                  client.get("/history/sessions", headers=headers).status_code == 200)
        finally:
            (settings.rate_limit_queries_per_minute, settings.rate_limit_queries_burst,
             settings.rate_limit_ingest_per_minute, settings.rate_limit_ingest_burst) = saved_limits
            limiter.reset()
        check("limits are read per request, so restoring them takes effect at once",
              limiter.check("queries", f"account:{username}")[0] is True)

        # A viewer may ask but not delete; the distinction has to survive the API.
        store.create_user("apitest-viewer", password, Role.VIEWER)
        viewer_login = client.post("/login", json={"username": "apitest-viewer", "password": password})
        viewer_headers = {"Authorization": f"Bearer {viewer_login.json().get('token')}"}
        check("a viewer may read the manifest",
              client.get("/manifest", headers=viewer_headers).status_code == 200)
        check("a viewer may not delete a source (403, not 401)",
              client.delete("/source/does-not-exist", headers=viewer_headers).status_code == 403)
        check("a viewer may read history",
              client.get("/history/sessions", headers=viewer_headers).status_code == 200)
        check("a viewer may not export a session (403)",
              client.get(f"/history/sessions/{new_id}/export",
                         headers=viewer_headers).status_code == 403)

        exported = client.get(f"/history/sessions/{new_id}/export?format=markdown",
                              headers=headers)
        check("an admin may export a session as Markdown",
              exported.status_code == 200 and "attachment" in
              exported.headers.get("content-disposition", ""), exported.status_code)
        as_csv = client.get(f"/history/sessions/{new_id}/export?format=csv", headers=headers)
        check("and as CSV", as_csv.status_code == 200 and as_csv.headers.get(
            "content-type", "").startswith("text/csv"), as_csv.headers.get("content-type"))
        bad_format = client.get(f"/history/sessions/{new_id}/export?format=pdf", headers=headers)
        check("an unknown export format is 400 with the accepted list",
              bad_format.status_code == 400 and "csv" in str(bad_format.json()), bad_format.json())

        deleted_session = client.delete(f"/history/sessions/{new_id}", headers=headers)
        check("DELETE removes a session",
              deleted_session.status_code == 200
              and client.get(f"/history/sessions/{new_id}", headers=headers).status_code == 404)
        if new_id in made_sessions:
            made_sessions.remove(new_id)

        # ---- model diagnostics -------------------------------------------
        # Read-only on purpose: the configuration of record is .env, and a control
        # that could change the model would break the manifest's attribution of a
        # result to a build. What this route has to say is which models are
        # actually installed, because "the configured model is not here" without
        # that list is half an error message.
        models = client.get("/models", headers=headers)
        model_body = models.json() if models.status_code == 200 else {}
        check("GET /models reports the configured model and what the server has",
              models.status_code == 200
              and {"reachable", "configured", "configured_present", "names",
                   "installed", "hint"} <= set(model_body),
              sorted(set(model_body) ^ {"reachable", "configured", "configured_present",
                                        "names", "installed", "hint"}))
        check("GET /models needs a token like every other read of service state",
              client.get("/models").status_code == 401)
        check("GET /models agrees with itself about whether the model is present",
              model_body.get("configured_present")
              == (model_body.get("configured") in (model_body.get("names") or [])),
              f"configured={model_body.get('configured')} names={model_body.get('names')}")
        check("GET /models offers a fix, and only when there is something to fix",
              bool(model_body.get("hint"))
              == (model_body.get("reachable") is True
                  and model_body.get("configured_present") is False),
              str(model_body.get("hint"))[:120])

        # ---- collections ---------------------------------------------------
        # The invariant under test is the last one: deleting a collection moves
        # its sources, it does not delete them. Everything is restored in the
        # finally block, because this runs against the real index.
        listed = client.get("/collections", headers=headers).json()
        collections_before = listed.get("collections")
        sources_before = client.get("/sources", headers=headers).json()
        moved_source = (sources_before.get("sources") or [{}])[0]
        original_collection = moved_source.get("collection", "general")
        group = "apitest-group"
        renamed = "apitest-renamed"
        try:
            created = client.post("/collections", headers=headers, json={"name": f"  {group} "})
            check("POST /collections declares a collection (201)",
                  created.status_code == 201 and created.json().get("collection") == group,
                  created.json())
            check("a declared collection is listed while still empty",
                  any(row["collection"] == group and row["sources"] == 0
                      for row in (created.json().get("collections") or [])))
            check("declaring it again is 409",
                  client.post("/collections", headers=headers,
                              json={"name": group}).status_code == 409)
            check("and 409 case-insensitively, so 'Group' cannot sit beside 'group'",
                  client.post("/collections", headers=headers,
                              json={"name": group.upper()}).status_code == 409)
            for bad, why in (("", "empty"), ("a/b", "a path separator"),
                             ("a\nb", "a control character")):
                refused = client.post("/collections", headers=headers, json={"name": bad})
                check(f"a name with {why} is 400, not stored",
                      refused.status_code == 400, f"{bad!r} -> {refused.status_code}")
            check("a viewer may read collections",
                  client.get("/collections", headers=viewer_headers).status_code == 200)
            check("a viewer may not create one (403, not 401)",
                  client.post("/collections", headers=viewer_headers,
                              json={"name": "nope"}).status_code == 403)

            assigned = client.post(f"/collections/{group}/sources", headers=headers,
                                   json={"source_ids": [moved_source["source_id"], "not-real"]})
            body = assigned.json() if assigned.status_code == 200 else {}
            check("sources can be filed under a collection, one request for many",
                  assigned.status_code == 200 and body.get("moved") == 1
                  and body.get("not_found") == ["not-real"], body)
            scoped = client.get(f"/sources?collection={group}", headers=headers).json()
            check("and a scoped listing then returns only that collection",
                  scoped.get("count") == 1
                  and scoped["sources"][0]["source_id"] == moved_source["source_id"],
                  scoped.get("count"))

            check("renaming the default collection is refused (400)",
                  client.patch("/collections/general", headers=headers,
                               json={"name": "everything"}).status_code == 400)
            check("renaming an unknown collection is 404",
                  client.patch("/collections/not-here", headers=headers,
                               json={"name": "x"}).status_code == 404)
            check("renaming onto an existing collection is 409",
                  client.patch(f"/collections/{group}", headers=headers,
                               json={"name": "general"}).status_code == 409)
            rename_body = client.patch(f"/collections/{group}", headers=headers,
                                       json={"name": renamed})
            check("renaming moves the sources with it",
                  rename_body.status_code == 200
                  and rename_body.json().get("sources_moved") == 1
                  and client.get(f"/sources?collection={renamed}",
                                 headers=headers).json().get("count") == 1,
                  rename_body.json())

            deleted = client.delete(f"/collections/{renamed}", headers=headers)
            deleted_body = deleted.json() if deleted.status_code == 200 else {}
            check("deleting a collection moves its sources out and deletes none",
                  deleted.status_code == 200 and deleted_body.get("sources_moved") == 1
                  and deleted_body.get("sources_deleted") == 0
                  and deleted_body.get("moved_to") == "general", deleted_body)
            after = client.get("/sources", headers=headers).json()
            still_there = [row for row in after.get("sources") or []
                           if row["source_id"] == moved_source["source_id"]]
            check("the source itself survives, back in the default collection",
                  len(still_there) == 1 and still_there[0]["collection"] == "general"
                  and after.get("count") == sources_before.get("count"),
                  {"count": after.get("count"), "before": sources_before.get("count")})
            detail = client.get(f"/source/{moved_source['source_id']}", headers=headers)
            check("and so do its indexed items",
                  detail.status_code == 200 and bool(detail.json().get("items")),
                  detail.status_code)
            check("deleting the default collection is refused (400)",
                  client.delete("/collections/general", headers=headers).status_code == 400)
            check("deleting an unknown collection is 404",
                  client.delete("/collections/not-here", headers=headers).status_code == 404)
        finally:
            for name in (group, renamed):
                client.delete(f"/collections/{name}", headers=headers)
            client.post(f"/collections/{original_collection}/sources", headers=headers,
                        json={"source_ids": [moved_source["source_id"]]})
            if original_collection != "general":
                client.delete(f"/collections/{original_collection}", headers=headers)
        check("the index is left exactly as the collections block found it",
              client.get("/collections", headers=headers).json().get("collections")
              == collections_before,
              f"before {collections_before}")

        logout = client.post("/logout", headers=headers)
        check("POST /logout revokes the session token",
              logout.status_code == 200 and logout.json().get("revoked") is True, logout.json())
        check("the revoked token no longer authenticates",
              client.get("/me", headers=headers).status_code == 401)

        # Five failures lock the account, and the fifth already says so: telling a
        # user "incorrect password" on the attempt that locked them invites a sixth.
        codes = [client.post("/login", json={"username": username, "password": "nope"}).status_code
                 for _ in range(6)]
        check("wrong password is 401 until the lock, then 423",
              codes[:4] == [401] * 4 and codes[4:] == [423, 423], codes)
        locked = client.post("/login", json={"username": username, "password": password})
        detail = (locked.json() or {}).get("detail") or {}
        check("a locked account refuses the right password as 423 and says until when",
              locked.status_code == 423 and detail.get("locked_until")
              and "set-password" in (detail.get("hint") or ""), detail)
        check("an unknown account is still 401, not 423",
              client.post("/login", json={"username": "no-such-user", "password": "x"}).status_code == 401)

        if args.with_query:
            store.create_user("apitest-query", password, Role.ADMIN)
            qlogin = client.post("/login", json={"username": "apitest-query", "password": password})
            qheaders = {"Authorization": f"Bearer {qlogin.json().get('token')}"}
            result = client.post("/query", headers=qheaders,
                                 json={"query": "What percentage of the budget was utilised?",
                                       "query_modality": "text"})
            data = result.json() if result.status_code == 200 else {}
            required = {"answer", "citations", "retrieved_items", "relationships", "abstained",
                        "confidence", "error_code", "query_representation", "latency_ms",
                        "stages", "technical_detail"}
            check("POST /query returns every field the React UI reads",
                  result.status_code == 200 and not (required - set(data)), sorted(required - set(data)))
            recorded = client.post(
                "/query", headers=qheaders,
                params={"session_id": "apitest-live-session"},
                json={"query": "What is the top risk carried into 2025?",
                      "query_modality": "text"})
            live = recorded.json() if recorded.status_code == 200 else {}
            live_id = live.get("session_id")
            if live_id:
                made_sessions.append(live_id)
            check("POST /query?session_id records the turn and returns the session",
                  recorded.status_code == 200 and bool(live_id), live_id)
            if live_id:
                turn = client.get(f"/history/sessions/{live_id}", headers=qheaders).json()
                messages = turn.get("messages") or []
                stored = messages[-1] if messages else {}
                trace = stored.get("trace") or {}
                check("the recorded answer carries its trace, citations and stages",
                      len(messages) == 2 and bool(stored.get("citations"))
                      and bool(trace.get("stages")) and trace.get("latency_ms") is not None,
                      {"messages": len(messages), "stages": len(trace.get("stages") or [])})
                client.delete(f"/history/sessions/{live_id}", headers=qheaders)
                made_sessions.remove(live_id)
            unrecorded = client.post("/query", headers=qheaders,
                                     json={"query": "not recorded", "query_modality": "text"})
            check("a query with no session_id is not recorded",
                  unrecorded.status_code == 200 and unrecorded.json().get("session_id") is None)

            # ---- streaming -------------------------------------------------
            # The refusal first: abstention happens before generation, so a
            # question the corpus cannot answer must stream no tokens at all and
            # still arrive in well under a second. A streaming route that turned
            # a decisive refusal into a slow one would be worse than no streaming.
            refused = client.post(
                "/query_stream", headers=qheaders,
                json={"query": "What was Project Alpha's budget in 2040?",
                      "query_modality": "text"})
            refused_lines = [json.loads(line) for line in refused.text.splitlines()
                             if line.strip()]
            refused_tokens = [event for event in refused_lines
                              if event.get("type") == "token"]
            refused_final = next((event["content"] for event in refused_lines
                                  if event.get("type") == "result"), None)
            check("POST /query_stream refuses without streaming a single token",
                  refused.status_code == 200 and not refused_tokens
                  and refused_final is not None and refused_final.get("abstained") is True,
                  f"{len(refused_tokens)} token line(s) before the refusal")
            check("the streamed result carries the same contract as POST /query",
                  refused_final is not None and not (required - set(refused_final)),
                  sorted(required - set(refused_final or {})))
            check("the stream is newline-delimited JSON and is not buffered",
                  str(refused.headers.get("content-type", ""))
                  .startswith("application/x-ndjson")
                  and refused.headers.get("cache-control") == "no-store",
                  refused.headers.get("content-type"))

            answered = client.post(
                "/query_stream", headers=qheaders,
                json={"query": "What percentage of the budget was utilised?",
                      "query_modality": "text"})
            answered_lines = [json.loads(line) for line in answered.text.splitlines()
                              if line.strip()]
            tokens = "".join(event.get("content", "") for event in answered_lines
                             if event.get("type") == "token")
            answered_final = next((event["content"] for event in answered_lines
                                   if event.get("type") == "result"), None)
            # A prefix, not an equality: a stream cannot un-say a token, so when a
            # post-generation guard fires the correction is appended to the answer
            # and the tokens already sent stay sent. Asserting equality here would
            # fail on exactly the behaviour that is correct.
            check("an answered question streams tokens that the final answer extends",
                  answered.status_code == 200 and answered_final is not None
                  and bool(tokens)
                  and str(answered_final.get("answer") or "").startswith(tokens),
                  f"streamed {tokens[:60]!r}, final "
                  f"{str((answered_final or {}).get('answer'))[:60]!r}")
            check("the streamed answer carries its citations and stage trace",
                  answered_final is not None
                  and isinstance(answered_final.get("citations"), list)
                  and len(answered_final.get("stages") or []) >= 15,
                  len((answered_final or {}).get("stages") or []))

            check("the query response carries a stage trace with a total",
                  len(data.get("stages") or []) >= 15
                  and any(s.get("stage") == "total" for s in data.get("stages") or []),
                  len(data.get("stages") or []))
            print(f"        answer: {str(data.get('answer'))[:120]!r}")
            print(f"        citations: {len(data.get('citations') or [])}, "
                  f"evidence: {len(data.get('retrieved_items') or [])}")
    finally:
        for name in (username, "apitest-viewer", "apitest-query", first_admin):
            if store.get(name) is None:
                continue  # never created in this mode; deleting it would just be noise
            try:
                store.delete_user(name)
            except Exception as exc:  # pragma: no cover - cleanup must not mask a failure
                print(f"        could not delete {name}: {exc}")

    for session_id in made_sessions:
        try:
            history.delete_session(session_id)
        except Exception as exc:
            print(f"        could not delete session {session_id}: {exc}")

    after = db_counts(settings.auth_db)
    check("the account database is left exactly as it was found",
          before == after, f"before {before} after {after}")
    index_after = index_state(settings.index_dir)
    check("the index is left exactly as it was found",
          index_before == index_after,
          f"before {index_before} after {index_after}")
    history_after = history.stats()
    check("chat history is left exactly as it was found",
          {k: history_before[k] for k in ("sessions", "messages")}
          == {k: history_after[k] for k in ("sessions", "messages")},
          f"before {history_before['sessions']}/{history_before['messages']} "
          f"after {history_after['sessions']}/{history_after['messages']}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    import warnings

    # TestClient emits a Starlette deprecation notice on this pinned version; it
    # is not a finding about this project and would only obscure real failures.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    raise SystemExit(main())
