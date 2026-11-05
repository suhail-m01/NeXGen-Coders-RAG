"""Test the browser-facing API surface without starting a server.

    python scripts/apitest.py                # 25 checks, no LLM, no network
    python scripts/apitest.py --with-query   # +2 more: the live /query contract (needs Ollama)
    python scripts/apitest.py -v             # print every check

This covers the part of the API that a browser actually uses, which is the part
`scripts/selftest.py` cannot reach: the static React bundle and the relative asset
paths under `/ui`, the first-run bootstrap, the session endpoints, the role and
lockout behaviour, the stale-version endpoint, and the manifest. It runs through FastAPI's TestClient, so
nothing binds a port and no Ollama call is made unless `--with-query` is passed.

One thing to know before reading the code: `config.auth_db` is derived from
`DATA_DIR`, so there is no way to point this at a throwaway account database while
still testing against the real index. It therefore creates one account, and removes
it in a `finally` block, and asserts at the end that the database holds exactly as
many accounts and tokens as it did at the start. If that assertion ever fails, this
script left credentials behind and should be treated as broken.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
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

        health = client.get("/health").json()
        runtime = health.get("runtime") or {}
        check("GET /health carries the runtime block and the index",
              bool(runtime.get("git_commit")) and "index" in health and "guards" in runtime,
              {"commit": runtime.get("git_commit"), "status": health.get("status")})

        # A viewer may ask but not delete; the distinction has to survive the API.
        store.create_user("apitest-viewer", password, Role.VIEWER)
        viewer_login = client.post("/login", json={"username": "apitest-viewer", "password": password})
        viewer_headers = {"Authorization": f"Bearer {viewer_login.json().get('token')}"}
        check("a viewer may read the manifest",
              client.get("/manifest", headers=viewer_headers).status_code == 200)
        check("a viewer may not delete a source (403, not 401)",
              client.delete("/source/does-not-exist", headers=viewer_headers).status_code == 403)

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

    after = db_counts(settings.auth_db)
    check("the account database is left exactly as it was found",
          before == after, f"before {before} after {after}")

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
