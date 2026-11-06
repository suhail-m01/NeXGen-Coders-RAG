# Deploying Evidence AI

Evidence AI runs in two inference modes. What you can deploy, and where,
depends on which mode you need.

| | Offline mode | Online mode |
| --- | --- | --- |
| Generation | Local Ollama (`qwen2.5vl:3b`) | Cloud API |
| Needs Ollama | **Yes** | **No** |
| Needs internet at runtime | No | Yes |
| Runs on a hosted server | Not practically | **Yes** |

Ingestion (OCR, image understanding, transcription, embeddings, retrieval,
reranking) is local in **both** modes. Only answer generation differs.

---

## 1. Local — both modes

```powershell
cd C:\EvidenceAI
copy .env.example .env      # then edit .env
.\start.ps1
```

For Online mode, set one variable in `.env`:

```
OPENAI_API_KEY=sk-your-real-key-here
```

`start.ps1` no longer exits when Ollama is missing — it warns and starts
anyway, so Online mode works on a machine with no Ollama installed.

---

## 2. Hosted Online deployment (Streamlit Community Cloud)

The public user needs **no Ollama, no Python, no Git, no Tesseract, no local
models** — only a browser.

### Steps

1. **Push to GitHub.** Confirm `.env` is not included:
   ```bash
   git check-ignore .env && echo "ignored - safe to push"
   ```
2. Go to <https://share.streamlit.io> and connect the repository.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-real-key-here"
   INFERENCE_BACKEND = "online"
   ```
   These are stored by the platform, never in Git. `config.py` bridges
   `st.secrets` into the environment automatically.
5. Deploy. Your public URL will be
   `https://<app-name>.streamlit.app`.

### Files that make this work

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `packages.txt` | apt packages — Tesseract OCR and `libgl1` for OpenCV |
| `.env.example` | documents every variable; placeholders only |

### Honest constraints of the hosted build

These are real and worth knowing before you rely on a hosted demo:

- **Offline mode will show "not ready" on the host.** There is no Ollama
  there. That is correct behaviour, not a bug — the UI offers *Switch to
  Online*.
- **The index does not travel by default.** `index/` is gitignored, so a
  fresh deployment starts empty and users must ingest through the UI. To
  ship a prepopulated corpus, commit `index/` deliberately — and only if the
  documents in it are safe to make public.
- **Resource limits.** The Community Cloud tier gives ~1 GB RAM. `torch`,
  `faiss` and `faster-whisper` together are close to that ceiling. Document
  and image ingestion are fine; audio transcription may be tight.
- **Evidence leaves the machine in Online mode.** Retrieved evidence is sent
  to the cloud provider to write the answer. For sensitive material, use
  Offline mode locally.

---

## 3. The API and the React UI as one service

This is the deployment to prefer for anything beyond a personal laptop: one
process, one port, no CORS, no second web server.

```bash
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

`api.py` serves the built React bundle from `webui/dist` at `/ui`, and `/`
redirects there. Because the UI is served by the same process it calls, every
request it makes is a **relative URL** — the bundle contains no host, no port and
no environment-specific value, so the same `dist/` works behind any prefix, on any
machine, with no rebuild. `webui/dist` is committed for exactly this reason: a
deployment needs no Node toolchain. If you do change the UI, rebuild with
`cd webui && npm install && npm run build`.

Before exposing it off-machine:

- **Keep `API_AUTH_ENABLED=true`.** Every endpoint except `/health`, `/login`,
  `/logout` and the static UI requires a bearer token bound to a role. `false`
  makes the service an open door on whatever interface it is bound to.
- **Put TLS in front of it.** The service speaks plain HTTP; passwords and tokens
  travel in the clear without a proxy. nginx, Caddy or your platform's TLS
  termination is the operator's job, and none of it needs a code change.
- **Create the first account headlessly.** `python scripts/create_user.py <name>
  <password> --role admin`, then mint tokens for anything that is not a browser.
- **Persist three directories.** `index/`, `history/` and `auth/` are SQLite plus
  a FAISS file, and they default to folders beside the code. On a container host,
  mount volumes for them or every restart loses the corpus, the sessions and the
  accounts. `INDEX_DIR` can be set independently; the account and history
  databases are derived from `DATA_DIR`, so moving one moves both.
- **Give it disk and RAM headroom.** `GET /health` reports free disk, available
  RAM and uptime, and reports `degraded` rather than `ok` when either is too low
  for ingestion or model loading. Poll it; it loads no model.

The Streamlit operator UI can run alongside it (`streamlit run app.py`) or not at
all — the React UI covers asking, evidence, trace, sources and health, while
Streamlit adds account management, chat-history replay and export.

---

## 4. Other hosts

Any container host works. The application is a normal Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT
```

Supply `OPENAI_API_KEY` through the platform's secret manager — never in the
image or the repository. `ONLINE_BASE_URL` lets you point at any
OpenAI-compatible endpoint (Azure OpenAI, OpenRouter, an internal gateway)
without changing code.

---

## 5. Security checklist before pushing

```bash
git check-ignore .env                                  # must print .env
git ls-files | grep -x ".env" && echo "STOP"           # must find nothing
git grep -nE "sk-(proj-)?[A-Za-z0-9_-]{20,}" HEAD      # must find nothing
```

The key is only ever read from configuration and sent in the `Authorization`
header. It is never logged, never rendered in the UI, and is redacted from
API error bodies before they reach a message.

For the local accounts and API tokens:

```bash
python scripts/create_user.py --list           # who can sign in
python scripts/create_user.py --list-tokens    # which tokens are live
python scripts/create_user.py --revoke-token eai_AbCdEfGh
python scripts/manifest.py --json              # the build you are about to ship
```

- Passwords are PBKDF2-HMAC-SHA256 with a per-user salt; tokens are stored only
  as SHA-256 digests and shown once. Deleting an account deletes its tokens in the
  same call, so a removed user cannot keep a live credential.
- Five failed sign-ins lock an account for five minutes. A locked account gets
  HTTP 423 with the unlock time; an unknown account and a wrong password both get
  401, so the username list is not enumerable by status code.
- `auth/users.db`, `history/chat_history.db` and `index/` are **unencrypted**.
  Anyone with the files has the corpus and the session history. Use OS or disk
  encryption if the material is sensitive, and never commit them.
- Chat history is an API resource, so decide who should reach it before you deploy:
  reading and deleting a session needs only the `ask` permission (every role), while
  exporting one needs `export_history` (admin). Sessions carry no owner column, so
  "your own session" means "any session" — a viewer can delete a session an admin
  created. If that is not acceptable for your material, do not expose the port.
- The routes that run models are rate limited: 30 questions/minute with a burst of 6
  and 60 ingestions/minute with a burst of 20, one token charged per uploaded file.
  A refusal is `429` with a `Retry-After` header and happens before any model loads.
  The counters live in process memory, so **two workers means twice the limit**; put
  the limiter behind your proxy's own limiting if you scale out, or run one worker.
  Only set `RATE_LIMIT_TRUST_FORWARDED_FOR=true` behind a proxy you control — it makes
  the rightmost `X-Forwarded-For` hop the client, and any client can write the rest of
  that header.
- The React UI keeps its bearer token in memory while the page lives and mirrors
  it to `localStorage` as a reload bridge where the browser allows storage.
  Sandboxed embed viewers - iframes without `allow-same-origin`, which is how
  preview panes ship a running app - make every storage call throw; there,
  memory carries the session instead of degrading every data call to a 401.
- Sign-out calls `POST /logout`, which revokes the token server-side and clears
  the session cookie rather than only forgetting it in the browser.
- The server accepts the token on four transports, in order: the
  `Authorization: Bearer` header, an `X-EAI-Token` header, the HttpOnly
  `eai_session` cookie that `/login` sets, and an `access_token` query
  parameter. The last three exist because reverse proxies and embed viewers
  are known to strip `Authorization`; the UI switches to the query parameter
  only after it observes a header attempt being dropped, since a token in a
  URL can land in proxy logs. Whichever transport served a request is logged
  at INFO, so a 401 in the wild is diagnosable from the server log.
- Attachments (`POST /query_attachment`) accept up to five documents or images per
  question, each within `max_file_bytes`. They are extracted into ephemeral evidence
  for that turn only - never indexed, never persisted - and flow through the same
  coverage and entity guards, abstention semantics and citation building as indexed
  material, so an attachment cannot buy an unguarded answer. Audio is deliberately
  excluded: transcription belongs to the ingest queue with its own rate bucket, not
  to a chat round trip. The turn is then routed by what the question asks. An
  instruction about the attachment itself - "explain this document", "summarise the
  report" - is answered from the attachment alone: a cross-encoder scores an
  instruction at the floor against every passage including the attachment, so a
  relevance race would be noise, and corpus pages scraping over the abstention bar
  measurably bled into such summaries. A factual question keeps the normal
  semantics with the attachment as an ordinary candidate: the score gate may not
  refuse the turn on attached evidence, corpus blocks join only above the
  per-modality relevance bar, and the merged context is ordered relevance-first,
  because pinning the attachment to the front measurably answered a corpus fact
  (87 % budget utilisation) from an attached scheme's passing criteria (50 %). The
  per-source diversity cap relaxes on attachment turns, because the attached file
  is one source by construction.
  That is normal for a browser app and worth knowing: on a shared machine, sign
  out rather than closing the tab.
