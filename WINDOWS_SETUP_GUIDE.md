# Evidence AI — Windows Setup Kit

**Offline-runtime multimodal RAG over documents, images and audio, with per-item citations and an abstention gate.**

Project: `AnishPreetham/Multi-Modal-RAG` — this kit is that repository **plus the files it was missing**, packaged so you can install and run it on a Windows PC.

---

## 1. What is in this ZIP

| Item | What it is |
|---|---|
| **`READ_ME_FIRST.txt`** | **Start here** — the commands on one screen |
| **`WINDOWS_SETUP_GUIDE.md`** | **This document** |
| `app.py` | Streamlit web UI (port **8501**) |
| `api.py` | FastAPI service (port **8000**, docs at `/docs`) |
| `rag_pipeline.py` | The pipeline: extract → embed → retrieve → rerank → abstain → generate → cite |
| `documents.py`, `images.py`, `audio.py` | Per-modality extractors (PDF/DOCX/CSV, OCR, Whisper) |
| `embeddings.py`, `vector_store.py`, `retriever.py`, `reranker.py` | MiniLM embeddings, FAISS + SQLite store, cross-encoder reranking |
| `citations.py`, `relationships.py`, `evaluation.py` | Programmatic citations, cross-source links, evaluation harness |
| `config.py`, `.env`, `.env.example` | All tunables; `.env` overrides defaults |
| `data/` | Demo dataset (PDF, DOCX, CSV, PNG, WAV) + your `MCA_MP_58-mini-project.pdf` |
| `requirements-lock.txt` | **Pinned** dependency versions (use this one) |
| `setup.ps1`, `start.ps1` | The repository's original scripts |
| **`setup_windows.ps1`** | **Added by me** — corrected installer (see §7) |
| **`scripts/verify_env.py`** | **Added by me** — proves every import works |
| **`scripts/prepare_models.py`** | **Added by me** — pre-downloads all 4 models + tests each |
| **`scripts/ingest_demo.py`** | **Added by me** — builds the index from `data/` |
| **`scripts/verify_offline.py`** | **Added by me** — blocks all internet, then answers a question |
| **`scripts/smoke_test.py`** | **Added by me** — asks test questions, prints answers + citations |

No index is shipped: it is built fresh on your machine during setup, which also proves the pipeline works on your hardware.

---

## 2. What to download (complete list)

### A. Installers — download these yourself

| # | What | Where | Size | Needed? |
|---|---|---|---|---|
| 1 | **Python 3.11+** (3.13 works) | https://www.python.org/downloads/windows/ | ~30 MB | **Required** |
| 2 | **Ollama for Windows** | https://ollama.com/download | ~1.1 GB | **Required** for offline answering |
| 3 | **Tesseract OCR** (UB Mannheim build) | `winget install UB-Mannheim.TesseractOCR`<br>or https://github.com/UB-Mannheim/tesseract/wiki | ~60 MB | Required for **image OCR** |
| 4 | **LibreOffice** | `winget install TheDocumentFoundation.LibreOffice` | ~350 MB | **Optional** — only for legacy `.doc` |

> **Python installer: tick “Add python.exe to PATH”.** This is the single most common cause of a failed setup.

### B. AI models — downloaded automatically by `setup_windows.ps1`

| Model | Role | Source | Size |
|---|---|---|---|
| **`qwen2.5vl:3b`** | Writes the answers; **vision-capable** | Ollama (`ollama pull`) | **~3.2 GB** |
| `sentence-transformers/all-MiniLM-L6-v2` | Embeds text into 384-d vectors | HuggingFace | ~90 MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks candidates; drives confidence + abstention | HuggingFace | ~90 MB |
| `faster-whisper base` (CTranslate2, int8) | Audio → timestamped text | HuggingFace | ~145 MB |

HuggingFace models land in `%USERPROFILE%\.cache\huggingface`. **Total download ≈ 4.5 GB; total disk use ≈ 12 GB** (the Python environment with CPU PyTorch is the largest part).

**Lower-spec alternatives** (pass `-Model` to the setup script):

| Model | Size | Trade-off |
|---|---|---|
| `qwen2.5vl:3b` (default) | 3.2 GB | Needs ~4 GB VRAM. Best quality, can look at images |
| `qwen2.5:1.5b` | ~1.0 GB | Text only, no vision. Good for 8 GB RAM, no GPU |
| `qwen2.5:0.5b` | ~0.4 GB | Text only, weak wording/numbers. Last resort |

### C. APIs and keys — **none required**

| Mode | Key needed? | Details |
|---|---|---|
| **Offline** (default) | **No key, no account, no internet** | Everything runs locally: Ollama + HuggingFace models |
| Online (optional) | `OPENAI_API_KEY` in `.env` | Only if you *want* cloud answers. Default model `gpt-4o-mini`, and `ONLINE_BASE_URL` accepts any OpenAI-compatible endpoint (Azure OpenAI, OpenRouter, a self-hosted gateway) |

The shipped `.env` has an **empty** `OPENAI_API_KEY` — no secrets are in this ZIP. Switch backends at runtime in the UI sidebar.

---

## 3. Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 11 |
| RAM | 8 GB | 16 GB |
| Disk | 12 GB free | 15 GB free (SSD) |
| GPU | none (CPU works, slowly) | **NVIDIA with 4 GB VRAM** for `qwen2.5vl:3b` |
| Internet | needed **once** for downloads | then fully offline |

The 4 GB GPU is entirely committed to Ollama, which is why PyTorch (embeddings + reranker) is deliberately installed as the **CPU** wheel.

---

## 4. Step-by-step installation

### Step 1 — Install Python
Run the installer from python.org, **tick “Add python.exe to PATH”**, then in a new PowerShell window:
```powershell
python --version
```
Expect `Python 3.11.x` or newer. If it says *not recognized*, reinstall and tick the PATH box.

### Step 2 — Install Ollama
Download `OllamaSetup.exe` from https://ollama.com/download and install it. Then **open a new PowerShell window** and check:
```powershell
ollama --version
```
Ollama starts its server automatically (port 11434). If it is not running: `ollama serve`.

### Step 3 — Install Tesseract OCR
```powershell
winget install UB-Mannheim.TesseractOCR
```
No `winget`? Download the installer from the UB-Mannheim wiki (link in §2A). Verify it landed where `.env` expects:
```powershell
Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```
`True` means image OCR will work with no further configuration. If it is installed elsewhere, edit `TESSERACT_CMD` in `.env` to the full path.

### Step 4 — (Optional) LibreOffice
Only needed for legacy `.doc` files. Skip it if you have no `.doc` documents.
```powershell
winget install TheDocumentFoundation.LibreOffice
```

### Step 5 — Unzip the kit
Extract to a **short local path with no spaces**, and **not** inside OneDrive:
```
C:\EvidenceAI
```
OneDrive sync and antivirus scanning of a folder containing a 6 GB Python environment cause real problems; a plain local folder avoids them.

### Step 6 — Unblock the downloaded files
Windows marks files from the internet as blocked, which stops PowerShell scripts from running:
```powershell
cd C:\EvidenceAI
Get-ChildItem -Recurse | Unblock-File
```

### Step 7 — Allow scripts for this window only
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
This affects only the current PowerShell window and changes nothing system-wide.

### Step 8 — Run the setup (the long step)
```powershell
.\setup_windows.ps1
```
Low-memory machine or no 4 GB GPU instead:
```powershell
.\setup_windows.ps1 -Model qwen2.5:1.5b
```

It performs 9 steps and prints `[OK] / [WARN] / [FAIL]` for each:
1. Python 3.11+ check (tries `python`, `py -3`, `python3`)
2. RAM, free disk, GPU/VRAM check — warns if `qwen2.5vl:3b` will not fit
3. Creates `.env` from `.env.example` if absent, creates `.venv`, upgrades pip
4. Installs the **CPU** PyTorch build at the pinned version (GPU reserved for Ollama)
5. Installs `requirements-lock.txt` (pinned)
6. `scripts\verify_env.py` — every import **plus a real audio decode** through faster-whisper, which is what catches the PyAV incompatibility
7. Tesseract + LibreOffice detection
8. Ollama check, pulls the model, **writes your chosen model into `.env`**
9. `scripts\prepare_models.py` — downloads and *actually runs* each HuggingFace model

`.env` is created for you, so there is nothing to configure for offline use. Edit it later only to change models, thresholds or binary paths.

Then it generates the demo data and builds the index with `scripts\ingest_demo.py`.

> Expect **10–30 minutes** depending on your connection and disk. Downloads resume if interrupted — just re-run.

### Step 9 — Start the app
```powershell
.\start.ps1
```
- UI → **http://localhost:8501**
- API → http://127.0.0.1:8000/docs

Lighter option (UI only, skips the API process):
```powershell
.\start.ps1 -UiOnly
```

### Step 10 — Verify it works
In a second PowerShell window:
```powershell
.\.venv\Scripts\python scripts\smoke_test.py
```
This asks four questions (tabular, audio, document, and one that should be refused) and prints each answer with its **citations**, confidence and latency. Check that the cited page/timestamp actually contains the claim.

Then prove the offline guarantee:
```powershell
.\.venv\Scripts\python scripts\verify_offline.py
```
It blocks every non-loopback socket and DNS lookup, then answers a question. `PASSED` means nothing reached the internet. (Local Ollama on 127.0.0.1 is intentionally allowed — “offline” means no internet, not no local processes.)

---

## 5. Using it

**Sign in** — the UI is gated. On first run there are no accounts, so it offers to create the **first administrator** (no default password is shipped, on purpose). Accounts live in `auth\users.db` as PBKDF2-HMAC-SHA256 hashes with a per-user salt; five failed attempts lock an account for five minutes.

Three roles:

| Role | Ask | Ingest | Delete source | Export session | Manage accounts |
|---|---|---|---|---|---|
| `viewer` | ✓ | — | — | — | — |
| `analyst` | ✓ | ✓ | — | — | — |
| `admin` | ✓ | ✓ | ✓ | ✓ | ✓ |

Administrators get **Manage accounts** in the sidebar (add accounts, change roles, reset passwords, delete accounts — you cannot delete the one you are signed in with). Accounts can also be managed headlessly:
```powershell
.\.venv\Scripts\python scripts\create_user.py --username reviewer --role viewer
.\.venv\Scripts\python scripts\create_user.py --list
.\.venv\Scripts\python scripts\create_user.py --username reviewer --set-password
```

**Ask a question** — UI → *Ask* tab. Every answer lists citations with the exact page number (documents), timestamp range (audio) or OCR region (images). If evidence is too weak, it **abstains** instead of guessing, and says so.

**Add your own files** — three ways:
1. UI → *Upload* tab (drag and drop)
2. CLI: `.\.venv\Scripts\python scripts\ingest_demo.py --dir "D:\my documents"`
3. API: `POST http://127.0.0.1:8000/ingest`

Supported: PDF, DOCX, `.doc` (needs LibreOffice), CSV/XLSX, PNG/JPG (OCR + optional vision), WAV/MP3/M4A/FLAC/OGG (Whisper with timestamps). Limit: 200 MB per file (`MAX_FILE_BYTES` in `.env`).

Re-ingesting is safe — `source_id` is a content hash and existing item IDs are skipped, so nothing duplicates. `scripts\ingest_demo.py` goes further and skips an already-indexed file **before** extracting it, which matters because extraction is the expensive stage: without that, re-running would make Whisper re-transcribe every recording for nothing. Add `--force` when you genuinely want re-extraction (after changing an extractor or swapping a model).

**Group sources into collections** — the *Ingest* tab has a **Collection** field (default `general`); on the CLI use `--collection`:
```powershell
.\.venv\Scripts\python scripts\ingest_demo.py --dir "D:\contracts" --collection contracts-2026
```
The *Ask* tab then has a **Search scope** dropdown. Scoping is applied inside retrieval (both the dense and the BM25 leg), so a scoped search never leaks evidence from another collection. All collections still live in one index — an index built before collections existed is migrated automatically on first open, with every existing source filed under `general`.

**Review a past session** — the sidebar **💬 History** panel lists saved sessions. Every question and answer is stored in `history\chat_history.db` (deliberately outside `index\`, so rebuilding the index never destroys a conversation). Each saved answer keeps its full audit trail: the evidence ranks, both scores, filenames, locations, excerpts, speaker labels, OCR confidence, citations, stage timings and any figure the numeric guard could not verify. Expanding a past answer replays all of it **from the stored snapshot**, without re-running the query and without needing the source to still be indexed. **⬇ Markdown** and **⬇ CSV** export a session for a reviewer.

**Guardrails you will see** — three of them, all before or after the model rather than inside it:
- *Abstention gate* — weak evidence is refused before the LLM is called (image evidence is judged on its own threshold, `ABSTENTION_THRESHOLD_IMAGE`).
- *Coverage guard* — refuses when a year the question commits to appears nowhere in the evidence.
- *Numeric guard* — after generation, any figure absent from the evidence gets a visible `Evidence warning:` line naming it and the sources searched. It never deletes an answer; the UI badge downgrades "Grounded answer" to "Partially grounded".

**OCR you can trust-check** — images and scanned PDF pages are OCR'd with per-word confidence. Weak words are dropped, digits are held to a higher bar than prose, small screenshots are upscaled first, and an unreliable passage is annotated in the evidence itself so the model is told a missing figure means *not legible*, never *not present*. The *Sources* and evidence panels show the confidence, how many words were kept and how many numeric tokens were withheld.

**Speaker labels in audio** — long recordings are diarized locally (MFCC + clustering); transcript segments carry `Speaker N` and the label is stored in item metadata. A clip too short to cluster is treated as one speaker rather than guessed at.

**Delete a source** — *Sources* tab, inside a source's expander (admin role only). It removes the source, its items and every relationship touching them, and rebuilds the vector index so the remaining `vector_id`s stay correct. Two things are deliberately left alone: the original file on disk, and any chat-history traces that cited it — a deleted source can still be audited after the fact.

**Score answers with an LLM judge (optional)**:
```powershell
.\.venv\Scripts\python evaluation.py --judge
```
This adds four advisory columns per query. Read `judge_consistency_rate` **first**: every binary question is asked twice with the option order swapped, and only counts when both replies pick the same statement. A model that always answers "A" disagrees with itself on everything, so its scores come back empty and the report prints `UNUSABLE at this model size`. Measured on `qwen2.5:0.5b`: consistency 0.0, verdict UNUSABLE — the judge arm needs a 3B+ local model to mean anything. The deterministic metrics (keyword coverage, number checks, retrieval set math) remain the score of record either way.

**Go fully offline** — after Step 8 succeeds, disconnect the network. Ollama, embeddings, reranker and Whisper all read from local caches.

**Run the evaluation harness**:
```powershell
.\.venv\Scripts\python evaluation.py
```
Reports (Markdown/CSV/JSON) are written to `evaluation\reports\`.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `...cannot be loaded because running scripts is disabled` | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (Step 7) |
| `python : not recognized` | PATH not set | Reinstall Python, tick **Add to PATH**, open a **new** window |
| Setup fails at “Import verification” | Partial install | `.\.venv\Scripts\python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` |
| pip cannot find `torch==2.13.0+cpu` | CPU builds live on PyTorch's index | The script already passes `--extra-index-url`; if it still fails, use `.\setup_windows.ps1 -Loose` |
| Audio ingestion error mentioning `metadata_errors` | PyAV ≥ 19 broke faster-whisper 1.2.1 | `.\.venv\Scripts\python -m pip install av==18.1.0 faster-whisper==1.2.1 ctranslate2==4.8.1` |
| `WinError 1314` / symlink error | Windows without Developer Mode | Already handled: `config.py` sets `HF_HUB_DISABLE_SYMLINKS=1` |
| Images indexed but no OCR text | Tesseract missing or wrong path | Install it; check `TESSERACT_CMD` in `.env` points to the real `tesseract.exe` |
| `model not found` from Ollama | Model not pulled | `ollama list`, then `ollama pull qwen2.5vl:3b` |
| Very slow answers / out of memory | Model too big for the machine | Re-run setup with `-Model qwen2.5:1.5b`; set `OLLAMA_NUM_CTX=4096` in `.env`; use `.\start.ps1 -UiOnly`; close other apps |
| First question takes 30–60 s | Cold model load | Normal. `.env` sets `OLLAMA_KEEP_ALIVE=30m` so later queries are fast |
| Port 8501 already in use | Another app | `.\start.ps1 -UiPort 8502` |
| UI spinner never ends | Server died or model reloading | Refresh the browser; check the PowerShell window for errors |
| “No index yet” warning | Nothing ingested | `.\.venv\Scripts\python scripts\ingest_demo.py` |
| `module 'coverage' has no attribute 'types'` during audio ingestion | An old `coverage.py` in the project root shadows the PyPI package numba imports | The module is now `coverage_guard.py`. Delete or rename any leftover `coverage.py` |
| Audio ingests but no `Speaker N` labels | Clip too short to cluster, or diarization off | Normal for short clips; check `DIARIZATION_ENABLED=true` and `DIARIZATION_MAX_SPEAKERS` |
| Answer carries an `Evidence warning:` line | The numeric guard found a figure absent from the evidence | Read the cited passage; if the figure is right, the evidence text is likely OCR-damaged — check the OCR confidence shown in the evidence panel |
| Image questions abstain even though the image is indexed | OCR could not read it reliably | The evidence panel shows confidence and withheld numeric tokens; use a vision-capable model (`qwen2.5vl:3b`) for that source |
| History panel is empty after asking | Session saved to a different working directory | Sessions live in `history\chat_history.db` relative to the project root; start the app from the kit folder |
| Forgot a password | No recovery by design — hashes are not reversible | `.\.venv\Scripts\python scripts\create_user.py --username NAME --set-password` |
| "Sign-in failed" after several tries | Account locked | Wait `AUTH_LOCKOUT_MINUTES` (default 5), or reset the password with the command above, which clears the lock |
| `judge_consistency_rate` is 0.0 or low | Model too small to judge | Expected below ~3B. The judge arm is advisory; the deterministic metrics are the score of record |
| Deleted a source and want it back | Deletion only touches the index | The original file is still on disk — ingest it again |
| Antivirus/SmartScreen blocks a file | Downloaded-file mark | `Get-ChildItem -Recurse \| Unblock-File` (Step 6) |
| Setup crawls or files vanish | Folder inside OneDrive | Move the kit to `C:\EvidenceAI` and re-run |

---

## 7. What I changed vs. the original repository

**This kit modifies core files.** An earlier version of this guide said original files were unmodified; that was true of the packaging work but is no longer true of the kit, and the claim is corrected here rather than left standing.

### Upgrade programme (what changed and why)

Every item below is config-gated, so each can be switched off in `.env` without touching code.

| Area | Change | New files |
|---|---|---|
| Retrieval | Dense + BM25 hybrid, fused by reciprocal rank fusion | `lexical.py` |
| Reranking | Length normalisation (long chunks were winning on size, not relevance) | — |
| Diversity | MMR re-selection with a per-source cap | `mmr.py` |
| Audio | Local speaker diarization (MFCC + agglomerative clustering + silhouette) | `diarization.py` |
| Abstention | Calibrated threshold tooling; separate threshold for OCR/image evidence | `scripts/calibrate_abstention.py` |
| Grounding | Deterministic coverage guard (question years must appear in evidence) | `coverage_guard.py` |
| Grounding | Post-generation numeric guard with a visible caveat | `numeric_guard.py` |
| OCR | Per-word confidence, digit-specific threshold, adaptive upscaling, reliability annotation | `ocr.py` |
| Generation | Token streaming through pipeline, providers and UI | — |
| UX | Persistent chat sessions with full retrieval traces, replay and export | `chat_history.py` |
| UX | Logical collections with scoped search | — |
| Access control | Local accounts, three roles, PBKDF2 hashing, lockout, headless account CLI | `auth.py`, `scripts\create_user.py` |
| Corpus management | `delete_source` with a safe index rebuild (positional FAISS ids make in-place removal unsafe) | — |
| Evaluation | Optional LLM-judge arm with order-swap consistency checking | `llm_judge.py` |
| Reproducibility | Fixed Ollama seed, so evaluation reports are comparable run to run | — |

Modified upstream files (16, verified with `git status`): `.env.example`, `app.py`, `audio.py`, `config.py`, `documents.py`, `evaluation.py`, `images.py`, `llm.py`, `providers.py`, `rag_pipeline.py`, `requirements-lock.txt`, `requirements.txt`, `reranker.py`, `retriever.py`, `schemas.py`, `vector_store.py`.

Added by this kit: `lexical.py`, `mmr.py`, `diarization.py`, `ocr.py`, `coverage_guard.py`, `numeric_guard.py`, `chat_history.py`, `auth.py`, `llm_judge.py`, `run_demo.py`, `setup_windows.ps1`, this guide, and everything under `scripts\` (`ingest_demo.py`, `smoke_test.py`, `verify_env.py`, `prepare_models.py`, `verify_offline.py`, `calibrate_abstention.py`, `create_user.py`).

`api.py` accepts a `QueryRequest` directly, so it picked up collection scoping and the new response behaviour from `schemas.py` with no change to the contract.

**The API is authenticated.** Every endpoint except `GET /health` requires a bearer token that resolves to one of the same local accounts the UI uses, and is checked against that account's role:

```
Authorization: Bearer eai_...
```

| Endpoint | Permission needed |
| --- | --- |
| `GET /health` | none (runtime status only) |
| `POST /query`, `POST /query_file`, `GET /sources`, `GET /source/{id}`, `GET /source/{id}/file` | `ask` (viewer and above) |
| `POST /ingest` | `ingest` (analyst and above) |
| `POST /ingest` with `replace=true` | `ingest` **and** `delete` (admin) |
| `DELETE /source/{id}` | `delete` (admin) |

Mint, list and revoke tokens with `scripts\create_user.py --create-token | --list-tokens | --revoke-token`. The plaintext is shown once; only its SHA-256 digest is stored, and deleting an account kills its tokens immediately. A missing or revoked token is `401`; a valid token whose role is not allowed is `403`. `API_AUTH_ENABLED=false` turns the gate off — which leaves the API open on whatever interface it is bound to, so keep it on and keep the port on `127.0.0.1`.

One rename worth knowing about: the coverage guard module is `coverage_guard.py`, **not** `coverage.py`. A bare `coverage.py` shadows the PyPI `coverage` package that numba imports, and numba is a librosa dependency — the collision broke the audio decode path with a confusing `module 'coverage' has no attribute 'types'`. If you keep any older copy of this kit, do not rename it back.

Measured effect on the six-query evaluation set (CPU, `qwen2.5:0.5b`): 3 PASS / 3 CHECK became **5 PASS / 1 CHECK**; answer correctness 0.667 → 0.833; citation accuracy 0.381 → 0.631; groundedness 0.833 → **1.0**; hallucination rate 0.167 → **0.0**; abstention accuracy 0.667 → **1.0**. The one remaining CHECK (Q5, a broad cross-modal question) is the substitute model copying its top passage verbatim rather than synthesising — retrieval and citation for that query are correct.

### The original packaging problems (still true)

The repository's own `setup.ps1` cannot complete on a fresh clone, for two reasons I verified by running it:

1. It calls `scripts\verify_env.py`, `scripts\prepare_models.py` and `scripts\ingest_demo.py` — **the `scripts\` folder does not exist in the repo**. Setup dies at “Import verification”.
2. It installs `requirements.txt`, whose loose pins let pip resolve PyAV ≥ 19, which removed the `metadata_errors=` argument that `faster-whisper` 1.2.1 passes — so **audio ingestion breaks later**, at the worst possible moment.

So this kit adds the four missing scripts, a `smoke_test.py`, and `setup_windows.ps1` which installs the **pinned** `requirements-lock.txt` (keeping `--extra-index-url` so the `+cpu` torch build resolves) and lets you choose a smaller model with `-Model`. `setup.ps1` still works too, now that `scripts\` exists — `setup_windows.ps1` is simply the safer route.

Two smaller portability notes found while running it:
- `create_demo_data.py` generates the demo WAV with **Windows speech (SAPI)**, so it only works on Windows. On other OSes it skips audio and tells you to drop in a real WAV/MP3.
- `.env` ships with Windows paths for `TESSERACT_CMD` and `SOFFICE_CMD`. Correct on Windows; on Linux/macOS override them (e.g. `TESSERACT_CMD=/usr/bin/tesseract`).

### How this kit was tested — and what was not

**Executed and passing** (Linux sandbox, Python 3.13, CPU only, with `qwen2.5:0.5b` substituted for `qwen2.5vl:3b` because the sandbox has 2 CPUs and ~1.9 GB RAM):

| Script | Result |
|---|---|
| `scripts/verify_env.py` | exit 0 — 22 packages + 14 project modules import; decoded a real 37.1 s WAV through faster-whisper |
| `scripts/prepare_models.py` | exit 0 — embedding loaded and encoded to 384-d, cross-encoder scored a pair (logit 2.357), Whisper loaded, Ollama generated a reply |
| `scripts/ingest_demo.py` | exit 0 — built the index (6 sources / 37 items); on re-run, skipped all 6 by content hash in 0.0 s |
| `scripts/smoke_test.py` | exit 0 — 4/4 questions answered; audio citation rendered as `00:00-00:21`; the unanswerable question abstained in <1 s **before** calling the LLM |
| `evaluation.py` | exit 0 — 5 PASS / 1 CHECK; groundedness 1.0, hallucination rate 0.0, abstention accuracy 1.0, mean latency 8.8 s |
| `scripts/calibrate_abstention.py` | dry runs only — it reports the threshold it would pick; `--apply` was **not** used, the committed thresholds are hand-set from measured score ranges |
| `auth.py` + role gating | verified by direct tests: PBKDF2 hash/verify, unique salts, username/password/role validation, the three-role permission matrix, lockout after 5 failures (a locked account refuses the *correct* password), role/password change and deletion. Streamlit `AppTest` ran the whole UI as anonymous, viewer, analyst and admin with **0 exceptions**: anonymous sees only the bootstrap form, viewer sees the ingest block and the export-gated caption, admin sees Manage accounts and a delete button per source |
| `delete_source` | verified by direct test: three sources ingested, the middle one deleted, and both survivors still resolved to the correct items by distinctive queries — before deletion, after deletion, and after reopening the index from disk |
| `llm_judge.py` | verified by direct test on three answers with known verdicts (correct / wrong figure / describes-the-source). At `qwen2.5:0.5b` the judge replied "A" to all nine order-swapped questions, so consistency is 0.0 and no scores are published — the failure is reported, not hidden |
| Streaming, numeric guard, OCR confidence, collections, chat history | verified by direct tests: streamed tokens with citations intact; guard caveat appended on a fabricated wrong figure and silent on a correct one; dashboard OCR 90.5% confidence with 0/14 numeric tokens withheld; scoped retrieval returned only in-collection sources through both retrieval legs; session saved with a replayable trace and exported to Markdown/CSV |
| `scripts/verify_offline.py` | exit 0 — PASSED with **zero** outbound connections or DNS lookups |

**Not executed:** `setup_windows.ps1`, `setup.ps1` and `start.ps1`. There is no PowerShell interpreter in the sandbox, so the installer was reviewed line by line (balanced braces, `param()` first, functions defined before use, pinned torch resolution, native-command exit codes) but **has never been run**. Two real bugs were found and fixed during that review: `-match` against a multi-line array does not populate `$Matches` (which would have made the Python version test silently fail), and nested quotes in the `cmd /c` venv creation. Treat the first run of `setup_windows.ps1` as the actual test, and report any error text back — the fixes will be small.

Also **not** tested on Windows-specific paths: `pyttsx3`/SAPI demo-audio generation, the `winget` installs, GPU/VRAM detection, and the `qwen2.5vl:3b` vision path (the sandbox could not run that model at all).

---

## 8. Honest limitations

- **`confidence` is not a probability of being correct.** It is a squashed cross-encoder rerank logit — a relative ordering signal. Always read the cited passage.
- **The abstention gate is phrasing-sensitive.** Tuned by `ABSTENTION_THRESHOLD=-6.0`. A question with no supporting evidence is usually refused, but rewording can slip past it — test your own phrasings before trusting a demo.
- **Small models degrade output.** With `qwen2.5:1.5b` or `0.5b` expect weaker wording, and occasional garbled numbers. That is the substitute model, not the pipeline.
- **The image path depends on the model.** `qwen2.5vl:3b` can genuinely look at an image; text-only models fall back to OCR text. OCR is now confidence-filtered and upscaled, which fixed the demo dashboard (mean confidence 63.6% → 90.5%, and the team size reads correctly as `6 engineers` instead of a fused `22024`) — but charts, handwriting and dense tables remain unreliable, and the system will now tell you so rather than guess.
- **The numeric guard compares text, not meaning.** It catches figures absent from the evidence and normalises `32,000` / `32000`, `99.40` / `99.4` and `1.2 million` / `1200000`. It cannot catch a number that is *present* in the evidence but attached to the wrong subject — that is what citations are for.
- **The sign-in gate and the API tokens are a local gate, not multi-tenant security.** Passwords are properly hashed and compared in constant time, and lockout slows guessing, but anyone with shell access to the machine already owns the index, the history and `auth\users.db`. Nothing is encrypted at rest — use OS/disk encryption if the corpus is sensitive. `api.py` is protected by per-account bearer tokens with role checks, and is still best kept on `127.0.0.1`.
- **The LLM judge is unusable at 0.5b, measured.** It answered "A" to nine out of nine order-swapped questions about a correct answer, a wrong-figure answer and a non-answer. The consistency check catches this and reports `UNUSABLE` instead of publishing flattering scores; expect it to work with `qwen2.5vl:3b` or larger, and treat it as advisory either way.
- **Chat history is local and unencrypted.** `history\chat_history.db` stores questions, answers and evidence excerpts in plain SQLite. Fine for one reviewer on one machine; do not put it on a shared drive without encrypting it.
- **Whisper `base` on CPU** is accurate enough for clean speech but will struggle with noise, accents and overlapping speakers. Raise `WHISPER_MODEL_SIZE` to `small` if you have the RAM.

---

## 9. Quick reference

```powershell
# one-time (online)
cd C:\EvidenceAI
Get-ChildItem -Recurse | Unblock-File
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1                    # or: -Model qwen2.5:1.5b

# every use (offline)
.\start.ps1                            # UI http://localhost:8501
.\start.ps1 -UiOnly                    # lighter

# checks
.\.venv\Scripts\python scripts\verify_env.py
.\.venv\Scripts\python scripts\prepare_models.py
.\.venv\Scripts\python scripts\ingest_demo.py --dir "D:\my files"
.\.venv\Scripts\python scripts\smoke_test.py
.\.venv\Scripts\python scripts\verify_offline.py
.\.venv\Scripts\python evaluation.py
.\.venv\Scripts\python evaluation.py --judge          # advisory judge columns
.\.venv\Scripts\python scripts\create_user.py --list    # local accounts
```

Ports: **8501** UI · **8000** API · **11434** Ollama.

---

## 9. Corpus lifecycle, the evaluation gate, and chunking

### Updating a file that is already indexed

`source_id` is a content hash, so an edited file is a **new** source: the previous
version stays indexed and keeps being cited beside the new one, and either can rank
higher. Three ways to handle it:

```powershell
:: ingest and replace older versions of the same filename
python scripts\ingest_demo.py --dir D:\reports --replace

:: see which filenames have more than one indexed version
:: (the UI shows the same list on the Sources tab, and offers a Replace button)

:: rebuild the whole index from data\ with the current settings
python scripts\reindex.py --dry-run
python scripts\reindex.py --yes
```

`reindex.py` exists because `ingest_demo.py` **cannot** apply changed settings: an
unchanged file is skipped by hash before extraction, so a new `CHUNK_SIZE`, OCR
threshold or embedding model never touches it. `--force` re-extracts but leaves the old
items in place beside the new ones. `reindex.py` starts from an empty index, preserves
each source's collection, and leaves `history\chat_history.db` and `auth\users.db`
alone — saved traces stay readable, which is the point of storing them.

### The evaluation gate

Dataset v2.0 is 16 queries over the demo corpus: the original six, two multi-hop
questions whose answer needs a fact from one source joined to a fact from another,
three adversarial ones (a near-miss year, a figure that does not exist, an entity that
does not exist), a table-fidelity question with a more salient near-miss number, two
per-modality pins, and two against the long academic deck. Each carries `notes` with
its ground truth and where it comes from, so an expectation can be checked against the
corpus instead of trusted.

```powershell
python evaluation.py                 # full run, writes json/csv/md into evaluation\reports
python evaluation.py --compare       # gate against evaluation\baseline.json, exit 1 on any regression
python evaluation.py --compare --tolerance 0.02 --strict-per-query
python evaluation.py --save-baseline # promote this run to the shipped baseline
```

Gated metrics: retrieval precision/recall, MRR, citation accuracy, answer correctness,
groundedness, abstention accuracy, hallucination rate and forbidden-leak rate —
aggregate **and** per expected modality, so one modality collapsing cannot hide behind
a healthy average. Latency and the numeric-guard rate are reported but never gated:
latency measures the machine, and a higher guard rate can mean better honesty or worse
OCR. Comparing across dataset versions is refused unless you pass
`--allow-dataset-mismatch`, because the aggregates would not mean the same thing.

Tolerance defaults to `0.0`. Generation at `qwen2.5:0.5b` still jitters a little
run to run (citation accuracy moved 0.4135 → 0.401 between two runs on an identical
index), so for routine checks `--tolerance 0.02` is the honest setting; keep `0.0`
when you need to know that nothing at all moved.

### Chunking: what is implemented and what is default

`chunking.py` implements structure-aware splitting — Markdown/numbered/ALL-CAPS/Title
Case headings form a heading stack, every chunk keeps its heading path (which becomes
the citation's section and heads the evidence block the model reads), paragraphs built
from wrapped PDF lines are never cut mid-sentence, table rows stay whole with the
header row repeated onto every chunk, and `chunk_size` is a cap rather than a target.
It also fixes the old splitter's mid-word cuts (`Extr|action`, `HT|ML`).

It is **not the default**, on measurement. On dataset v2.0 it improved every retrieval
metric — precision .3115 → .3240, MRR .615 → .672, citation accuracy .414 → .492,
forbidden leaks .1875 → .125 — and cut answer correctness .8125 → .6458, so
`--compare` failed it. The cause is known: finer chunks mean more items per file, and
`mmr_max_per_source` (2) then keeps only two of them, so the one chunk holding the
answer loses its slot to its own siblings. The fix is a wider per-source fan-out or
small-to-big parent/child retrieval (`ContentItem.parent_id` already exists for it).
Until that is measured, `CHUNKING_MODE=fixed` ships and `structure` stays opt-in.
