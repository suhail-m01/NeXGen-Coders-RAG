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

**Ask a question** — UI → *Ask* tab. Every answer lists citations with the exact page number (documents), timestamp range (audio) or OCR region (images). If evidence is too weak, it **abstains** instead of guessing, and says so.

**Add your own files** — three ways:
1. UI → *Upload* tab (drag and drop)
2. CLI: `.\.venv\Scripts\python scripts\ingest_demo.py --dir "D:\my documents"`
3. API: `POST http://127.0.0.1:8000/ingest`

Supported: PDF, DOCX, `.doc` (needs LibreOffice), CSV/XLSX, PNG/JPG (OCR + optional vision), WAV/MP3/M4A/FLAC/OGG (Whisper with timestamps). Limit: 200 MB per file (`MAX_FILE_BYTES` in `.env`).

Re-ingesting is safe — `source_id` is a content hash and existing item IDs are skipped, so nothing duplicates. `scripts\ingest_demo.py` goes further and skips an already-indexed file **before** extracting it, which matters because extraction is the expensive stage: without that, re-running would make Whisper re-transcribe every recording for nothing. Add `--force` when you genuinely want re-extraction (after changing an extractor or swapping a model).

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
| Antivirus/SmartScreen blocks a file | Downloaded-file mark | `Get-ChildItem -Recurse \| Unblock-File` (Step 6) |
| Setup crawls or files vanish | Folder inside OneDrive | Move the kit to `C:\EvidenceAI` and re-run |

---

## 7. What I changed vs. the original repository

Original files are **unmodified**. The repository's own `setup.ps1` cannot complete on a fresh clone, for two reasons I verified by running it:

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
| `scripts/verify_offline.py` | exit 0 — PASSED with **zero** outbound connections or DNS lookups |

**Not executed:** `setup_windows.ps1`, `setup.ps1` and `start.ps1`. There is no PowerShell interpreter in the sandbox, so the installer was reviewed line by line (balanced braces, `param()` first, functions defined before use, pinned torch resolution, native-command exit codes) but **has never been run**. Two real bugs were found and fixed during that review: `-match` against a multi-line array does not populate `$Matches` (which would have made the Python version test silently fail), and nested quotes in the `cmd /c` venv creation. Treat the first run of `setup_windows.ps1` as the actual test, and report any error text back — the fixes will be small.

Also **not** tested on Windows-specific paths: `pyttsx3`/SAPI demo-audio generation, the `winget` installs, GPU/VRAM detection, and the `qwen2.5vl:3b` vision path (the sandbox could not run that model at all).

---

## 8. Honest limitations

- **`confidence` is not a probability of being correct.** It is a squashed cross-encoder rerank logit — a relative ordering signal. Always read the cited passage.
- **The abstention gate is phrasing-sensitive.** Tuned by `ABSTENTION_THRESHOLD=-6.0`. A question with no supporting evidence is usually refused, but rewording can slip past it — test your own phrasings before trusting a demo.
- **Small models degrade output.** With `qwen2.5:1.5b` or `0.5b` expect weaker wording, and occasional garbled numbers. That is the substitute model, not the pipeline.
- **The image path depends on the model.** `qwen2.5vl:3b` can genuinely look at an image; text-only models fall back to OCR text, which is unreliable for charts and screenshots.
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
```

Ports: **8501** UI · **8000** API · **11434** Ollama.
