# Evidence AI — 15-Minute Jury Demo Runbook

Everything below was executed and measured on the build in this zip. Numbers quoted
are real run outputs, not targets.

---

## 1. What your laptop needs (prerequisites)

| # | Requirement | Why | Install |
|---|---|---|---|
| 1 | **Windows 10/11 64-bit** | target platform | — |
| 2 | **Python 3.11 or 3.12 (64-bit)** | runtime | python.org → tick **"Add python.exe to PATH"** |
| 3 | **Ollama for Windows** | local LLM, offline answers | `winget install Ollama.Ollama` or ollama.com/download |
| 4 | **A local model** | the generator | `ollama pull qwen2.5:0.5b` (400 MB, runs on CPU) |
| 5 | **Tesseract OCR** | reads text out of images/scans | `winget install UB-Mannheim.TesseractOCR` |
| 6 | **8 GB RAM minimum** (16 GB comfortable) | embeddings + Whisper + LLM live in RAM | — |
| 7 | **~10 GB free disk** | models + Python packages | — |
| 8 | **Internet, first run only** | downloads models (~1.5 GB total) | after that it runs fully offline |

**GPU:** optional. An NVIDIA card with **as little as 4 GB VRAM** accelerates it; the
numbers in this runbook were produced **CPU-only**, so a laptop with no NVIDIA GPU
still demos fine (first question ~20 s cold, then ~8–12 s).

**No separate FFmpeg install needed** — audio decoding goes through PyAV, which ships
the libraries.

Models downloaded on first run: `all-MiniLM-L6-v2` (embeddings), `bge-reranker-base`
(reranker), `whisper` (audio transcription), `qwen2.5:0.5b` (generation).

---

## 2. Setup (pick ONE path)

### Path A — automated (recommended)
```powershell
cd EvidenceAI_Windows_Kit
copy .env.example .env
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
powershell -ExecutionPolicy Bypass -File start.ps1
```
`setup_windows.ps1` creates `.venv`, installs pinned dependencies, checks Ollama and
Tesseract. `start.ps1` verifies Ollama is serving, starts the API and opens the UI.

### Path B — manual (if PowerShell policy blocks scripts)
```powershell
cd EvidenceAI_Windows_Kit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
copy .env.example .env
python create_demo_data.py
python scripts\ingest_demo.py
python scripts\create_user.py --username demo --role admin
streamlit run app.py
```

### `.env` — the only line most people must edit
If Tesseract is not on PATH:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 3. First run: create the administrator

No default password ships with this project. On first launch the UI shows a
**"Create the first administrator"** form — fill it in, then log in.
(Equivalent on the command line: `python scripts\create_user.py --username demo --role admin`.)

Roles: **viewer** (ask only) · **analyst** (ask + ingest) · **admin** (everything:
ingest, delete sources, manage collections, export history, manage accounts).

---

## 4. Verify before you walk in (2 minutes)

```powershell
python scripts\verify_env.py     # 21 modules importable, models reachable
python scripts\smoke_test.py     # 4 end-to-end questions: text, audio, document, abstention
python evaluation.py             # full harness: writes evaluation\reports\*
```
Expected: `verify_env` all OK · `smoke_test` **4/4** · `evaluation.py` **12 PASS / 4 CHECK**
on the 16-query dataset v2.0 (report `report_20261105_145625`, index 6 sources / 37 items).

---

## 5. The demo itself (5–7 minutes, in this order)

**1. Log in as admin.** One line: *"Local accounts, PBKDF2-hashed, role-gated, with
lockout after five failures — nothing leaves the machine."*

**2. Text evidence.** Ask:
> How many records did the ingestion platform process in 2024?

Expected: **"1.2 million records"**, cited to `project_report.pdf — Page 2`.

**3. Image evidence (OCR).** Ask:
> What does the project dashboard show about team size?

Expected: **"6 engineers"** from `project_dashboard.png`, with an **OCR confidence
badge (≈90%)**. Point at the badge: low-confidence digits are *withheld*, not guessed.

**4. Audio evidence.** Ask:
> What did the quarterly review recording say about accuracy?

Expected: **"99.4%"** cited to `project_meeting.wav — 00:00-00:21`, with speaker
diarisation. Same corpus, three modalities, one index.

**5. The differentiator — abstention.** Ask:
> What was Project Alpha's budget in 2040?

Expected: **"I could not find sufficient evidence in the available sources…"** with no
citations. Say: *"It refuses rather than invents. This is measured, not a party trick —
abstention accuracy is 0.875 on the harness."*

**6. The numeric guard.** Ask:
> How many records are in the project metrics dataset?

Expected: an answer that carries an appended
**"Evidence warning: the figure 32,000 could not be found in the retrieved evidence…"**
— the model reached for a number the evidence does not contain, and the system said so
in the same breath. This is the strongest honesty demo in the product.

**7. Sources tab.** Show the six indexed sources, their **collections**, OCR badges and
admin-only delete. Mention: deleting rebuilds the vector index rather than leaving
dangling ids, and `--replace` on ingest swaps an updated file instead of keeping the
stale version citable beside the new one.

**8. History tab.** Replay an earlier question: it shows the **full retrieval trace**
(what was retrieved, reranked, cited, and how long each stage took), exportable to
Markdown/CSV. Auditable answers, not just answers.

**9. Close with the harness.**
```powershell
python evaluation.py --compare
```
*"Every change is gated: 16 fixed queries, deterministic metrics, and a regression gate
that fails the build if any metric drops. That is how we found and reverted our own
prompt change and our own chunking change — both are recorded in the guide."*

---

## 6. If something goes wrong live

| Symptom | Fix |
|---|---|
| UI won't answer, "provider error" | Ollama isn't running: `ollama serve` in another window, then retry |
| Answers say insufficient evidence for everything | Index is empty: `python scripts\ingest_demo.py` |
| Upload button does nothing / 403 | Restart with `streamlit run app.py --server.enableXsrfProtection false` |
| No OCR text from images | `TESSERACT_CMD` in `.env` is wrong; check the path, re-ingest |
| Audio source skipped | `pip install -r requirements-lock.txt` again — librosa/soundfile missing |
| Very slow first answer | Normal: models load lazily on first use (~20 s CPU), then stay warm |
| Out of memory | Close the API (`api.py`) while demoing the UI, or use a smaller model |

**Demo-safe fallback:** steps 2, 3, 4 and 5 are the core story and each takes one
question. If time collapses, do 5 (abstention) and 6 (numeric guard) — nothing else in
the market segment does those.

---

## 7. Say this if the jury asks about limitations

Being straight about these is worth more than overselling:

- **Model size.** At `qwen2.5:0.5b` the system does not *synthesise* two figures from
  two sources reliably (harness item Q5 stays a CHECK), and it is fooled by adversarial
  near-misses: a 2023 question answered with 2024's figure (Q9), "Project Beta"
  answered with Alpha's headcount (Q11), and Q3's 99.0 answered as Q4's 99.4 (Q12).
  All four are written into the dataset with their ground truth so anyone can re-check.
  A 3B+ local model is the fix; the architecture does not change.
- **LLM-as-judge** is implemented but reports itself **UNUSABLE at 0.5b**
  (`judge_consistency_rate` 0.0 — it disagrees with itself when the option order is
  swapped). Deterministic metrics remain the score of record.
- **Structure-aware chunking** (headings/paragraphs/table rows with heading paths) is
  implemented and opt-in via `CHUNKING_MODE=structure`. It measurably improved
  retrieval (precision .3115→.3240, MRR .615→.672, citation accuracy .414→.492) but
  reduced answer correctness (.8125→.6458) at this model size, so the regression gate
  failed it and the default stays `fixed`. That is the gate doing its job.
- **Current measured state** (this build, this index): **12 PASS / 4 CHECK** ·
  answer correctness 0.875 · groundedness 0.9375 · hallucination rate 0.0625 ·
  abstention accuracy 0.875 · retrieval recall 0.9635 · precision 0.3115 ·
  MRR 0.6146 · citation accuracy 0.401 · forbidden-leak rate 0.1875. The four
  CHECKs are Q5 (multi-figure synthesis), Q9 (near-miss year), Q11 (entity
  confusion) and Q12 (table near-miss) — all documented in the dataset with their
  ground truth.
