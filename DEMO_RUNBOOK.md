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

Models downloaded on first run: `sentence-transformers/all-MiniLM-L6-v2` (embeddings,
384-d), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker), `faster-whisper` base int8
(audio transcription), `qwen2.5:0.5b` (generation). Node is **not** required: the React
UI ships as a committed static build under `webui/dist`.

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
Tesseract. `start.ps1` verifies Ollama is serving (reading the model name from `.env`,
so the 0.5b demo model is accepted), starts the API **and** the Streamlit operator UI,
and opens <http://127.0.0.1:8000/ui/> in your browser — the React UI is served by the
API itself, so there is no second port and nothing to build. Add `-NoBrowser` to keep
the window closed, `-ApiOnly` for just the API and React UI.

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
python -m uvicorn api:app --host 127.0.0.1 --port 8000     # React UI at /ui
python -m streamlit run app.py                              # operator UI (history, accounts)
```

### `.env` — the only line most people must edit
If Tesseract is not on PATH:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 3. First run: create the administrator

No default password ships with this project. On first launch **both** UIs show a
**"Create the first administrator"** form — the React UI at
`http://127.0.0.1:8000/ui/` and the Streamlit UI at `:8501`. Fill it in once, then
log in. That form is a one-shot door: it closes (`409`) as soon as any account
exists, so it cannot be used later to add a second administrator.
(Equivalent on the command line: `python scripts\create_user.py --username demo --role admin`.)

Roles: **viewer** (ask only) · **analyst** (ask + ingest) · **admin** (everything:
ingest, delete sources, manage collections, export history, manage accounts).

---

## 4. Verify before you walk in (2 minutes)

```powershell
python scripts\selftest.py       # 86 offline checks: guards, parents, trace, manifest
python scripts\apitest.py        # 30 offline checks: the API and the React UI
python scripts\verify_env.py     # every import resolves, binaries reachable, audio decodes
python scripts\smoke_test.py     # 4 end-to-end questions + a real abstention
python scripts\verify_offline.py # blocks all non-loopback sockets, then answers
python evaluation.py --compare   # the regression gate against evaluation\baseline.json
```

Expected, all of it measured on this build (index 6 sources / 37 items / 43 relationships):

| Command | Result | Exit |
|---|---|---|
| `scripts\selftest.py` | **86/86 checks passed** — no LLM, no network, throwaway index | 0 |
| `scripts\apitest.py` | **30/30 checks** on the API and the React UI it serves — first-run bootstrap, login, roles, lockout, `/ui` assets, manifest. No server, no model | 0 |
| `scripts\verify_env.py` | `Environment verified - all required imports succeeded` (pyttsx3 optional) | 0 |
| `scripts\smoke_test.py` | **4/4** questions answered, abstention correct | 0 |
| `scripts\verify_offline.py` | `none - every socket stayed on loopback` → `Offline verification PASSED` | 0 |
| `evaluation.py --compare` | **16 PASS / 0 CHECK** on dataset v2.0, `PASS: no gated metric regressed` (report `report_20261105_200057`) | 0 |

`selftest.py` is the one to run in the room if a laptop has not finished downloading
models: it needs nothing but Python and the repository.

---

## 5. The demo itself (5-7 minutes, in this order)

Demo in the **React UI**: <http://127.0.0.1:8000/ui/>. `start.ps1` opens it. Keep the
Streamlit operator UI on :8501 for step 10.

**1. Log in as admin** (or create the first administrator, if this laptop has never run
it — the same form, and it closes after one use). The role badge next to your name is
read back from the server on every load, not stored in the browser. One line: *"Local accounts, PBKDF2-hashed,
role-gated, lockout after five failures — and a viewer never sees the delete button at
all, rather than seeing it fail."*

**2. Text evidence.** Ask:
> How many records did the ingestion platform process in 2024?

Expected: **"1.2 million records"**, cited to `project_report.pdf - Page 2`. Then
**click the [1] in the answer**: the evidence block scrolls into view and outlines
itself; click it again to release. Say: *"Interactive citations — the claim and the
sentence it came from are one click apart, in both directions."*

**3. Image evidence (OCR).** Ask:
> What does the project dashboard show about team size?

Expected: **"6 engineers"** from `project_dashboard.png`. The evidence panel shows
**OCR confidence ~90%**, words kept and the upscale factor. Point at it: low-confidence
digits are *withheld*, not guessed.

**4. Audio evidence.** Ask:
> What did the quarterly review recording say about accuracy?

Expected: **"99.4%"** cited to `project_meeting.wav - 00:00-00:21`, with speaker
diarisation. Same corpus, three modalities, one index.

**5. The differentiator — a refusal that explains itself.** Ask:
> What was Project Alpha's budget in 2040?

Expected: refused, no citations. Now open **Pipeline trace** and read the last two rows
out loud:

```
coverage guard: question numbers absent from evidence: 2040
End to end 439.9 ms  abstained (INSUFFICIENT_EVIDENCE)
```

Say: *"Four hundred milliseconds — the language model was never called. The trace lists
all fifteen stages, including the ones that did not run and why. Abstention accuracy is
1.0 on the harness, and this is the proof for one question."*

**6. The entity guard.** Ask:
> What was Project Beta's team size?

Expected: refused in **~0.7 s**, decisive check `entity guard: question asks about
Project Beta; evidence only mentions Alpha`, and the answer text itself says
*"The sources indexed here describe Alpha, so I will not answer for Project Beta."*
Say: *"It does not just refuse, it names the constraint that failed. Near-miss questions
are the ones that make RAG systems lie confidently."*

**7. Answer integrity, without promising a live failure.** In the same trace panel, point
at the guard rows: the numeric guard caveats an answer figure the evidence does not
contain, the repetition guard collapses a looping small model, and evidence completion
appends a figure **labelled as quoted from the evidence** rather than woven in. On the
gated run both `numeric_guard_rate` and `forbidden_leak_rate` are **0.0** — the guards
fire when they should and stay silent when they should not. Do not stage a question
hoping for a warning: whether the 0.5b model invents a number on a given run is not
yours to control, and a demo that promises a failure is a demo that can fail to fail.

**8. Sources panel.** Six indexed sources with type, collection and size; admin-only
delete; and the **stale-version banner** that appears when a filename has more than one
indexed version, newest marked, older ones removable in one click. Say: *"Re-uploading a
changed file used to leave the old version citable beside the new one, with nothing in
any UI to tell you. Now the system says so and offers the cleanup."* Deleting rebuilds
the vector index rather than leaving dangling ids.

**9. Health & build.** Show the runtime block (git commit, clean/dirty tree, CPU, RAM,
disk free, uptime, every guard flag) and expand the **full run manifest**. Say: *"Any
answer, and any evaluation score, traces back to the exact build that produced it —
commit, models, thresholds, dataset version. The same manifest is embedded in every
report."*

**10. History (Streamlit, :8501).** Replay an earlier question: it renders the **stored
pipeline trace** as recorded, plus what was retrieved, reranked and cited. Exportable to
Markdown/CSV. Auditable answers, not just answers.

**11. Close with the harness.**
```powershell
python evaluation.py --compare
```
*"Sixteen fixed queries, deterministic metrics, and a regression gate that fails the
build if any gated metric drops. That is how we found and reverted our own prompt change
and our own chunking change — both are recorded in the guide, with the numbers."*

---

## 6. If something goes wrong live

| Symptom | Fix |
|---|---|
| UI won't answer, "provider error" | Ollama isn't running: `ollama serve` in another window, then retry |
| Answers say insufficient evidence for everything | Index is empty: `python scripts\ingest_demo.py` |
| Upload button does nothing / 403 | Restart with `streamlit run app.py --server.enableXsrfProtection false` |
| No OCR text from images | `TESSERACT_CMD` in `.env` is wrong; check the path, re-ingest |
| Audio source skipped | `pip install -r requirements-lock.txt` again — librosa/soundfile missing |
| Very slow first answer | Normal: models load lazily on first use — measured 35 s cold on this corpus, then 10-12 s. The React UI says so while it waits |
| Out of memory | Close the API (`api.py`) while demoing the UI, or use a smaller model |
| React UI is blank or 404 | `webui\dist` is missing: `cd webui && npm install && npm run build` (Node 18+) |
| React UI says 401 after a while | The token was revoked or the account deleted: sign in again |
| `/login` answers 423 | Account locked after five failures: wait 5 minutes, or `scripts\create_user.py <name> <new-password> --set-password` |
| It refuses questions it should answer | Open the pipeline trace and read the **decisive check** row — it names the guard. Then check `ABSTENTION_THRESHOLD` / `_IMAGE` |

**Demo-safe fallback:** steps 2, 5 and 6 are the core story and each takes one question.
If time collapses, do 5 and 6 — a refusal in 0.4 s with the guard named in the trace is
the thing nobody else in this segment shows. Both are deterministic: they do not depend
on the model behaving on the day.

---

## 7. Say this if the jury asks about limitations

Being straight about these is worth more than overselling.

- **The model is small, and that shows.** At `qwen2.5:0.5b` the four adversarial items
  that used to fail — Q5 multi-figure synthesis, Q9 a 2023 question answered with 2024's
  figure, Q11 "Project Beta" answered with Alpha's headcount, Q12 a table near-miss
  (99.0 read as 99.4) — now pass, but **not because the model improved**. They pass
  because deterministic guards run before and after generation and compare numbers, years
  and names instead of trusting a 0.5b model to notice. Give the same model *more*
  context and it confabulates: in structure mode it produced a confident "87%" for a
  figure the evidence does not contain, and denied a limitation printed in the deck with
  confidence 0.004. A 3B+ local model is the fix; the architecture does not change.
- **The guards are deterministic, not semantic.** They catch a missing year, an absent
  entity, an unsupported number, a repeated span. A paraphrase that drops the year
  entirely can pass the temporal guard, and coverage checks that a question's figures
  appear in the evidence — not that the evidence means what the answer says. The honest
  claim is "an unsupported figure does not reach the user unnoticed", not "no wrong answer
  is possible".
- **LLM-as-judge** is implemented but reports itself **UNUSABLE at 0.5b**
  (`judge_consistency_rate` 0.0 — it disagrees with itself when the option order is
  swapped). Deterministic metrics remain the score of record.
- **Structure-aware chunking is opt-in and currently loses.** `CHUNKING_MODE=structure`
  cuts on headings, slides, table rows and transcript segments, and stores each block's
  enclosing section as a **parent**, so retrieval matches a small passage and generation
  reads the wider section (small-to-big). Retrieval genuinely improved — on the last full
  gated run: precision .3115 to .3635, MRR .6458 to .7031, citation accuracy .551 to
  .7323 — but the run scored **13 PASS / 3 CHECK** against 16/16 for `fixed`. A
  passage-level temporal fix then closed the Q9 leak (probe: it now abstains), while Q10
  still invented a figure and Q16 still denied the documented limitation, so structure
  mode stayed opt-in rather than being re-gated and shipped. Both runs are in
  `evaluation/reports/`. That is the gate doing its job, not a feature we hid.
- **`confidence` is not calibrated.** It is the best cross-encoder score normalised to
  0-1: an ordering signal, not a probability that the answer is right. The 0.004 above is
  why abstention is scored separately rather than read off a threshold on confidence.
- **Precision reads low and was left un-gamed.** 0.3115, because any retrieved file not
  in `expected_sources` counts as a miss and the demo corpus is deliberately overlapping —
  the CSV and the PDF report both genuinely answer the budget question. Recall (0.9635),
  MRR, groundedness and abstention are the meaningful signals here.
- **Single-machine security.** Accounts, token digests and chat history live in local
  SQLite and the service speaks plain HTTP. Passwords are salted PBKDF2, tokens are stored
  only as digests, and five failures lock an account for five minutes — but putting this
  on a network means putting TLS in front of it. The lockout slows guessing; it does not
  stop an attacker who has the database file.
- **Current measured state** (this build, this index, `qwen2.5:0.5b`, CPU-only):
  **16 PASS / 0 CHECK** · answer correctness 1.0 · groundedness 0.9375 · hallucination
  rate 0.0625 · abstention accuracy 1.0 · retrieval recall 0.9635 · precision 0.3115 ·
  MRR 0.6458 · citation accuracy 0.551 · numeric-guard rate 0.0 · forbidden-leak rate 0.0
  · mean latency 11.0 s. Report `report_20261105_200057`, which ships in this kit at
  `evaluation\reports\` so a claimed score can be opened instead of trusted; its
  embedded manifest records
  commit `67be890` with a dirty tree — the work is now committed as `a956ae7`, and the
  manifest is what makes that checkable instead of a claim.
