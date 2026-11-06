# Evidence AI — 15-Minute Jury Demo Runbook

Everything below was executed and measured on the build in this zip. Numbers quoted
are real run outputs, not targets.

**Printing one page?** `DEMO_CUE_CARD.md` is this runbook reduced to what you need in
your hand: the seven questions in order, what each one costs in seconds, the lines to
say, and what to do if the room goes quiet. It was measured against a copy of this
archive unpacked into a clean directory, not against the working tree.

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
python scripts\selftest.py       # 276 offline checks: guards, collections, tables, context
python scripts\apitest.py        # 78 offline checks: the API and the React UI
python scripts\apitest.py --with-query   # 88 checks, including the streaming route
python scripts\verify_env.py     # every import resolves, binaries reachable, audio decodes
python scripts\smoke_test.py     # 4 end-to-end questions + a real abstention
python scripts\verify_offline.py # blocks all non-loopback sockets, then answers
python evaluation.py --retrieval-only --compare   # fast gate, 33 s, no LLM needed
python evaluation.py --compare   # the full regression gate against evaluation\baseline.json
```

Expected, all of it measured on this build (index 6 sources / 36 items / 41 relationships):

| Command | Result | Exit |
|---|---|---|
| `scripts\selftest.py` | **276/276 checks passed** — no LLM, no network, throwaway index | 0 |
| `scripts\verify_kit.py` | **10/10 checks** on the archive you installed from — run this first, on any machine, before the environment exists | 0 |
| `scripts\apitest.py` | **81/81 checks** on the API and the React UI it serves — first-run bootstrap, login, roles, lockout, `/ui` assets, chat history, the 429 path, manifest, collections, and a final assertion that the accounts, the history and the index are exactly as the suite found them. No server, no model | 0 |
| `scripts\verify_env.py` | `Environment verified - all required imports succeeded` (pyttsx3 optional) | 0 |
| `scripts\smoke_test.py` | **4/4** questions answered, abstention correct | 0 |
| `scripts\dump_chunks.py --all` | 6 indexed sources with their chunk counts — the fastest way to confirm the index is the corpus you think it is | 0 |
| `scripts\verify_offline.py` | `none - every socket stayed on loopback` → `Offline verification PASSED` | 0 |
| `evaluation.py --retrieval-only --compare` | **15 PASS / 0 CHECK / 1 UNDECIDED** in 33 s, retrieval metrics identical to the full run. The undecided row is Q9, whose refusal comes from the post-generation temporal guard — reported, not guessed | 0 |
| Fresh install from this archive | unpacked into a clean directory, `.env` copied from `.env.example`, index built by `ingest_demo.py` in **28 s**: `verify_kit.py` **12/12** with `--extract`, `selftest.py` **276/276**, `apitest.py` **81/81 on its first run against that index**, `smoke_test.py` **4/4**, `verify_offline.py` PASSED, and `evaluation.py --compare` **16 PASS / 0 gated failures** with **all ten gated metrics identical** to `report_20261106_000901` (`report_20261106_011329`). Latency differs and should: 10.69 s mean with the model resident, against 15.77 s in the record where `OLLAMA_KEEP_ALIVE=0` unloaded it between questions |
| `evaluation.py --compare` after round 3 | exit 0 — **16 PASS / 0 CHECK**, 0 gated failures, clean tree at `09caf60`, and **every gated metric identical** to `report_20261106_000901`. The streaming route changed when tokens arrive, not what the answer is (`report_20261106_014909`) |
| `evaluation.py --compare` after the attachments feature | exit 0 — **16 PASS / 0 CHECK**, 0 gated failures, clean tree at `a798d97`: attachment evidence joins the shared rerank → guards → synthesis path, and **every gated metric is identical** to `report_20261106_000901` with zero per-query drift (`report_20261106_032714`) |
| `evaluation.py --compare` after the attachment routing fix | exit 0 — **16 PASS / 0 CHECK**, 0 gated failures, clean tree at `e781e49`: “explain this document” and its kin are answered from the attachment alone, factual questions keep the shared corpus path unchanged, and **every gated metric is identical** to `report_20261106_000901` with zero per-query drift — mean latency 16.24 s against the record's 15.77 s is the usual `OLLAMA_KEEP_ALIVE=0` jitter (`report_20261106_043956`) |
| `evaluation.py --compare` | **16 PASS / 0 CHECK** on dataset v2.0, `PASS: no gated metric regressed`, retrieval precision 0.3531, citation accuracy 0.5635, no per-query regression to report (report `report_20261106_000901`, commit `1fac077`, clean tree) | 0 |

`selftest.py` is the one to run in the room if a laptop has not finished downloading
models: it needs nothing but Python and the repository. `evaluation.py
--retrieval-only` is the next cheapest real check — it needs the index but no LLM,
so it still works if Ollama has not started, and it takes half a minute.

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

Expected: **"1.2 million records"**, cited to `project_report.pdf - Page 2` among five
citations. Then **click a citation**: the evidence block scrolls into view and outlines
itself; click it again to release.

Where to click depends on the model. If the answer prose carries inline `[1]` markers,
those are buttons. A small model often emits none — `qwen2.5:0.5b` did not in rehearsal —
and the **numbered chips directly under the answer are there either way**, as are the
chips on each evidence block. All three drive the same selection. Do not promise the
audience an inline marker before you have looked at the answer.

Say: *"Interactive citations — the claim and the sentence it came from are one click
apart, in both directions."*

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

**8. Sources and collections panel.** Start with the **Add documents** card and drop a
file on it. The queue gives every file its own row and its own outcome — items
indexed, warnings listed, or the reason it failed — because a batch upload that
reports "finished" does not report which PDF was password protected. Drag something
unacceptable in (a `.zip`) to show that a rejected file becomes a named row with a
reason rather than quietly disappearing. The formats that card accepts are read from
`GET /manifest`, not written into the page, so the drop zone cannot promise what the
server would refuse. Press **?** for the shortcuts: Alt+1..4 moves between tabs,
Ctrl+K jumps to the question box — worth knowing live, because it means never
hunting for a tab in front of an audience. Then the **Collections** card: declare a
collection, file a source under it with the dropdown in the source table, then ask a
question scoped to it on the Ask tab — and say what deleting a collection does, because
everybody assumes the wrong thing: *"Deleting this group moves its two documents back to
`general`. It deletes nothing, and it does not re-index anything, because a collection is
a label on a source row."* Then the source table: six indexed sources with type,
collection and size; admin-only delete; and the **stale-version banner** that appears when a filename has more than one
indexed version, newest marked, older ones removable in one click. Say: *"Re-uploading a
changed file used to leave the old version citable beside the new one, with nothing in
any UI to tell you. Now the system says so and offers the cleanup."* Deleting rebuilds
the vector index rather than leaving dangling ids.

Last thing on this panel, and it is a good one — what the indexer actually made of a
PDF:

```powershell
python scripts\dump_chunks.py --source MCA_MP_58-mini-project.pdf
```

Two of the deck's chunks are markdown grids, lifted off pages 4 and 5. Say: *"Plain
text extraction reads a table in block order, so one row's cells end up in different
chunks — a paper's name beside a diagram caption, its finding three chunks away. We
detect the ruled region and index it as a grid instead, so a row stays a row. On our
own harness that moved the deck question from retrieving three items for one answer
to retrieving exactly the right one."* If somebody asks what happens when the detector
gets it wrong: *"A region has to reach two rows, two columns and three filled cells
before it is treated as a table, and every word it claimed goes back to the page text
if it is then rejected. The self-test checks that word by word against this deck, so
nothing is quietly withheld from the index."*

**9. Health & build.** Show the runtime block (git commit, clean/dirty tree, CPU, RAM,
disk free, uptime, every guard flag) and expand the **full run manifest**. Say: *"Any
answer, and any evaluation score, traces back to the exact build that produced it —
commit, models, thresholds, dataset version. The same manifest is embedded in every
report."*

**10. History — the audit trail, in the same browser tab.** Open the **History** tab in
the React UI. Every conversation you ran in steps 4-8 is listed; open one and each answer
renders with the **stored pipeline trace** as it was recorded, the citations, and any
figure the numeric guard could not verify. Say: *"This was asked twenty minutes ago and
nothing about it has been re-computed — the trace you are looking at is the one the
answer was produced with. If the corpus changes next week, this record does not."* With
an admin account, export it to Markdown or CSV and hand the file to the jury: a reviewer
who will never sign in can still read the provenance. The Streamlit operator UI at :8501
replays exactly the same records, because both front ends write through one API and one
database. Auditable answers, not just answers.

**11. Close with the harness.**
```powershell
python evaluation.py --compare
```
*"Sixteen fixed queries, deterministic metrics, and a regression gate that fails the
build if any gated metric drops. That is how we found and reverted our own prompt change
and our own chunking change — both are recorded in the guide, with the numbers."*

If somebody asks whether the current retrieval techniques were tried, this is the answer,
and it is a stronger one than a feature list: *"Contextual retrieval — a per-chunk blurb
from the model, prepended to what gets embedded. We implemented it and measured it against
the same sixteen queries with the flag off and on. Retrieval precision went .353 to .382
and groundedness to 1.0, but answer correctness went 1.0 to .979, because one query
answered both of its hops in better prose and dropped the figure the dataset requires. It
ships off, both runs are in the repository, and the blurb never touches a citation — it is
prepended to the indexed text only."* The two reports are `report_20261105_215324` (off)
and `report_20261105_223230` (on) in `evaluation\reports\`.

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
| A request answers **429** | The rate limiter: 30 questions/minute with a burst of 6, 60 ingestions/minute with a burst of 20 (`RATE_LIMIT_*` in `.env`). Wait the seconds named in the `Retry-After` header. A presenter clicking through demo questions does not hit this; a retry loop or a folder of 50 PDFs dropped twice does |
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
