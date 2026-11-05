# Evidence AI

**Offline-runtime multimodal Retrieval-Augmented Generation over documents, images and audio.**

Ask one question and get a grounded answer assembled from PDFs, Word documents,
spreadsheets, screenshots and voice recordings at the same time — with a citation
back to the exact page, timestamp or image behind every claim, and an explicit
refusal when the evidence isn't there.

Every model runs locally. After a one-time online preparation step, the
application needs no internet at all.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Data flow](#data-flow)
- [Technology stack](#technology-stack)
- [Hardware requirements](#hardware-requirements)
- [Setup](#setup)
- [Running it](#running-it)
- [Ingestion](#ingestion)
- [Querying — four modes](#querying--four-modes)
- [Citations](#citations)
- [Relationships](#relationships)
- [Abstention](#abstention)
- [Evidence guards](#evidence-guards)
- [API](#api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Observability](#observability)
- [Offline operation](#offline-operation)
- [Persistence](#persistence)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What it does

| Capability | How |
| --- | --- |
| Multimodal ingestion | PDF, DOC, DOCX, TXT, CSV, PNG/JPG/WEBP, WAV/MP3/M4A |
| Local OCR | Tesseract 5 |
| Local image understanding | Qwen2.5-VL:3B through Ollama — real image input |
| Local speech-to-text | faster-whisper, timestamped segments |
| Semantic retrieval | all-MiniLM-L6-v2 + FAISS (exact cosine) |
| Hybrid retrieval | dense + BM25 candidates fused by reciprocal rank fusion |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2, length-normalised |
| Diversity | MMR re-selection with a per-source cap, so one file cannot fill the context |
| Relationship-aware retrieval | deterministic links + query-time semantic links |
| Grounded generation | Qwen2.5-VL:3B, evidence-only prompt |
| Citation transparency | built in code, never emitted by the model |
| Abstention | refuses before the LLM is ever called, per modality |
| Evidence guards | coverage, entity, temporal, period, numeric and repetition checks — each can refuse, annotate or reorder |
| Answer completion | a figure stated elsewhere in the retrieved evidence is appended and labelled, never invented |
| Speaker diarization | spectral clustering over the transcript, so audio evidence can carry more than one voice |
| Chat history | per-session records with the full pipeline trace, replayable and exportable |
| Collections | logical groupings; a query can be scoped to one |
| Accounts and roles | local users, three roles with distinct permissions, lockout, API bearer tokens |
| Reproducible evaluation | fixed 16-query dataset, versioned reports, regression gate against a baseline |
| Observability | per-query stage trace and a run manifest naming the exact build |
| React UI | built to static files and served by the API at `/ui` |
| Offline runtime | verified with an outbound-socket guard |
| Offline self-test | 86 checks over the guards, store and trace — no LLM, no network |

---

## Architecture

```
                          EVIDENCE AI
                               |
              +----------------+----------------+
              |                                 |
          INGESTION                          QUERY
              |                                 |
     PDF DOC DOCX TXT CSV              text / image / audio / document
     images   audio                              |
              |                                  v
              v                        query representation
     documents.py  images.py  audio.py           |
              |                                  v
              +--------> ContentItem[] <---------+
                               |
                        all-MiniLM-L6-v2 (CPU)
                               |
                    +----------+----------+
                    |                     |
                  FAISS                SQLite
             (384-d cosine)     (items, sources, links)
                    |                     |
                    +----------+----------+
                               |
                          retrieval (top_k=10)
                               |
                    cross-encoder rerank (top 5)
                               |
                       ABSTENTION GATE
                               |
                  relationship expansion (capped)
                               |
                       final evidence set
                               |
                    Qwen2.5-VL:3B (Ollama, local)
                               |
              +----------------+----------------+
              |                                 |
     programmatic citations              relationships
              |                                 |
              +----------------+----------------+
                               |
                         QueryResponse
                               |
                    +----------+----------+
                    |                     |
             Streamlit UI            FastAPI ---- React UI (/ui)
```

**One core, three surfaces.** `app.py` and `api.py` both call `rag_pipeline.py`
and nothing else, so the UIs and the API cannot drift apart. The React app in
`webui/` is built to static files that `api.py` itself serves at `/ui`, so it
calls relative URLs: one port, one origin, nothing to configure, and it works
offline like everything else here.

---

## Data flow

Every modality is reduced to a **text representation**, and that representation
is what gets embedded. This is called **representation-based multimodal
retrieval**.

```
PDF / DOCX / TXT / CSV  ->  extracted, chunked text        ->  MiniLM  ->  FAISS
IMAGE                   ->  Qwen description + OCR text    ->  MiniLM  ->  FAISS
AUDIO                   ->  timestamped transcript window  ->  MiniLM  ->  FAISS
```

> **Being precise about this:** MiniLM is a *text* encoder. It never sees pixels
> or waveforms. An image is searchable because Qwen describes it and Tesseract
> reads it; audio is searchable because Whisper transcribes it. This is **not** a
> native joint image-text embedding space, and the project does not claim to be
> one. The original image and audio files are preserved and remain reachable for
> display, playback, and direct visual inspection at generation time.

---

## Technology stack

| Layer | Component | Runs on |
| --- | --- | --- |
| LLM + vision | Ollama `qwen2.5vl:3b` | GPU (partial) + CPU |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU |
| Speech-to-text | `faster-whisper` base, int8 | CPU |
| OCR | Tesseract 5.5 | CPU |
| Vector index | FAISS `IndexFlatIP` | CPU |
| Lexical index | BM25 over the same items, fused with dense by RRF | CPU |
| Metadata | SQLite | — |
| PDF | PyMuPDF | — |
| DOCX | python-docx | — |
| DOC | LibreOffice headless → DOCX | — |
| CSV | pandas | — |
| API | FastAPI + Uvicorn | — |
| UI | React 18 + Vite, served as static files at `/ui` | browser |
| UI (operator) | Streamlit — accounts, history, ingestion, trace | browser |

**Why PyTorch is pinned to CPU:** on a 4 GB GPU, Ollama already uses ~2.8 GB for
Qwen. Loading MiniLM, the cross-encoder and Whisper onto CUDA would evict Qwen
layers and make every query dramatically slower. Embedding a chunk on this CPU
takes single-digit milliseconds, so the GPU is reserved entirely for the LLM.

---

## Hardware requirements

Verified on the target machine:

| | |
| --- | --- |
| CPU | Intel i5-9300H, 4 cores / 8 threads |
| RAM | 16 GB (**~6 GB free is enough**) |
| GPU | GTX 1650, 4 GB VRAM |
| Disk | ~8 GB for models and dependencies |
| OS | Windows 11 |
| Python | 3.13.7 |

**Measured performance on this hardware:**

| Operation | Time |
| --- | --- |
| Text query, warm | 6–10 s |
| Query with direct image inspection | 12–25 s |
| First query after idle (model reload) | +14 s |
| Image ingestion (OCR + description) | ~7 s |
| Audio transcription | ~1–3× realtime |
| Full 5-file demo corpus ingestion | ~34 s |

`ollama ps` reports the model split **57% CPU / 43% GPU** — a 3B vision model
does not fully fit in 4 GB, so generation runs at 8–12 tokens/second. This is
expected, not a misconfiguration.

> **Run ingestion and querying sequentially.** With ~6 GB of free RAM, a Whisper
> transcription running at the same time as a Qwen generation will thrash.

---

## Setup

### One command

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates the venv, installs CPU PyTorch and all dependencies, verifies every
import, checks Tesseract / LibreOffice / Ollama, pulls `qwen2.5vl:3b`, downloads
and caches MiniLM + CrossEncoder + Whisper, and generates the demo data.

### Manual steps

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\verify_env.py
```

### External tools

```powershell
winget install UB-Mannheim.TesseractOCR                # OCR (required for images)
winget install Gyan.FFmpeg                             # broader audio format support
winget install TheDocumentFoundation.LibreOffice       # ONLY for legacy .doc
```

Ollama comes from <https://ollama.com/download>, then:

```powershell
ollama pull qwen2.5vl:3b
```

### Cache models locally

```powershell
.venv\Scripts\python.exe scripts\prepare_models.py
```

Models are cached at:

| Model | Location |
| --- | --- |
| `qwen2.5vl:3b` | `C:\Users\<you>\.ollama\models` |
| MiniLM, CrossEncoder, Whisper | `C:\Users\<you>\.cache\huggingface\hub` |
| Tesseract | `C:\Program Files\Tesseract-OCR` |

> On Windows without Developer Mode, HuggingFace cannot create symlinks
> (`WinError 1314`). `config.py` sets `HF_HUB_DISABLE_SYMLINKS=1` before any
> HuggingFace import, so this is handled automatically — no admin rights needed.

**After this succeeds you can disconnect from the internet.**

---

## Running it

```powershell
.\start.ps1                 # API on :8000 and UI on :8501
.\start.ps1 -ApiOnly        # API only
.\start.ps1 -UiOnly         # UI only
```

`start.ps1` starts Ollama if it isn't running, verifies the model is present,
warns about a missing index, and then launches.

Manually:

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --port 8000
.venv\Scripts\python.exe -m streamlit run app.py
```

The React UI needs no separate command: with the API running, open
<http://127.0.0.1:8000/ui/>. `/` redirects there. The built bundle is committed
under `webui/dist`, so Node is not required to run it — only to change it
(`cd webui && npm install && npm run build`).

---

## Ingestion

Generate the demo corpus:

```powershell
.venv\Scripts\python.exe create_demo_data.py
```

Ingest everything under `data/`:

```powershell
.venv\Scripts\python.exe scripts\ingest_demo.py
```

Or use the **Sources** panel in the React UI, the **Ingest** tab in Streamlit, or
`POST /ingest`. Re-ingesting a file that changed replaces the old version when
`replace=true` is passed; without it both versions stay searchable, and
`GET /sources/stale` (and the React Sources panel) says which files are in that
state. `scripts\reindex.py` rebuilds the whole index from `data/` in one pass.

| Format | Extracted as | Location preserved |
| --- | --- | --- |
| PDF | text per page, OCR fallback for scanned pages | `page_number` |
| DOCX | paragraphs + tables, grouped by heading | `section` |
| DOC | LibreOffice → DOCX → as above | `section` |
| TXT | chunked text | — |
| CSV | one `column: value` sentence per row | `row_index` |
| Images | Qwen description + Tesseract OCR | `image_path` |
| Audio | windowed timestamped transcript | `timestamp_start/end` |

Chunking defaults to 800 characters with 100 overlap, breaking at sentence
boundaries where possible (`CHUNKING_MODE=fixed`).

`CHUNKING_MODE=structure` instead cuts on document structure — headings, slides,
table rows, transcript segments — and stores each block's enclosing section as a
**parent**, so retrieval matches a small passage and generation reads the wider
section around it (small-to-big). It is implemented and measured, but it is
**opt-in**: on this corpus and this model it scored 13/16 against 16/16 for
`fixed`, because richer context gave a 0.5B model more room to confabulate. The
gate report for both runs is in `evaluation/reports/`. Switching modes needs a
reindex — `scripts\reindex.py`.

**`source_id` is a content hash.** Re-ingesting the same bytes produces the same
IDs and adds no duplicate vectors, which is what makes evaluation reproducible.

---

## Querying — four modes

### Text → anything

> *"What evidence do we have about Project Alpha's 2024 development?"*

Returns evidence from PDF, image and audio in one ranked list.

### Image → anything

Upload a screenshot. It is OCR'd and described by Qwen, and that representation
becomes the query.

> *"Find documents and audio recordings related to this screenshot."*

### Audio → anything

Upload a recording. Whisper transcribes it and the transcript becomes the query.

> *"What information in the reports corresponds to this recording?"*

### Document → anything

Upload a PDF or DOCX, then either ask about it or find related material.

> *"What were the main findings in this document?"*
> *"Find screenshots and recordings related to this document."*

> **Self-exclusion:** when you query *with* a file that is already indexed, its
> own chunks are excluded from the results. Otherwise a document's nearest
> neighbours are trivially itself, and "find related material" returns nothing
> useful.

**Discovery vs. factual questions** are detected and prompted differently. A
factual question must be answerable from evidence or it abstains; a discovery
request ("find things related to this") is answered by describing the retrieved
evidence, because there the retrieved set *is* the answer.

---

## Citations

Citations are constructed in `citations.py` from the final evidence set. **The
model never writes citation markers** — the prompt forbids it. A fabricated
citation is structurally impossible: every `Citation` is built from a real
`ContentItem` that was actually retrieved.

Each answer sentence is embedded and matched against the evidence, so citations
reflect what actually supports the text.

```
[1] project_report.pdf - Page 2
[2] project_notes.docx - Q4 Review Meeting
[3] project_dashboard.png
[4] project_meeting.wav - 00:06-00:19
[5] project_metrics.csv - Row 4
```

DOCX cites a **section, never a page**: `python-docx` cannot know page
boundaries, because pagination is computed by the renderer and not stored in the
file. Inventing one would be fabrication.

---

## Relationships

**Deterministic — computed at ingestion, stored in SQLite:**

| Type | Rule |
| --- | --- |
| `SAME_SOURCE` | items from the same file |
| `SAME_PAGE` | items on the same PDF page |
| `TRANSCRIPT_OF` | adjacent audio windows |
| `OCR_OF` | text derived from an image |

**Semantic — computed at query time, never persisted:**

| Type | Rule |
| --- | --- |
| `RELATED_TO` | cosine ≥ `RELATED_TO_THRESHOLD` among *retrieved* evidence only |

`RELATED_TO` compares only the handful of already-retrieved items — a few dozen
pairs, never a scan of the database. Same-source pairs are skipped because
`SAME_SOURCE` already covers them, which leaves the genuinely interesting
cross-modal links.

Expanded evidence is ranked strictly below the weakest direct hit, so a related
item can never outrank real evidence.

---

## Abstention

The abstention gate sits **before the LLM call**. When the best rerank score
falls below `ABSTENTION_THRESHOLD`, the model is never asked, so it cannot
improvise:

```
I could not find sufficient evidence in the available sources to answer this question.
```

with `abstained = true` and `error_code = INSUFFICIENT_EVIDENCE`. The retrieved
evidence is still returned so you can see *why* it abstained. A second check
catches the model emitting `INSUFFICIENT_EVIDENCE` itself.

The threshold is **per modality**. OCR-derived image evidence scores about six
cross-encoder logits below prose on the same corpus, so when the top-ranked item
is an image the gate uses `ABSTENTION_THRESHOLD_IMAGE` (-9.0) and otherwise
`ABSTENTION_THRESHOLD` (-6.0). One global bar would either refuse valid image
evidence or accept text-scale noise. `scripts\calibrate_abstention.py` measures
the split on your own corpus instead of guessing it.

---

## Evidence guards

Abstention decides *whether* there is evidence. The guards decide whether the
evidence answers **this** question, and whether the answer stayed inside it. All
are deterministic — string, number and date comparisons, never a model judging
itself — and each is one config flag away.

| Guard | Runs | What it does | Flag |
| --- | --- | --- | --- |
| Coverage | before generation | refuses when a number, year or unit in the question appears nowhere in the evidence ("budget in 2040") | `COVERAGE_GUARD_ENABLED` |
| Entity | before generation | refuses when the question asks about an entity the evidence never mentions, and the evidence mentions a similar one (Project **Beta** vs Project **Alpha**) | `ENTITY_GUARD_ENABLED` |
| Temporal | after generation | judges each retrieved passage: a figure stated for 2023 does not support an answer about 2024 | `TEMPORAL_GUARD_ENABLED` |
| Period targeting | before generation | reorders evidence so items mentioning the question's period come first. Reorders only — it never rescores, because rescoring would silently re-calibrate abstention | `PERIOD_TARGETING_ENABLED` |
| Numeric | after generation | compares every number in the answer against the evidence; an unsupported one appends a caveat and is listed in the trace | `NUMERIC_GUARD_ENABLED` |
| Repetition | after generation | collapses a small model's looping output; markdown ordinals and locator digits are excluded so a list is not mistaken for a loop | `REPETITION_GUARD_ENABLED` |
| Evidence completion | after generation | when the answer mentions a metric whose value sits in the evidence but not in the answer, appends it **labelled as quoted from the evidence** | `SYNTHESIS_COMPLETION_ENABLED` |

A guard that fires is visible: the caveat or correction is rendered as its own
block in both UIs, the stage trace names the decisive check, and the evaluation
scores `numeric_guard_rate` and `forbidden_leak_rate` so a guard that stops
working shows up as a regression rather than as a quieter answer.

---

## API

| Endpoint | Purpose | Needs |
| --- | --- | --- |
| `GET /` | redirects to the React UI | — |
| `GET /ui/` | the React front end (static build) | — |
| `GET /bootstrap` | does this service still have no administrator? | — |
| `POST /bootstrap` | create the first administrator; closes once one exists | — |
| `POST /login` | username + password → bearer token | — |
| `GET /me` | the token's account, role and permissions | `ask` |
| `POST /logout` | revokes the presented token | — |
| `GET /health` | backend readiness, dependencies, index, host headroom, build | — |
| `GET /manifest` | the exact build behind a result | `ask` |
| `POST /ingest` | upload and index files (`replace=true` retires older versions) | `ingest` |
| `POST /query` | text query; response carries the stage trace | `ask` |
| `POST /query_file` | image / audio / document query | `ask` |
| `GET /sources` | list indexed sources and collections | `ask` |
| `GET /sources/stale` | filenames with more than one indexed version | `ask` |
| `GET /source/{id}` | source detail and its items | `ask` |
| `GET /source/{id}/file` | original file (constrained to `data/`) | `ask` |
| `DELETE /source/{id}` | remove a source and rebuild the vector index | `delete` |

Interactive docs at `http://127.0.0.1:8000/docs`.

**First run.** A fresh install has no accounts, so nobody can sign in and nobody
can be granted the permission that creates one. `GET /bootstrap` reports that
state and `POST /bootstrap` creates the first **administrator** — the only
unauthenticated endpoint that can create an account, and it answers `409` from
then on. Both UIs offer the form; after that, accounts come from someone holding
`manage_users` or from `scripts\create_user.py`.

**Authentication.** Every endpoint except `/health`, `/bootstrap`, `/login` and
`/logout` needs
`Authorization: Bearer eai_...`. Tokens belong to a local account with one of
three roles — `viewer` (ask), `analyst` (ask, ingest), `admin` (also delete,
manage collections, export history, manage users) — and are minted by
`scripts\create_user.py <name> --create-token`, or by `POST /login` for a
browser. Only the token's digest is stored; five failed password attempts lock an
account for five minutes, and a locked account is told so (HTTP 423) rather than
being invited to try again. Set `API_AUTH_ENABLED=false` for a single-user
machine and every endpoint answers anonymously.

```powershell
# Use curl.exe, not curl: in PowerShell `curl` is an alias for
# Invoke-WebRequest, which does not accept these flags. Single quotes keep the
# JSON intact, so no escaped quotes are needed.
$body = '{"query":"What was budget utilisation in 2024?","query_modality":"text"}'
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer eai_..." `
  -d $body
```

---

## Configuration

Copy `.env.example` to `.env`. Key values:

| Setting | Default | Why |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | |
| `OLLAMA_NUM_CTX` | `8192` | one image costs ~1030 tokens; 4096 is too small |
| `OLLAMA_KEEP_ALIVE` | `30m` | avoids a measured 14 s reload mid-demo |
| `OLLAMA_TEMPERATURE` | `0.0` | minimises generation variance |
| `TOP_K` | `10` | FAISS candidates |
| `RERANK_TOP_K` | `5` | after cross-encoder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | |
| `RELATED_TO_THRESHOLD` | `0.70` | |
| `CITATION_THRESHOLD` | `0.35` | sentence→evidence attribution |
| `ABSTENTION_THRESHOLD` | `-6.0` | cross-encoder logit floor |
| `ABSTENTION_THRESHOLD_IMAGE` | `-9.0` | OCR evidence scores ~6 logits lower |
| `OLLAMA_NUM_PREDICT` | `512` | output cap: without it a small model can loop until the request times out |
| `HYBRID_ENABLED` | `true` | BM25 + dense, fused by RRF |
| `MMR_ENABLED` / `MMR_MAX_PER_SOURCE` | `true` / `2` | diversity, and how much of the context one file may take |
| `CHUNKING_MODE` | `fixed` | `structure` cuts on headings/rows/segments and stores parent sections |
| `PARENT_EXPANSION_ENABLED` / `PARENT_MAX_CHARS` | `true` / `1800` | small-to-big: match a passage, generate from its section |
| `STRUCTURE_MIN_CHUNK_CHARS` | `200` | floor below which structure fragments are merged |
| `COVERAGE_GUARD_ENABLED` | `true` | refuse when the question's numbers are absent from the evidence |
| `ENTITY_GUARD_ENABLED` | `true` | refuse when the asked-about entity is absent |
| `TEMPORAL_GUARD_ENABLED` | `true` | a 2023 figure does not support a 2024 answer |
| `PERIOD_TARGETING_ENABLED` | `true` | show period-matching evidence first (reorders, never rescores) |
| `NUMERIC_GUARD_ENABLED` | `true` | caveat an answer number the evidence does not contain |
| `REPETITION_GUARD_ENABLED` | `true` | collapse looping generation |
| `SYNTHESIS_COMPLETION_ENABLED` | `true` | append a labelled figure the evidence states but the answer omitted |
| `DIARIZATION_ENABLED` / `DIARIZATION_MAX_SPEAKERS` | `true` / `6` | speaker clustering over transcripts |
| `STREAMING_ENABLED` | `true` | token streaming (`query_stream`) |
| `API_AUTH_ENABLED` | `true` | bearer tokens and roles |

`.env.example` documents every one of these with the reason it exists.
| `MAX_CONTEXT_ITEMS` | `5` | |
| `MAX_INSPECT_IMAGES` | `1` | context budget |
| `TORCH_DEVICE` | `cpu` | GPU reserved for Ollama |

---

## Evaluation

```powershell
.venv\Scripts\python.exe evaluation.py
```

Fixed dataset at `evaluation/dataset/queries.json` (**version 2.0**, 16 queries):
per-modality, multi-hop, an unanswerable question, a temporal trap, a
forbidden-leak trap and a paper-limitation question. Each query carries expected
sources, expected answers, `must_not_contain` strings and tags.

```powershell
.venv\Scripts\python.exe evaluation.py                # run and report
.venv\Scripts\python.exe evaluation.py --compare      # gate against baseline.json
```

`--compare` is the regression gate: it diffs every gated metric against
`evaluation/baseline.json`, per modality and per query, and exits 1 on any
regression (2 when it cannot gate at all, for example a dataset-version mismatch,
which it refuses to compare silently). Nothing may land unless it exits 0.

Reports land in `evaluation/reports/` as JSON + CSV + Markdown, each stamped with
the dataset version, all four model names, the full configuration, the git commit,
a timestamp and the **run manifest** — so any score can be traced back to the
build that produced it.

**Measured results** — dataset v2.0, demo corpus, `qwen2.5:0.5b`, gate run
`report_20261105_200057` (**16 PASS / 0 CHECK**, `--compare` exit 0):

| Metric | Baseline | Now |
| --- | --- | --- |
| Retrieval precision | 0.3115 | 0.3115 |
| Retrieval recall | 0.9635 | 0.9635 |
| MRR | 0.6146 | **0.6458** |
| Citation accuracy | 0.4135 | **0.5510** |
| Answer correctness | 0.8125 | **1.0000** |
| Groundedness | 0.9375 | 0.9375 |
| Hallucination rate | 0.0625 | 0.0625 |
| Abstention accuracy | 0.8750 | **1.0000** |
| Numeric-guard rate | — | 0.0000 |
| Forbidden-leak rate | 0.1875 | **0.0000** |
| Mean latency | 6.58 s | 11.01 s |

Latency roughly doubled because the guards and the completion pass added work and
the output cap lengthened some generations; every gated quality metric moved up or
held. One per-query change is reported rather than gated: Q16's citation accuracy
went 1.0 → 0.8, because the answer now quotes two sources where the dataset
expects one.

> **Reading precision honestly.** Precision looks low because the metric counts
> any retrieved file not in `expected_sources` as a miss — but the demo corpus is
> *deliberately* overlapping, so retrieving `project_metrics.csv` alongside
> `project_report.pdf` is genuinely relevant and still scores as a false
> positive. Recall, MRR, groundedness and abstention are the meaningful signals
> here. The metric was left un-gamed rather than tuned to flatter the system.

**Determinism.** Retrieval, reranking, relationships and citations are exactly
reproducible. Generation is not: Qwen is sampled and Ollama gives no
cross-version seed guarantee, even with `OLLAMA_SEED=42` and temperature 0.0 —
measured here as three identical calls producing answers of 385, 3114 and 3114
characters. Answer-level metrics are therefore *approximately* reproducible, and a
suspect query is re-run two or three times before a conclusion is drawn from it. Groundedness is scored by keyword and
number overlap, never by an LLM judge — a 3B model grading its own output would
be neither reliable nor honest.

---

## Observability

Two artefacts answer the questions an evidence system actually gets asked.

**The stage trace** — every `POST /query` response carries `stages`, and both UIs
render it. Fifteen ordered stages, each with its milliseconds, what it moved (10
candidates → 5 selected → 2 context blocks → 3 citations), whether it ran, and
*why not* if it didn't: "not reached" for a stage the pipeline stopped before,
"disabled by `entity_guard_enabled`" for one switched off. A guard that never
fired and a guard that is turned off must be distinguishable, or the trace is
decorative. The last rows are the end-to-end time with the outcome
(`answered` / `abstained (INSUFFICIENT_EVIDENCE)`) and the **decisive check** —
the one sentence that explains the result:

```
entity guard: question asks about Project Beta; evidence only mentions Alpha
```

Refusals carry the same trace as answers, including their total time, so
"refused in 636 ms without calling the model" is a fact you can show rather than
assert. Traces are stored with each chat-history message and replayed from
history.

**The run manifest** — `GET /manifest`, `scripts\manifest.py`, the Streamlit
sidebar and every evaluation report all carry the same document: git commit and
whether the tree is dirty, platform, the four model names, every retrieval,
chunking, guard, generation and auth setting, index counts and dataset version.
A screenshot of an answer is only reproducible if the build behind it is recorded
too.

`GET /health` adds host headroom to what it already reported — CPU, RAM, free
disk, uptime, auth mode, chunking mode, guard flags — and degrades from `ok` to
`degraded` with a note when free disk or available RAM is too low for ingestion
or model loading. "The service answered 200" is not the same as "the service can
still write to its index".

---

## Offline operation

```powershell
.venv\Scripts\python.exe scripts\verify_offline.py
```

This does more than assert. It **monkey-patches `socket.connect` to block every
non-loopback address** for the whole run, then loads all four models, reloads the
index, runs a full cross-modal query and an abstention check. If anything
silently reaches for the network it raises instead of quietly succeeding while
you still have Wi-Fi.

Verified result on this machine (`qwen2.5:0.5b`, 6-source demo index):

```
=== Embeddings, offline ===
  [OK]   384-d vector in 11.9s, no network used
=== Reranking, offline ===
  [OK]   cross-encoder logit 2.295 in 0.4s
=== Full question answering, offline ===
  [OK]   answered using only local models and the local Ollama server
=== Outbound access attempted during this run ===
  [OK]   none - every socket stayed on loopback

Offline verification PASSED. The runtime works with no internet.
```

**Loopback is not the internet.** Ollama runs at `127.0.0.1:11434` and Streamlit
talks to FastAPI over localhost. Those are local processes. The README says this
plainly rather than claiming zero sockets are opened.

For a physical test, disable your network adapter and run `start.ps1` — both the
guard and the real network will then be down.

---

## Persistence

FAISS writes to `index/faiss.index`, SQLite to `index/store.db`, on every ingest.

```powershell
.venv\Scripts\python.exe scripts\ingest_demo.py   # ingest
# close everything
.\start.ps1                                       # restart
```

The index reloads from disk. `test_index_survives_restart` and
`test_store_insert_search_and_reload` cover this in CI.

---

## Verification

There is no pytest suite in this repository, and nothing here claims otherwise.
Verification is five commands, run in this order, each with a real exit code:

| Command | Proves | Needs | Last run |
| --- | --- | --- | --- |
| `python scripts\selftest.py` | **86/86 checks**: constraint guards (entity, temporal, period targeting, ordinal-quarter pairing, and that reordering does not rescore), evidence completion, repetition collapse, numeric-guard locator exclusions, parent building and expansion, stage trace, manifest, and the `parents` table round-trip including delete cascade | nothing — no LLM, no network, throwaway index | exit 0 |
| `python scripts\apitest.py` | **30 checks** over the browser-facing API, through a test client so no port is bound: the static bundle and its relative asset paths under `/ui`, `POST /login`, `GET /me`, `POST /logout`, viewer-versus-admin permissions (403 not 401), stale versions, manifest, first-run bootstrap and the lockout ladder. `--with-query` adds the live `/query` contract (**32/32** measured) | nothing for the 30; Ollama for `--with-query` | exit 0 |
| `python scripts\verify_env.py` | every import resolves and each external binary is reachable, including a real audio decode | installed dependencies | exit 0 |
| `python scripts\smoke_test.py` | **4/4** end-to-end questions answered with citations, plus a genuine abstention | Ollama + cached models | exit 0 |
| `python scripts\verify_offline.py` | blocks every non-loopback socket for the whole run, then loads the models and answers — proving the offline claim rather than asserting it | cached models | exit 0 |
| `python evaluation.py --compare` | **16/16 PASS**, no gated metric below `baseline.json`, per modality and per query | Ollama + built index | exit 0 |

`selftest.py` is the one to run before a demo: it needs no model, no index and no
network, so it works on a laptop that has not finished downloading anything yet,
and it exits non-zero if a guard has stopped behaving.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `OLLAMA_NOT_AVAILABLE` | daemon down | `ollama serve` |
| `QWEN_MODEL_NOT_AVAILABLE` | model missing | `ollama pull qwen2.5vl:3b` |
| `OCR_NOT_AVAILABLE` | Tesseract missing | `winget install UB-Mannheim.TesseractOCR` |
| `CONVERSION_UNAVAILABLE` | LibreOffice missing | install it, or convert `.doc` → `.docx` manually |
| `WinError 1314` on model download | Windows symlink privilege | handled automatically; else enable Developer Mode |
| First query takes ~40 s | model was unloaded | expected; raise `OLLAMA_KEEP_ALIVE` |
| Very slow generation | model split to CPU | expected on 4 GB VRAM |
| Empty results | nothing indexed | run `scripts\ingest_demo.py` |
| Everything abstains | threshold too strict | raise `ABSTENTION_THRESHOLD` (and `ABSTENTION_THRESHOLD_IMAGE` for image questions) |
| Refused but evidence looks fine | a guard fired | read the **decisive check** row of the pipeline trace; it names the guard and why |
| Answer repeats itself | small model looping | `REPETITION_GUARD_ENABLED` collapses it; `OLLAMA_NUM_PREDICT` caps it |
| First query takes ~35 s | models loading | expected once per process; the UI says so while it waits |
| `401` from the API | no or revoked token | `scripts\create_user.py <name> --create-token`, or `POST /login` |
| `423` from `/login` | account locked | wait 5 minutes, or reset the password with `--set-password` |
| React UI shows nothing | `webui/dist` missing | `cd webui && npm install && npm run build` |

---

## Limitations

Stated plainly rather than hidden.

1. **Not a joint embedding space.** Images and audio are retrieved through text
   proxies (description + OCR, transcript). A purely visual similarity with no
   describable content will not retrieve well.
2. **`confidence` is not calibrated.** It is a sigmoid of the best cross-encoder
   score — a relative ordering signal, useful for comparing results within one
   query, not a probability that the answer is correct.
3. **Citations are sentence-to-evidence, not claim-level.** A sentence is matched
   to its best-supporting evidence by embedding similarity. That is an
   approximation, not proof of entailment.
4. **Generation is not bit-reproducible.** See [Evaluation](#evaluation).
5. **A 3B model is a small model.** It occasionally reformats numbers; the prompt
   explicitly forbids this and the evaluation checks for invented numbers, but
   the risk is not zero.
6. **DOCX has no page numbers.** By design — see [Citations](#citations).
7. **`.doc` needs LibreOffice.** Without it, `.doc` ingestion fails loudly with
   `CONVERSION_UNAVAILABLE` rather than silently producing nothing.
8. **Diarization clusters, it does not identify.** `diarization.py` groups
   transcript segments by spectral clustering over the audio, so a block can say
   "speaker 2" — it cannot say who that is, and with short or single-voice
   recordings it will report one speaker. It is gated by `DIARIZATION_ENABLED`
   and never invents a second voice where the audio has one.
9. **English only** as configured (Tesseract `eng`, Whisper default).
10. **Single-machine, local scale.** `IndexFlatIP` is exact search, ideal up to
    ~10⁵ vectors; beyond that, switch to an approximate index. Accounts, tokens
    and history live in local SQLite files, and the service speaks plain HTTP —
    passwords are salted and hashed and tokens are stored as digests, but putting
    this on a network means putting TLS in front of it. The lockout slows guessing;
    it does not stop an attacker who has the database file.
11. **The guards are deterministic, not semantic.** They compare numbers, years,
    entities and repeated spans. A paraphrase that drops the year entirely can
    pass the temporal guard, and the coverage guard checks that a question's
    figures appear in the evidence, not that the evidence *means* what the answer
    says. They are a floor under answer quality, not a proof of it.
12. **Structure mode is opt-in and measured worse here.** `CHUNKING_MODE=structure`
    with parent expansion scored 13/16 against 16/16 for `fixed` on this corpus
    with a 0.5B model: more context per block gave the model more room to
    confabulate (it produced a confident "87%" for a figure the evidence did not
    contain, and denied a limitation that was in the deck). Both runs are in
    `evaluation/reports/`. It stays in the codebase because the mechanism is sound
    and a larger model is the variable that changed — not because it currently
    wins.

---

## Project layout

```
EvidenceAI/
├── app.py                  Streamlit operator UI (accounts, history, ingestion, trace)
├── api.py                  FastAPI service + serves the React UI at /ui
├── rag_pipeline.py         the single orchestrator
├── config.py               all configuration
├── schemas.py              shared data contracts
├── providers.py            offline (Ollama) / online inference backends
│
├── documents.py            PDF / DOC / DOCX / TXT / CSV
├── chunking.py             fixed and structure-aware chunking
├── images.py               OCR + Qwen vision
├── ocr.py                  upscaling, confidence, digit filtering
├── audio.py                faster-whisper
├── diarization.py          spectral speaker clustering
│
├── embeddings.py           MiniLM (CPU, cached)
├── vector_store.py         FAISS + SQLite (items, sources, links, parents)
├── lexical.py              BM25 over the same items
├── retriever.py            hybrid dense + lexical search
├── reranker.py             cross-encoder
├── mmr.py                  diversity re-selection with a per-source cap
├── relationships.py        deterministic + semantic links
├── parents.py              small-to-big parent sections
├── llm.py                  Ollama / Qwen prompt building and generation
├── citations.py            programmatic citations
├── synthesis.py            labelled evidence completion
│
├── coverage_guard.py       question numbers must exist in the evidence
├── constraint_guard.py     entity, temporal and period-targeting checks
├── numeric_guard.py        answer numbers must exist in the evidence
├── repetition_guard.py     collapse looping generation
│
├── observability.py        stage trace + run manifest
├── chat_history.py         sessions, messages, stored traces
├── auth.py                 local accounts, roles, lockout, API tokens
├── llm_judge.py            optional LLM-graded arm (unreliable on small models)
├── evaluation.py           16-query harness + regression gate
├── create_demo_data.py     synthetic demo corpus
├── run_demo.py             the seven demo scenarios
│
├── webui/                  React front end
│   ├── src/                App, api client, Login, Ask, Answer, Evidence,
│   │                       StageTrace, Sources, Health
│   ├── dist/               committed build — no Node needed to run it
│   ├── vite.config.js      relative base + dev proxy
│   └── package.json        react, react-dom, vite
│
├── scripts/
│   ├── verify_env.py       import and binary verification
│   ├── prepare_models.py   online model caching
│   ├── ingest_demo.py      bulk ingestion
│   ├── reindex.py          rebuild the whole index from data/
│   ├── smoke_test.py       4 end-to-end questions + an abstention
│   ├── verify_offline.py   offline proof with a socket guard
│   ├── selftest.py         86 offline checks, no model or network
│   ├── calibrate_abstention.py  measure the threshold split on your corpus
│   ├── create_user.py      accounts, roles, tokens
│   └── manifest.py         print or write the run manifest
│
├── data/{documents,images,audio}/
├── index/                  FAISS + SQLite (generated)
├── history/                chat sessions (generated)
├── auth/                   accounts and token digests (generated)
├── evaluation/
│   ├── dataset/queries.json     16 queries, version 2.0
│   ├── expected_answers/        32 mirrored expectations
│   ├── expected_sources/
│   ├── baseline.json            the gate baseline — never overwritten silently
│   └── reports/                 versioned JSON + CSV + Markdown
│
├── README.md  DEMO_RUNBOOK.md  WINDOWS_SETUP_GUIDE.md  DEPLOYMENT.md
├── READ_ME_FIRST.txt  .env.example
├── requirements.txt  requirements-lock.txt  packages.txt
└── start.ps1  setup_windows.ps1  setup.ps1
```

---

## Demo script

```powershell
.venv\Scripts\python.exe create_demo_data.py
.venv\Scripts\python.exe scripts\ingest_demo.py
.venv\Scripts\python.exe run_demo.py
```

Seven scenarios: cross-modal text query, image→documents, audio→documents,
document Q&A, document→media discovery, abstention, and direct image inspection.

---

*Demo data is synthetic and clearly labelled. "Project Alpha" is fictional.*
