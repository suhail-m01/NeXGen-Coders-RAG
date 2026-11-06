EVIDENCE AI - WINDOWS SETUP KIT
================================

Offline multimodal RAG over documents, images and audio, with per-item
citations, seven deterministic evidence guards, speaker labels, collections,
streamed answers, a persistent replayable chat history, a per-query pipeline
trace, and two front ends: a React UI served by the API at /ui, and a Streamlit
operator UI.

Source project: AnishPreetham/Multi-Modal-RAG, plus the files it was missing
and an upgrade programme on top of it (section 7 of the guide lists every
change, and which upstream files were modified).

START HERE
----------
DEMONSTRATING SOON? Open DEMO_RUNBOOK.md - prerequisites, exact commands, the
questions to ask in front of an audience with the answers they should produce,
what to do if something fails live, and the limitations worth stating out loud.

For installation detail, open WINDOWS_SETUP_GUIDE.md - it lists every download with sizes, the models,
the API/key situation (none needed offline), and the 10 installation steps.

Short version, in PowerShell:

    1. Install Python 3.11+        https://www.python.org/downloads/windows/
       (tick "Add python.exe to PATH")
    2. Install Ollama              https://ollama.com/download
    3. Install Tesseract OCR       winget install UB-Mannheim.TesseractOCR
    4. Unzip this kit to           C:\EvidenceAI      (no spaces, not OneDrive)
       Prove the download first - needs only Python, nothing installed yet:
       python scripts\verify_kit.py --zip EvidenceAI_Windows_Kit.zip --extract C:\EvidenceAI
    5. cd C:\EvidenceAI
       Get-ChildItem -Recurse | Unblock-File
       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    6. .\setup_windows.ps1
       (low-memory machine instead:  .\setup_windows.ps1 -Model qwen2.5:1.5b)
    7. Print DEMO_CUE_CARD.md if you are demonstrating - one page, the seven
       questions in order with the seconds each costs and the lines to say.
    8. .\start.ps1
       React UI ->  http://127.0.0.1:8000/ui/   (opened for you)
       Streamlit ->  http://localhost:8501       (accounts, history, collections)
       API docs ->  http://127.0.0.1:8000/docs

    Checks, any time (cheapest first):
       .venv\Scripts\python scripts\selftest.py         258 checks, no model/network
       .venv\Scripts\python scripts\verify_kit.py       10 checks on this archive
                                                          (RECOVERY.md if one fails)
       .venv\Scripts\python scripts\apitest.py          78 checks on the API + React UI
       .venv\Scripts\python scripts\verify_env.py       environment
       .venv\Scripts\python scripts\smoke_test.py       real Q&A with citations
       .venv\Scripts\python scripts\verify_offline.py   proves no internet needed
       .venv\Scripts\python evaluation.py --compare     16-query report + the gate
       .venv\Scripts\python scripts\dump_chunks.py --all  what is indexed, chunk sizes
       .venv\Scripts\python scripts\manifest.py         the build behind a result
       .venv\Scripts\python scripts\calibrate_abstention.py   threshold report

    Group sources at ingest time, then scope a search to them:
       .venv\Scripts\python scripts\ingest_demo.py --dir "D:\contracts" ^
           --collection contracts-2026

Use setup_windows.ps1, NOT setup.ps1. The original setup.ps1 calls
scripts\verify_env.py, scripts\prepare_models.py and scripts\ingest_demo.py,
which do not exist in the published repository, so it fails at "Import
verification". This kit supplies those scripts, and setup_windows.ps1 also
installs the PINNED requirements-lock.txt: the loose requirements.txt lets pip
pick PyAV 19+, which breaks faster-whisper's audio ingestion.

Nothing here needs an API key. Offline mode runs entirely on your machine.
The shipped .env.example has an empty OPENAI_API_KEY; setup copies it to .env.

WHAT TO EXPECT IN THE UI
------------------------
React UI (http://127.0.0.1:8000/ui/) - the asking surface:
  Ask       - sign in, ask, and read the answer with its citation markers as
              buttons: selecting one scrolls to and outlines the evidence block
              it came from. Guard output (an Evidence warning, a Correction, an
              "Also stated in the evidence" addition) is rendered as its own
              block, not folded into the paragraph. A refusal is shown as a
              refusal, with its error code.
  Evidence  - every block the model was allowed to use, in order, with its
              retrieval and rerank scores, OCR confidence and dropped-digit
              counts for images, and the matched passage shown separately from
              a wider parent section when the index was built in structure mode.
  Trace     - all fifteen stages with their milliseconds, what each moved, which
              did not run and why, and the decisive check that explains the
              result. Refusals carry the same trace, so "refused in 0.7 s
              without calling the model" is shown, not claimed.
  Sources   - the indexed corpus, admin-only delete, and a banner naming any
              file with more than one indexed version so a stale copy stops
              being citable by accident.
  Health    - backend readiness, index size, host headroom, every guard flag,
              and the full run manifest.

Streamlit (http://localhost:8501) - the operator surface:
  Ask tab    - streamed answer with the same citations and evidence panel; an
               answer containing an unsupported figure is badged "Partially
               grounded" and carries an Evidence warning line.
  History    - sidebar. Sessions persist in history\chat_history.db with their
               stored pipeline trace, so a past answer can be re-opened and
               inspected without re-running the query. Markdown/CSV export.
  Ingest tab - Collection field groups sources; Search scope in the Ask tab
               restricts retrieval to one collection.
  Accounts   - create users, set roles, mint and revoke API tokens.

Read section 8 of the guide ("Honest limitations") before demoing this to
anyone: the confidence score is not a probability of correctness, the guards
compare text rather than meaning, structure-aware chunking measures worse at
this model size and ships opt-in, and chat history is stored unencrypted.
