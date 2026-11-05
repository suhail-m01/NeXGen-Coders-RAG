EVIDENCE AI - WINDOWS SETUP KIT
================================

Offline multimodal RAG over documents, images and audio, with per-item
citations, three grounding guardrails, speaker labels, collections, streamed
answers and a persistent, replayable chat history.

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
    5. cd C:\EvidenceAI
       Get-ChildItem -Recurse | Unblock-File
       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    6. .\setup_windows.ps1
       (low-memory machine instead:  .\setup_windows.ps1 -Model qwen2.5:1.5b)
    7. .\start.ps1
       UI  ->  http://localhost:8501
       API ->  http://127.0.0.1:8000/docs

    Checks, any time:
       .venv\Scripts\python scripts\verify_env.py       environment
       .venv\Scripts\python scripts\smoke_test.py       real Q&A with citations
       .venv\Scripts\python scripts\verify_offline.py   proves no internet needed
       .venv\Scripts\python evaluation.py               6-query scored report
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
Ask tab    - streamed answer, citations with page/timestamp, evidence panel
             showing rerank scores, OCR confidence and speaker labels. An
             answer containing a figure the evidence does not support is
             badged "Partially grounded" and carries an Evidence warning line.
History    - sidebar. Sessions persist in history\chat_history.db with their
             full retrieval trace, so a past answer can be re-opened and
             inspected without re-running the query. Markdown/CSV export.
Ingest tab - Collection field groups sources; Search scope in the Ask tab
             restricts retrieval to one collection.
Sources    - every indexed item, its location and its collection.

Read section 8 of the guide ("Honest limitations") before demoing this to
anyone: the confidence score is not a probability of correctness, the
abstention gate is sensitive to how a question is phrased, and chat history is
stored unencrypted.
