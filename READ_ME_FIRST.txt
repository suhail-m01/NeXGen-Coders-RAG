EVIDENCE AI - WINDOWS SETUP KIT
================================

Offline multimodal RAG over documents, images and audio, with per-item
citations and an abstention gate.

Source project: AnishPreetham/Multi-Modal-RAG, plus the files it was missing.

START HERE
----------
Open WINDOWS_SETUP_GUIDE.md - it lists every download with sizes, the models,
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

Use setup_windows.ps1, NOT setup.ps1. The original setup.ps1 calls
scripts\verify_env.py, scripts\prepare_models.py and scripts\ingest_demo.py,
which do not exist in the published repository, so it fails at "Import
verification". This kit supplies those scripts, and setup_windows.ps1 also
installs the PINNED requirements-lock.txt: the loose requirements.txt lets pip
pick PyAV 19+, which breaks faster-whisper's audio ingestion.

Nothing here needs an API key. Offline mode runs entirely on your machine.
The shipped .env.example has an empty OPENAI_API_KEY; setup copies it to .env.

Read section 8 of the guide ("Honest limitations") before demoing this to
anyone: the confidence score is not a probability of correctness, and the
abstention gate is sensitive to how a question is phrased.
