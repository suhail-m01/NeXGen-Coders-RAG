# =============================================================
# Evidence AI - start the application (OFFLINE RUNTIME)
# =============================================================
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   .\start.ps1 -ApiOnly     FastAPI + the React UI it serves at /ui
#   .\start.ps1 -UiOnly      Streamlit operator UI only
#   .\start.ps1 -NoBrowser   do not open a browser window
# =============================================================

param(
    [switch]$ApiOnly,
    [switch]$UiOnly,
    [switch]$NoBrowser,
    [int]$ApiPort = 8000,
    [int]$UiPort  = 8501
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"

function Say($text)  { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Good($text) { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "  [FAIL] $text" -ForegroundColor Red }

if (-not (Test-Path $py)) {
    Bad "No virtual environment found. Run .\setup.ps1 first."
    exit 1
}

Say "Preflight"

# Offline AI (Ollama). This is checked but NEVER fatal: Online mode does not
# need Ollama, and the app reports offline readiness in its own UI with a
# Check Again / Install / Switch to Online flow.
$ollamaReady = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaReady = $true
    Good "Ollama is running"
} catch {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Warn "Ollama is not responding - starting it..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
            $ollamaReady = $true
            Good "Ollama started"
        } catch {
            Warn "Could not start Ollama. Offline mode will be unavailable."
        }
    } else {
        Warn "Ollama is not installed. Offline mode will be unavailable."
    }
}

if ($ollamaReady) {
    # Which model to look for comes from .env, because the demo kit ships with
    # OLLAMA_MODEL=qwen2.5:0.5b (400 MB, CPU-only) while the default in config.py
    # is the 3B vision model. Checking a hardcoded name here warned "not
    # installed" on a laptop that had followed the runbook exactly.
    $model = "qwen2.5vl:3b"
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        $found = Select-String -Path $envFile -Pattern '^\s*OLLAMA_MODEL\s*=\s*(.+?)\s*$' | Select-Object -First 1
        if ($found) { $model = $found.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
    }
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    if ($tags.models.name -contains $model) {
        Good "Offline AI ready - $model"
    } else {
        Warn "$model not installed. Run: ollama pull $model"
        $ollamaReady = $false
    }
}

# Online AI credentials. Presence only - the key itself is never printed.
if ($env:OPENAI_API_KEY -or (Test-Path (Join-Path $root ".env"))) {
    Good "Online AI credentials source present (.env or environment)"
} else {
    Warn "No OPENAI_API_KEY found. Online mode will need configuring in .env."
}

if (-not $ollamaReady) {
    Warn "Offline AI is not ready. The app will still start - select Online"
    Warn "in the sidebar, or use the setup buttons shown in the UI."
}

# Tesseract
if (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe") {
    Good "Tesseract OCR available"
} else {
    Warn "Tesseract missing - images will be indexed without OCR text"
}

# Index
$dbPath = Join-Path $root "index\store.db"
if (Test-Path $dbPath) {
    Good "Persistent index found"
} else {
    Warn "No index yet - ingest files from the UI, or run:"
    Warn "  .venv\Scripts\python scripts\ingest_demo.py"
}

Say "Launching"

if (-not $UiOnly) {
    Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -WorkingDirectory $root
    Good "FastAPI    http://127.0.0.1:$ApiPort/docs"
    Good "React UI   http://127.0.0.1:$ApiPort/ui/"
    Good "Streamlit  http://127.0.0.1:$UiPort  (operator UI: accounts, history)"
    if (-not $NoBrowser) {
        # The React UI is served by the API itself, so it is up as soon as the
        # port answers. Give uvicorn a moment rather than opening a dead tab.
        Start-Sleep -Seconds 4
        Start-Process "http://127.0.0.1:$ApiPort/ui/"
    }
}

if (-not $ApiOnly) {
    Good "Streamlit http://localhost:$UiPort"
    Write-Host "`n  Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    & $py -m streamlit run (Join-Path $root "app.py") --server.port $UiPort
} else {
    Write-Host "`n  API running. Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    while ($true) { Start-Sleep -Seconds 3600 }
}
