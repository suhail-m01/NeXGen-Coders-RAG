<#
.SYNOPSIS
    Evidence AI - one-time ONLINE preparation for Windows (pinned, fixed version).

.DESCRIPTION
    This is a corrected replacement for setup.ps1. The original fails on a fresh
    clone for two reasons, both fixed here:

      1. It calls scripts\verify_env.py, scripts\prepare_models.py and
         scripts\ingest_demo.py, which are not present in the repository.
         This kit ships them.
      2. It installs requirements.txt, whose loose pins let pip resolve a PyAV
         build (>=19) that breaks faster-whisper's audio path. This script
         installs requirements-lock.txt by default instead.

    After it succeeds the app runs with no internet connection.

.PARAMETER Model
    Ollama model tag to pull and to write into .env.
    Default qwen2.5vl:3b (~3.2 GB, needs a 4 GB VRAM GPU).
    Use qwen2.5:1.5b or qwen2.5:0.5b on machines with less memory (no vision).

.PARAMETER Loose
    Install from requirements.txt instead of requirements-lock.txt.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Model qwen2.5:1.5b
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipOllama
#>
param(
    [string]$Model = "qwen2.5vl:3b",
    [switch]$Loose,
    [switch]$SkipOllama,
    [switch]$SkipModels,
    [switch]$SkipDemoData,
    [switch]$SkipIngest,
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root "$VenvName\Scripts\python.exe"
$script:hadFailure = $false

function Say($text)  { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Good($text) { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "  [FAIL] $text" -ForegroundColor Red; $script:hadFailure = $true }
function Note($text) { Write-Host "         $text" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Evidence AI - Windows setup (online phase)" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Project folder : $root"
Write-Host " Ollama model   : $Model"
Write-Host " Dependencies   : $(if ($Loose) { 'requirements.txt (loose)' } else { 'requirements-lock.txt (pinned)' })"
Write-Host ""
Write-Host " This downloads several GB. Keep the internet connected until" -ForegroundColor DarkGray
Write-Host " it finishes; afterwards the app runs offline." -ForegroundColor DarkGray

# --------------------------------------------------------------- Python
Say "Step 1/9  Python 3.11 or newer"
$pythonCmd = $null
foreach ($candidate in @("python", "py -3", "python3")) {
    try {
        # Join to a single string: -match against an ARRAY returns the matching
        # elements but does NOT populate $Matches, which would make the version
        # test below silently fail.
        $out = (& cmd /c "$candidate --version 2>&1") -join " "
        if ($LASTEXITCODE -eq 0 -and $out -match "Python\s+(\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 11) { $pythonCmd = $candidate; break }
            else { Warn "Found $out but 3.11 or newer is required." }
        }
    } catch { }
}
if (-not $pythonCmd) {
    Bad "Python 3.11+ not found on PATH."
    Note "Download the installer from https://www.python.org/downloads/windows/"
    Note "Tick 'Add python.exe to PATH' during installation, then re-run."
    exit 1
}
$versionText = (& cmd /c "$pythonCmd --version 2>&1") -join " "
Good "$versionText  (via '$pythonCmd')"

# --------------------------------------------------------------- Machine
Say "Step 2/9  Machine check (informational)"
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    if ($ramGB -ge 8) { Good "System RAM: $ramGB GB" }
    else { Warn "System RAM: $ramGB GB - 8 GB or more is recommended." }

    $freeGB = [math]::Round((Get-PSDrive -Name ($root.Substring(0,1))).Free / 1GB, 1)
    if ($freeGB -ge 12) { Good "Free disk on this drive: $freeGB GB" }
    else { Warn "Free disk on this drive: $freeGB GB - about 12 GB is needed." }
} catch { Warn "Could not read system information: $_" }

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $gpuLine = (& cmd /c "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1") -join " "
        Good "NVIDIA GPU: $gpuLine"
        if ($gpuLine -match "(\d+)\s*MiB") {
            $vramMB = [int]$Matches[1]
            if ($vramMB -lt 4096 -and $Model -like "qwen2.5vl*") {
                Warn "Less than 4 GB VRAM. $Model may not fit."
                Note "Re-run with a smaller model, e.g.  -Model qwen2.5:1.5b"
            }
        }
    } catch { Warn "nvidia-smi present but could not be queried." }
} else {
    Warn "No NVIDIA GPU detected (nvidia-smi not on PATH)."
    Note "The app still runs - Ollama will use the CPU and be much slower."
    Note "A small model is strongly advised:  -Model qwen2.5:1.5b"
}

# ------------------------------------------------------------ Virtualenv
Say "Step 3/9  Virtual environment and .env"
$envPath = Join-Path $root ".env"
if (-not (Test-Path $envPath)) {
    $example = Join-Path $root ".env.example"
    if (Test-Path $example) {
        Copy-Item $example $envPath
        Good "Created .env from .env.example (OPENAI_API_KEY is empty - offline mode needs no key)"
    } else {
        Warn ".env.example not found - the built-in defaults in config.py will be used."
    }
} else {
    Good ".env already present"
}

if (-not (Test-Path $py)) {
    # Create the venv at an absolute path, so the script works when invoked
    # from any working directory. Split the launcher ("py -3" -> py, -3) and
    # let PowerShell handle quoting, which copes with spaces in $root; going
    # through cmd /c would need nested quotes that cmd mis-parses.
    $venvPath = Join-Path $root $VenvName
    $pythonParts = @($pythonCmd -split ' ' | Where-Object { $_ })
    if ($pythonParts.Count -gt 1) {
        & $pythonParts[0] $pythonParts[1..($pythonParts.Count - 1)] -m venv $venvPath
    } else {
        & $pythonParts[0] -m venv $venvPath
    }
    if (-not (Test-Path $py)) {
        Bad "Could not create $venvName - no interpreter at $py"
        Note "If 'py' launched the Microsoft Store stub, install Python from"
        Note "python.org and tick 'Add python.exe to PATH'."
        exit 1
    }
    Good "Created $venvPath"
} else {
    Good "$VenvName already exists"
}
& $py -m pip install --upgrade pip setuptools wheel --quiet --disable-pip-version-check
Good "pip, setuptools and wheel upgraded"

# ---------------------------------------------------------------- Torch
# Everything from here on is a native command whose success is judged by
# $LASTEXITCODE. Under "Stop", PowerShell can turn a native command's stderr
# into a terminating error, which would abort setup on harmless warnings.
$ErrorActionPreference = "Continue"

Say "Step 4/9  PyTorch (CPU wheel)"
Note "The GPU is reserved for Ollama, so torch must be the CPU build."
# Install the exact pinned build now; otherwise pip fetches the newest CPU
# wheel here and then downgrades it in Step 5, wasting a ~500 MB download.
$torchSpec = "torch"
$lockPath = Join-Path $root "requirements-lock.txt"
if (-not $Loose -and (Test-Path $lockPath)) {
    $pin = Select-String -Path $lockPath -Pattern '^\s*torch\s*==\s*(\S+)' | Select-Object -First 1
    if ($pin) {
        $torchSpec = "torch==" + $pin.Matches[0].Groups[1].Value
        Note "Version pinned by requirements-lock.txt: $torchSpec"
    }
}
& $py -m pip install --quiet --disable-pip-version-check `
    --index-url https://download.pytorch.org/whl/cpu $torchSpec
if ($LASTEXITCODE -eq 0) { Good "torch (CPU) installed: $torchSpec" } else { Bad "torch install failed"; exit 1 }

# --------------------------------------------------------- Dependencies
Say "Step 5/9  Python dependencies"
$reqFile = if ($Loose) { "requirements.txt" } else { "requirements-lock.txt" }
& $py -m pip install --quiet --disable-pip-version-check `
    -r (Join-Path $root $reqFile) `
    --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -eq 0) {
    Good "$reqFile installed"
} else {
    Bad "$reqFile install failed"
    if (-not $Loose) {
        Note "A pinned wheel may be unavailable for your Python version."
        Note "Retry with:  .\setup_windows.ps1 -Loose   (then re-run verify_env)"
    }
    exit 1
}

Say "Step 6/9  Import verification"
& $py (Join-Path $root "scripts\verify_env.py")
if ($LASTEXITCODE -ne 0) {
    Bad "One or more imports failed - see the output above."
    exit 1
}

# -------------------------------------------------------- External tools
Say "Step 7/9  External tools"
$tesseractCandidates = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)
$tesseractFound = $tesseractCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($tesseractFound) {
    Good "Tesseract OCR: $tesseractFound"
} elseif (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Good "Tesseract OCR found on PATH"
    Warn "It is not at the path in .env - set TESSERACT_CMD to its full path."
} else {
    Warn "Tesseract OCR not installed - images will be indexed WITHOUT OCR text."
    Note "Install with:  winget install UB-Mannheim.TesseractOCR"
    Note "Then restart PowerShell so PATH updates."
}

if (Test-Path "C:\Program Files\LibreOffice\program\soffice.exe") {
    Good "LibreOffice present - legacy .doc conversion enabled"
} else {
    Warn "LibreOffice not installed - only legacy .doc files are affected."
    Note "Optional:  winget install TheDocumentFoundation.LibreOffice"
}

# -------------------------------------------------------------- Ollama
Say "Step 8/9  Ollama and the answer model"
if ($SkipOllama) {
    Warn "Skipped (-SkipOllama). Offline answering will not work until you run:"
    Note "ollama pull $Model"
} else {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Bad "Ollama is not installed or not on PATH."
        Note "Download the Windows installer from https://ollama.com/download"
        Note "Install it, open a NEW PowerShell window, then re-run this script."
        exit 1
    }
    Good (& cmd /c "ollama --version 2>&1" | Select-Object -First 1)

    # Make sure the server is up so the pull and the .env update are consistent.
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    } catch {
        Warn "Ollama server not responding - starting it..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 6
    }

    $installed = @()
    try {
        $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10
        $installed = @($tags.models.name)
    } catch { Warn "Could not list local models." }

    if ($installed -contains $Model) {
        Good "$Model already present locally"
    } else {
        if ($installed.Count -gt 0) { Note "Installed now: $($installed -join ', ')" }
        Write-Host "  Pulling $Model (one time, several GB)..." -ForegroundColor Yellow
        & ollama pull $Model
        if ($LASTEXITCODE -ne 0) { Bad "ollama pull $Model failed"; exit 1 }
        Good "$Model pulled"
    }

    # Keep .env in step with the model actually installed.
    $envPath = Join-Path $root ".env"
    if (Test-Path $envPath) {
        $envText = Get-Content $envPath -Raw
        if ($envText -match "(?m)^\s*OLLAMA_MODEL\s*=\s*(.+?)\s*$") {
            $current = $Matches[1]
            if ($current -ne $Model) {
                $envText = $envText -replace "(?m)^(\s*OLLAMA_MODEL\s*=\s*).+$", "`$1$Model"
                Set-Content -Path $envPath -Value $envText -NoNewline -Encoding UTF8
                Good ".env updated: OLLAMA_MODEL $current -> $Model"
            } else {
                Good ".env already targets $Model"
            }
        }
    }
}

# ------------------------------------------------------- HuggingFace models
Say "Step 9/9  Local model cache (embeddings, reranker, speech-to-text)"
if ($SkipModels) {
    Warn "Skipped (-SkipModels). Run scripts\prepare_models.py before going offline."
} else {
    $prepareArgs = @((Join-Path $root "scripts\prepare_models.py"))
    if ($SkipOllama) { $prepareArgs += "--skip-ollama" } else { $prepareArgs += "--ollama-model"; $prepareArgs += $Model }
    & $py @prepareArgs
    if ($LASTEXITCODE -ne 0) { Bad "Model preparation failed - see above"; exit 1 }
}

# ------------------------------------------------------------- Demo data
if (-not $SkipDemoData) {
    Say "Demo dataset"
    & $py (Join-Path $root "create_demo_data.py")
    if ($LASTEXITCODE -ne 0) {
        Warn "Demo data generation reported a problem (audio needs Windows speech)."
        Note "Other formats are unaffected. You can ingest your own files instead."
    }
}

if (-not $SkipIngest) {
    Say "Building the index from the demo dataset"
    & $py (Join-Path $root "scripts\ingest_demo.py")
    if ($LASTEXITCODE -ne 0) {
        Warn "Index build reported failures - see above. Re-running is safe;"
        Note "already-ingested files are skipped, not duplicated."
    }
}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
if ($script:hadFailure) {
    Write-Host " SETUP FINISHED WITH WARNINGS" -ForegroundColor Yellow
} else {
    Write-Host " SETUP COMPLETE" -ForegroundColor Green
}
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Next:"
Write-Host "   .\start.ps1                                  API + UI"
Write-Host "   .\start.ps1 -UiOnly                          UI only (lighter)"
Write-Host "   .venv\Scripts\python scripts\smoke_test.py   ask test questions"
Write-Host "   .venv\Scripts\python scripts\verify_offline.py   prove no internet"
Write-Host "   .venv\Scripts\python evaluation.py           run the evaluation"
Write-Host ""
Write-Host " UI:  http://localhost:8501     API docs:  http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host " You can now disconnect from the internet." -ForegroundColor Cyan
Write-Host ""
