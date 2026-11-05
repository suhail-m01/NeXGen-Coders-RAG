$ErrorActionPreference = "Stop"

$Repo = (Get-Location).Path
$Package = Split-Path -Parent $MyInvocation.MyCommand.Path

Copy-Item (Join-Path $Package "grounding_audit.py") (Join-Path $Repo "grounding_audit.py") -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Repo "tests") | Out-Null
Copy-Item (Join-Path $Package "tests\test_grounding_audit.py") (Join-Path $Repo "tests\test_grounding_audit.py") -Force

Write-Host "Copied the additive Grounding Audit module and tests."
Write-Host "Now follow docs\INTEGRATION.md manually to patch rag_pipeline.py and citations.py."
Write-Host "Run: python -m pytest tests/test_grounding_audit.py -q"
