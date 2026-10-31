# Start the QA service on port 8000 (Windows PowerShell).
# The token comes from the environment or from the untracked .env file -- never from code.
Set-Location (Join-Path $PSScriptRoot "..")
$hasEnvFile = (Test-Path .env) -and (Select-String -Path .env -Pattern '^QA_API_TOKEN=' -Quiet)
if (-not $env:QA_API_TOKEN -and -not $hasEnvFile) {
    Write-Error "QA_API_TOKEN is not set (`$env:QA_API_TOKEN = '...' or create .env from .env.example)"; exit 1
}
# 127.0.0.1: only the tunnel (running on this laptop) can reach the service, not the classroom Wi-Fi.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
