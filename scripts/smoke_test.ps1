# The five curl tests of Section IX (Windows). Usage: .\scripts\smoke_test.ps1 https://<public-url>
param([string]$Url = "http://localhost:8000")
$Url = $Url.TrimEnd("/")
if (-not $env:QA_API_TOKEN) { Write-Error "Set `$env:QA_API_TOKEN first"; exit 1 }
$req = Join-Path $PSScriptRoot "request.json"
Write-Host "== 1 /health, no token (expect 200)";     curl.exe -s -i "$Url/health"; Write-Host ""
Write-Host "== 2 /info, no token (expect 401)";       curl.exe -s -i "$Url/info"; Write-Host ""
Write-Host "== 3 /info, wrong token (expect 401)";    curl.exe -s -i -H "Authorization: Bearer wrong" "$Url/info"; Write-Host ""
Write-Host "== 4 /info, valid token (expect 200)";    curl.exe -s -i -H "Authorization: Bearer $env:QA_API_TOKEN" "$Url/info"; Write-Host ""
Write-Host "== 5 /predict, valid token (expect 200)"
curl.exe -s -i -X POST "$Url/predict" `
  -H "Authorization: Bearer $env:QA_API_TOKEN" `
  -H "Content-Type: application/json" `
  -d "@$req"
Write-Host ""
Write-Host 'Expected answer for sample-01: "16" or "16 attention heads" (start 383)'
