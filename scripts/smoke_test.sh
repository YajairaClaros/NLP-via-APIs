#!/usr/bin/env bash
# The five curl tests of Section IX.  Usage: scripts/smoke_test.sh https://<public-url>
set -uo pipefail
URL="${1:-http://localhost:8000}"; URL="${URL%/}"
: "${QA_API_TOKEN:?export QA_API_TOKEN first}"
REQ="$(dirname "$0")/request.json"
echo "== 1 /health, no token (expect 200)";      curl -s -i "$URL/health"; echo
echo "== 2 /info, no token (expect 401)";        curl -s -i "$URL/info"; echo
echo "== 3 /info, wrong token (expect 401)";     curl -s -i -H "Authorization: Bearer wrong" "$URL/info"; echo
echo "== 4 /info, valid token (expect 200)";     curl -s -i -H "Authorization: Bearer $QA_API_TOKEN" "$URL/info"; echo
echo "== 5 /predict, valid token (expect 200)";  curl -s -i -X POST "$URL/predict" \
  -H "Authorization: Bearer $QA_API_TOKEN" -H "Content-Type: application/json" -d @"$REQ"; echo
echo "Expected answer for sample-01: \"16\" or \"16 attention heads\" (start 383)"
