# Extractive QA as a Service — team-Maiz

A RoBERTa-base model that we fine-tuned on SQuAD 2.0. Given a paragraph and a question, it returns
the exact span of the paragraph that answers it, or `null` when the paragraph does not contain the
answer. It runs on a laptop CPU behind a token-protected REST API that implements
[`docs/brief/qa_api_contract.yaml`](docs/brief/qa_api_contract.yaml), and it is published to the
internet through a Cloudflare tunnel.

| Deliverable | Where it is |
|---|---|
| Code (training + service) | `train/`, `qa/`, `app/` (this repository, tag `submission`) |
| Training logs and loss curve | `logs/roberta-base/` (`train_log.jsonl`, `loss.png`, `train_summary.json`, `eval_*.json`, `threshold.png`) |
| Running service | `app/main.py`, started with `scripts/serve.ps1`; dry-run evidence in `docs/dry_run/` |
| Report | [`REPORT.md`](REPORT.md) |
| Model weights | Not in git (496 MB, excluded by `.gitignore`). They live on the laptop that serves the API, in `models/roberta-base/`; step 3 explains how to rebuild them |

---

## Project structure

```
app/            FastAPI service: main.py (endpoints, token middleware, errors), settings.py, static/demo.html
qa/             inference.py (windows, null score, exact character offsets), metrics.py (EM / F1)
train/          train.py (fine-tuning), evaluate.py (dev metrics + no-answer threshold), plot_loss.py
eval/           harness.py (replica of the instructor's evaluator), shadow_set.json, challenge_set.json
scripts/        serve.ps1/.sh, smoke_test.ps1/.sh (the 5 curl tests), report_numbers.py, githooks/
tests/          89 offline tests (tiny models, no download needed)
logs/           training and evaluation logs (committed)
models/         fine-tuned weights (NOT committed)
docs/           brief files (contract, public sample), dry_run screenshots, QA_PREP.md
```

---

## Step by step (Windows PowerShell)

Commands for Linux/macOS are the same except where noted.

### 1. Requirements

- Python 3.11 or 3.12 and Git.
- To **train**: an NVIDIA GPU with at least 6 GB (we used an RTX 3050 Laptop, 6 GB, about 2 h 15 min).
- To **serve**: any laptop. Inference runs on CPU.
- Allow local scripts once (Windows only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Get the code and create the environment

```powershell
git clone https://github.com/YajairaClaros/NLP-via-APIs.git
cd NLP-via-APIs
python -m venv .venv
.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt       # serving only
# training machine: install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/), then:
# pip install -r requirements-train.txt
git config core.hooksPath scripts/githooks   # blocks commits that contain the token, .env or weights
```

Every new terminal needs `cd NLP-via-APIs` and `.venv\Scripts\Activate.ps1`. The prompt must start
with `(.venv)`.

Optional check that everything is installed (requires `requirements-train.txt`):

```powershell
python -m pytest -q        # expected: 89 passed
```

### 3. Get the model

The fine-tuned weights (496 MB) are not in the repository, as the brief requires. On the team's
laptop they are already in `models\roberta-base\` (`model.safetensors`, `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `qa_meta.json`). If you copy that folder from the team,
place it there and skip to step 5. To rebuild it from scratch you need a GPU:

```powershell
# quick check, ~15 min
python -m train.train --run_name roberta-base-20k --max_train_samples 20000 --epochs 1
# full training, ~2 h 15 min on an RTX 3050 6 GB
python -m train.train --epochs 2 --lr 3e-5 --batch_size 16
# with less GPU memory: add --batch_size 8 --grad_accum 2 ; to resume after a crash: add --resume
```

This writes `models\roberta-base\` (weights) and `logs\roberta-base\` (logs and `loss.png`).

### 4. Choose the no-answer threshold (only after training)

```powershell
python -m train.evaluate --model_dir models/roberta-base --data squad_v2 eval/shadow_set.json docs/brief/public_sample.json --write_threshold
```

It prints EM/F1 for each dataset and saves the best threshold in `models\roberta-base\qa_meta.json`
(ours: −2.049). The team's model folder already includes it.

### 5. Configure the service

Create `.env` from the example. Never commit it; `.gitignore` and the pre-commit hook prevent that.

```powershell
@"
QA_API_TOKEN=<your token>
QA_TEAM=team-Maiz
QA_MODEL_DIR=models/roberta-base
"@ | Set-Content -Path .env -Encoding ascii
Get-Content .env
```

| Variable | Meaning |
|---|---|
| `QA_API_TOKEN` | **Required.** Bearer token for `/info` and `/predict`. On presentation day, replace it with the instructor's token and restart. |
| `QA_TEAM` | Team id returned by `/info` (no spaces): `team-Maiz` |
| `QA_MODEL_DIR` | Folder with the fine-tuned model |
| `QA_NUM_THREADS` | Optional. CPU threads for PyTorch |

### 6. Start the API (terminal 1, keep it open)

```powershell
.\scripts\serve.ps1
# same as: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Wait for `Model loaded from ...`. Then check it from another terminal:

```powershell
curl.exe http://localhost:8000/health     # {"status":"ok"}
```

| Endpoint | Access | Returns |
|---|---|---|
| `GET /health` | public | `200 {"status":"ok"}`, or `503 {"status":"loading"}` while the model loads |
| `GET /info` | token | `team`, `model`, `params`, `max_seq_length`, `doc_stride` |
| `POST /predict` | token | `{"id", "answer", "start", "score"}`; `answer`/`start` are `null` when there is no answer |
| `GET /demo` | public page | Demo for the pitch; it asks for the token before calling the API |

Opening the bare URL (`/`) gives `404 Not Found` on purpose: only the routes above exist.

### 7. Open the public tunnel (terminal 2, keep it open)

We use **Cloudflare Quick Tunnel**: no account is needed, and it is one of the tools allowed by the brief.

```powershell
winget install --id Cloudflare.cloudflared     # once; then open a NEW terminal
cloudflared tunnel --url http://localhost:8000
# if "cloudflared" is not recognized:
# & "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```

Copy the `https://<random-words>.trycloudflare.com` URL that it prints. The URL changes every time
the tunnel restarts, so keep this window open for the whole session and send the instructor the
current URL.

### 8. Test it like the instructor (terminal 3)

Ideally do this from another network, e.g. a phone hotspot.

```powershell
$env:QA_API_TOKEN = "<your token>"
$URL = "https://<random-words>.trycloudflare.com"

curl.exe -i $URL/health                                                 # 1) 200
curl.exe -i $URL/info                                                   # 2) 401
curl.exe -i -H "Authorization: Bearer wrong" $URL/info                  # 3) 401
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" $URL/info      # 4) 200 with team, model, params
curl.exe -i -X POST $URL/predict -H "Authorization: Bearer $env:QA_API_TOKEN" -H "Content-Type: application/json" -d "@scripts/request.json"   # 5) 200, answer "16"

python -m eval.harness --url $URL                                       # contract + access control + public sample
python -m eval.harness --url $URL --data eval/shadow_set.json --skip_contract   # our out-of-domain set
```

The harness applies the 5 s per-request timeout and the 150 s global cap, prints the projected
live-evaluation points and saves the results in `eval\results\`.

### 9. Demo for the pitch

Open `https://<random-words>.trycloudflare.com/demo` and paste the token. Pick an example, or paste
your own English text, and press **Ask**. The page highlights the answer in the passage, or shows
"Not in the text", together with the confidence and the latency.

---

## Presentation-day checklist

- [ ] Laptop plugged in, high-performance mode, sleep disabled (`powercfg /change standby-timeout-ac 0`), heavy programs closed.
- [ ] Set the instructor's token in `.env` (`QA_API_TOKEN=...`), start the API (step 6) and wait for `Model loaded`.
- [ ] Start the tunnel (step 7) and run the 5 curl tests (step 8) with the new token from another network.
- [ ] Send the current tunnel URL to the instructor and keep both windows open for the whole session.
- [ ] Fallback if no tunnel works: the instructor's harness runs on this laptop against `http://localhost:8000`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1` or `serve.ps1` "cannot be loaded" / "is not digitally signed" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then `Unblock-File .\scripts\*.ps1` |
| `QA_API_TOKEN is not set` | `.env` is missing or was saved as `.env.txt`; create it with the command in step 5 |
| `/health` stays at `loading` | Read the error in terminal 1; usually `QA_MODEL_DIR` points to a folder without the model |
| `curl: (7) Could not connect to server` | The API is not running; start it again (step 6) and keep that window open |
| `Could not resolve host ...trycloudflare.com` | The tunnel window was closed; start it again (step 7) and use the new URL |
| `/info` shows `team-XX` | Set `QA_TEAM` in `.env` and restart the API (the value is read once at startup) |
| `cloudflared` / `ngrok` "not recognized" right after installing | Open a new terminal (the PATH is refreshed only in new terminals) |
| ngrok `ERR_NGROK_121` (agent too old) or blocked by Windows Defender | Use Cloudflare (step 7); this is what we did |
| Harness says `Set QA_API_TOKEN` | Run `$env:QA_API_TOKEN = "..."` in that terminal (the harness does not read `.env`) |
| `CUDA out of memory` while training | `--batch_size 8 --grad_accum 2` |
| `pre-commit: refusing to commit` | You tried to commit `.env`, weights or the token: `git reset <file>` |
