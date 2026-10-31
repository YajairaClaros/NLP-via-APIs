# Extractive QA as a Service

Fine-tuned Transformer encoder for SQuAD 2.0-style extractive question answering (with
unanswerable questions), served as an access-controlled REST API that implements
[`docs/brief/qa_api_contract.yaml`](docs/brief/qa_api_contract.yaml) exactly.

- **Model:** a base encoder fine-tuned by us (default `FacebookAI/roberta-base`, baseline `distilbert/distilbert-base-uncased`).
- **Serving:** FastAPI + uvicorn, CPU-only, on a laptop, exposed through a public tunnel.
- **Report:** [`REPORT.md`](REPORT.md).

```
qa/            inference + post-processing shared by evaluation and the service; SQuAD metrics
train/         train.py (fine-tuning), evaluate.py (dev metrics + no-answer threshold), plot_loss.py
app/           FastAPI service: main.py (endpoints, auth middleware, errors), settings.py (env vars)
eval/          harness.py (replica of the instructor's harness), shadow_set.json, challenge_set.json
scripts/       serve.sh/.ps1, smoke_test.sh/.ps1 (the 5 curl tests), githooks/pre-commit
tests/         offline tests with tiny BERT/RoBERTa models (no download needed)
logs/<run>/    training logs, loss curve, eval results (committed)
models/<run>/  fine-tuned weights (NOT committed; shared via GitHub release / shared folder)
docs/brief/    API contract and public sample set distributed with the brief
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
# GPU machine: first install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/
pip install -r requirements-train.txt  # training machine
pip install -r requirements.txt        # serving laptop only
git config core.hooksPath scripts/githooks   # blocks commits containing the token, .env or weights
python -m pytest -q                    # ~85 offline tests, a few seconds
```

## 2. Train (GPU)

Quick pipeline check (~10–20 min):

```bash
python -m train.train --model FacebookAI/roberta-base --run_name roberta-base-20k --max_train_samples 20000 --epochs 1
```

Full run:

```bash
python -m train.train --model FacebookAI/roberta-base --epochs 2 --lr 3e-5 --batch_size 16
# 8 GB GPU: --batch_size 8 --grad_accum 2      resume after a crash: add --resume
# baseline:  python -m train.train --model distilbert/distilbert-base-uncased --epochs 2 --lr 3e-5 --batch_size 32
```

Outputs: `models/roberta-base/` (weights, git-ignored) and `logs/roberta-base/` (`train_log.jsonl`,
`loss.png`, `train_summary.json`, committed).

## 3. Evaluate and choose the no-answer threshold

```bash
python -m eval.build_sets        # only if you edited eval/source/*.json
python -m train.evaluate --model_dir models/roberta-base \
       --data squad_v2 eval/shadow_set.json docs/brief/public_sample.json --write_threshold
```

The threshold that maximises the mean F1 across the datasets is written to
`models/roberta-base/qa_meta.json`. Curves: `logs/roberta-base/threshold.png`.
Per-example predictions: `logs/roberta-base/eval_*.json` (use them for error analysis).

## 4. Serve (laptop, CPU)

```bash
cp .env.example .env             # then edit: QA_API_TOKEN, QA_TEAM, QA_MODEL_DIR   (.env is git-ignored)
scripts/serve.sh                 # Windows: .\scripts\serve.ps1
```

| Variable | Meaning |
|---|---|
| `QA_API_TOKEN` | **required**, bearer token for `/info` and `/predict` (dev token now, new token on presentation day) |
| `QA_TEAM` | team id registered with the instructor, no spaces |
| `QA_MODEL_DIR` | folder with the fine-tuned model (default `models/roberta-base`) |
| `QA_NUM_THREADS` | optional, CPU threads for PyTorch |

The service listens on `127.0.0.1:8000`. `/health` returns 503 while the model loads, then 200.

## 5. Tunnel

```bash
ngrok http 8000                               # primary (fixed domain)
cloudflared tunnel --url http://localhost:8000  # backup
```

Test from another network (e.g. phone hotspot):

```bash
scripts/smoke_test.sh https://<public-url>                     # the 5 curl tests of the brief
python -m eval.harness --url https://<public-url>               # contract + access control + public sample
python -m eval.harness --url https://<public-url> --data eval/shadow_set.json
```

The harness applies the 5 s per-request timeout and 150 s global cap, and projects the live-evaluation
points with the grading formula. Results are saved in `eval/results/`.

## 6. Presentation-day checklist

- [ ] Laptop plugged in, sleep disabled, model in `models/`, `.env` with `QA_TEAM` set.
- [ ] New token: `export QA_API_TOKEN=<new token>` (PowerShell: `$env:QA_API_TOKEN="..."`), restart the service.
- [ ] Start tunnel, run `scripts/smoke_test.sh <url>` from another network, send the URL to the instructor.
- [ ] Keep service and tunnel running for the whole session.

## 7. Submission

- Weights: upload `models/<run>/` as a GitHub release asset or shared folder; link it in REPORT.md.
- `git tag submission && git push origin submission`. Do **not** squash history (merge PRs with a merge commit).
