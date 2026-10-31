# Project Report — Extractive QA as a Service

> **TODO before tagging `submission`:** replace every `TODO` with real values from
> `logs/<run>/` and `eval/results/`. A missing or empty section counts as not delivered.

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-XX` <!-- TODO --> |
| Members | Edwin Hernández, German Dueñas, Leandro Amaya, Yajaira Claros |
| Repository URL |https://github.com/YajairaClaros/NLP-via-APIs.git |
| `submission` commit hash | TODO |
| Public service URL used at the dry run | TODO |
| AI tools used (which, and for what) | Claude (Anthropic): scaffolding of the training, evaluation and serving code, offline tests and the local harness; every module was reviewed and is explained by the team. No AI or commercial API is called at inference time. TODO: add any other tool used. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `FacebookAI/roberta-base` (TODO: confirm)
- **Alternatives considered** (at least one) and why you did not choose them:
  - `distilbert/distilbert-base-uncased` (66M params): ~2× faster on CPU, but it is the
    instructor's reference model (0.70). Matching it gives full points with no margin for the
    domain shift of the hidden set (texts not from SQuAD). We trained it as a baseline. TODO: its dev F1.
  - `google-bert/bert-base-uncased` (110M): same CPU cost as RoBERTa-base but weaker on SQuAD 2.0.
    RoBERTa was pre-trained on ~10× more text with dynamic masking and no next-sentence prediction.
- **Trade-off:** RoBERTa-base has 124M parameters (~500 MB). On a 2-core CPU it took ~0.2 s per
  question with a short context and ~1.8 s in the worst case (5,000 characters = 5 windows).
  The limits are 5 s per request and 150 s for 50 items, so it fits. It is cased and uses
  byte-level BPE, so it has no unknown tokens on names, numbers or accents
  (e.g. "Cerén", "2,381"). TODO: measured latency on our laptop (Section 6).

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | TODO (e.g. full: 130,319 questions → N windows) |
| Validation data (what you validated on, how many examples) | Loss: 2,000 dev questions. EM/F1: full SQuAD 2.0 dev (11,873), our out-of-domain `eval/shadow_set.json` (33) and `public_sample.json` (5) |
| Epochs | 2 (TODO: confirm) |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, 10 % linear warmup, AdamW, weight decay 0.01 |
| Batch size | 16 (TODO: confirm, incl. gradient accumulation) |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | TODO (GPU model, minutes; see `logs/<run>/train_summary.json`) |

Each training window is labelled with the token positions of the answer. It is labelled with
the CLS token (index 0) when the question is unanswerable or when the answer is not entirely
inside that window, and this teaches the model the no-answer score. The loss is the mean of the
start and end cross-entropies.

- **Loss curve:** `logs/roberta-base/loss.png` (raw log: `logs/roberta-base/train_log.jsonl`)
- **Validation results:** TODO, from `logs/roberta-base/eval_*.json`

| Dataset | EM | F1 | HasAns F1 | NoAns acc. |
|---|---|---|---|---|
| SQuAD 2.0 dev | TODO | TODO | TODO | TODO |
| shadow set (out-of-domain) | TODO | TODO | TODO | TODO |

## 3. No-answer threshold

- **Decision rule** (`qa/inference.py`):
  - The null score is `start_logit[CLS] + end_logit[CLS]`, taking the **minimum** over all windows
    of the context. A window that does not contain the answer always votes "no answer", so we keep
    the window where the model is least sure of that.
  - The span score is the best `start_logit[i] + end_logit[j]` with `i ≤ j`, both tokens inside the
    context and at most 30 tokens long, taking the **maximum** over all windows.
  - We return `null` when `null_score − span_score > threshold`.
- **Choosing the value:** `train/evaluate.py` computes `null_score − span_score` once per question and
  sweeps the threshold. It selects the value that maximises the mean F1 over the SQuAD 2.0 dev set and
  our shadow set. The hidden set has ~25 % unanswerable questions versus ~50 % in SQuAD dev, so
  tuning on SQuAD alone would favour null too much. Final value: TODO (plot: `logs/roberta-base/threshold.png`).
- **If too low:** more nulls, so correct answers become "wrong null" (0 points). In the limit we get the
  always-null baseline (0.25). **If too high:** we answer unanswerable questions (0 points), e.g. the
  adversarial `sample-04`. TODO: numbers at both extremes from the sweep.

## 4. Serving architecture

- **Framework and server:** FastAPI + uvicorn with **1 worker**. A second worker would load a second
  copy of the model and compete for the same CPU cores.
- **Model loading:** at startup a background thread loads the model, tokenizer and `qa_meta.json`
  (threshold, window size) from `QA_MODEL_DIR`, then runs one warm-up prediction. Until then
  `/health` returns `503 {"status": "loading"}`, and afterwards `200 {"status": "ok"}`.
- **Request flow** (`/predict`):
  1. The auth middleware checks the token.
  2. Pydantic validates the body (`extra="forbid"`, strict types, length limits). Invalid bodies get
     `400 {"error": "bad_request", ...}`.
  3. The question is clipped to 128 tokens and tokenized together with the context into windows of
     384 tokens with a 128-token overlap. Contexts up to 5,000 characters (~5 windows) are covered entirely.
  4. One batched forward pass runs on CPU, under a lock so there is only one inference at a time.
  5. The best span and the null score are computed over all windows and the threshold is applied.
  6. The span is cut from the **original** context string using the tokenizer's character offsets, so
     `context[start:start+len(answer)] == answer` holds by construction. It is also re-checked before
     responding.
  7. The response carries `score` = softmax probability of the returned output among {null} ∪ candidates.
- **Access control:** an HTTP middleware in `app/main.py` runs before routing and body parsing on
  `/info` and `/predict`. It requires `Authorization: Bearer <token>`, with the scheme and token
  case-sensitive, and compares the token in constant time (`hmac.compare_digest`). Failures return
  `401 {"error": "unauthorized"}` with `WWW-Authenticate: Bearer`. The token comes only from the
  environment variable `QA_API_TOKEN`, or from an untracked `.env` file. The service refuses to start
  without it. `/docs` and `/openapi.json` are disabled, and the server binds to 127.0.0.1, so it is
  reachable only through the tunnel. A git pre-commit hook blocks commits containing the token value,
  `.env` or weights.
- **Tunnel:** TODO (ngrok `ngrok http 8000`, backup Cloudflare `cloudflared tunnel --url http://localhost:8000`)
- **How to run** from a clean machine (commands and environment variables):

```bash
pip install -r requirements.txt
# download the fine-tuned model into models/roberta-base/  (TODO: release link)
export QA_API_TOKEN=...          # PowerShell: $env:QA_API_TOKEN = "..."
export QA_TEAM=team-XX
export QA_MODEL_DIR=models/roberta-base
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
ngrok http 8000
```

## 5. Results on the public sample set

TODO: fill from `python -m eval.harness --url <public-url>` (saved in `eval/results/`).

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` / `16 attention heads` | TODO | TODO | TODO |
| sample-02 | `3.5 days` | TODO | TODO | TODO |
| sample-03 | `null` | TODO | TODO | TODO |
| sample-04 | `null` | TODO | TODO | TODO |
| sample-05 | `50,257` / `50,257 tokens` | TODO | TODO | TODO |

Comment on any wrong answer: TODO. Watch for `sample-04`: the question is adversarial, asking
about BERT-base while the context is about GPT-3. Also watch `sample-05`: the answer is in the
last sentence, beyond the first 384-token window.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | TODO (`/info` → `params`; RoBERTa-base QA = 124,056,578) |
| Size on disk | TODO (`models/roberta-base/model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | TODO (harness `latency_ms_mean`, local and through the tunnel) |
| Laptop CPU and RAM | TODO |

## 7. Problems encountered and how you fixed them

1. **Problem:** a valid request with a long question (the contract allows 500 characters) crashed
   `/predict` with a 500 error during our contract tests. **Cause:** we tokenize with
   `truncation="only_second"`, which can only shorten the context. A question that is longer than the
   window by itself makes the tokenizer raise an error. **Fix:** `QAModel._clip_question` cuts the
   question to `min(128, max_seq_length // 2)` tokens using character offsets, before windowing.
   Covered by `tests/test_api.py::test_limits_accepted`. **Commit:** `d7a5137`
2. **Problem:** TODO, a real issue from training or serving (e.g. GPU memory, threshold, tunnel
   latency, a wrong answer type). **Cause:** … **Fix:** … **Commit:** `…`

## 8. Decoder investigation

TODO: base each point on our own failures, from `eval/results/challenge_set_*.json` and the wrong
answers in the shadow set. The categories below come from `eval/challenge_set.json`.

- **Yes/no questions** (e.g. "Is 403 Forbidden a server error?"): an extractive model can only copy
  a span, so "No" is impossible. TODO: what our model returned. A decoder (or encoder–decoder such as
  T5) *generates* "No". Cost: exact-match grading breaks with free text, it needs a constrained output
  or a classifier head, and it is slower.
- **Non-contiguous answers** (e.g. "Which three volcanoes form Los Volcanes National Park?", where
  "Santa Ana" appears far from "Izalco and Cerro Verde"): a single span cannot cover both. TODO: our
  output. A decoder can compose "Santa Ana, Izalco and Cerro Verde".
- **Arithmetic** (e.g. "How many years passed between the discovery and the UNESCO listing?" → 17).
  TODO.
- **Long contexts:** we need about 5 windows for 5,000 characters, and each window is scored separately.
  Decoders with long context windows read everything at once.
- **Cost:** TODO, with numbers. A small instruction-tuned decoder generates token by token, so a 5,000
  character prompt would take seconds per answer on our CPU, against our ~0.2–1.8 s. It is also larger
  on disk and may hallucinate text that is not in the context. Fine-tuning it within 1.5 weeks would
  require LoRA (Hu et al., 2021).

## 9. Known limitations

- **English only.** A question in Spanish would still return a span or null, but quality is not measured.
- **Answers longer than 30 tokens** are cut or not found (`max_answer_tokens`).
- **Yes/no, multi-span, arithmetic and summary questions** cannot be answered correctly. The service
  returns a span or null (Section 8).
- **Contexts over 5,000 characters** are rejected with 400, as the contract requires. Questions over
  128 tokens are clipped.
- **One request at a time.** Concurrent requests queue behind a lock, which is fine for the sequential
  harness but would add latency under load.
- **The threshold** was tuned on SQuAD dev plus a small hand-written set (33 items). It may not be
  optimal for the instructor's texts.
- **`score` is a softmax over candidates.** It is not calibrated and must not be read as a probability
  of being correct.
