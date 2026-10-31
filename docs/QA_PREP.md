# Q&A preparation (individual, 20 points)

One random question per student, about **our** code, no AI, repo on screen allowed.
A level-4 answer is correct, **specific to this repository** and covers the key points.
Practise by answering each question out loud while pointing at the file and line.

## Model and training — `train/train.py`

1. **Why RoBERTa-base and not DistilBERT?** Reference = DistilBERT at 0.70 = full points; we wanted margin for out-of-domain texts. Cost: 124M vs 66M params, ~2× CPU time; still ≤ ~1.8 s worst case vs 5 s timeout.
2. **How is an unanswerable question labelled?** `make_features_fn`: start = end = 0 (CLS). Also when the answer is not fully inside that window.
3. **What do `max_seq_length` and `doc_stride` do?** Window of 384 tokens; consecutive windows overlap by 128 tokens so an answer cut at a border appears whole in the next window.
4. **Why `truncation="only_second"`?** Only the context (2nd sequence) is split into windows; the question is repeated in every window.
5. **What loss is minimised?** Cross-entropy on start position + cross-entropy on end position, averaged (inside `AutoModelForQuestionAnswering`).
6. **How do you show you did not start from a SQuAD checkpoint?** `--model` is a base id, `FORBIDDEN_HINTS` check, `qa_meta.json["base_model"]`, loss starts high (≈ 5–6, i.e. a random guess over a few hundred positions) in `logs/<run>/train_log.jsonl`.
7. **Warmup / scheduler / precision?** 10 % linear warmup then linear decay; bf16/fp16 picked automatically in `pick_precision`.

## Post-processing and threshold — `qa/inference.py`, `train/evaluate.py`

8. **How does the service decide to return null?** `null_score - span_score > null_threshold`; null = min over windows of CLS start+end; span = max over windows of start[i]+end[j].
9. **Why the minimum null score over windows?** Windows without the answer always look "unanswerable"; the window that contains the answer is the one with the lowest null score.
10. **How was the threshold chosen?** Sweep in `evaluate.py::sweep`, maximise mean F1 over SQuAD dev + shadow set; written to `qa_meta.json`.
11. **What if it is too high / too low?** Too high → answer unanswerables; too low → wrong nulls; extreme low = always-null baseline 0.25.
12. **How is `start` computed and why is it always correct?** Character offsets of the start/end tokens (`offset_mapping`) on the original context; whitespace trimmed with offset adjusted; invariant re-checked in `app/main.py`.
13. **How are contexts longer than 512 tokens handled?** Several windows in one batch; spans from all windows compete.
14. **What is `score`?** Softmax over {null} ∪ candidate spans; probability of the returned output. Not graded, not calibrated.

## Service — `app/main.py`, `app/settings.py`

15. **Where is the token checked?** `bearer_auth` middleware, before routing/body parsing, for `/info` and `/predict`; `hmac.compare_digest`; 401 + `WWW-Authenticate: Bearer`.
16. **Where does the token come from?** `QA_API_TOKEN` env var (or untracked `.env`), read in `get_settings`; the app refuses to start without it. Never in code; `.gitignore` + pre-commit hook.
17. **Why is `/health` 503 at startup?** Model loads in a background thread (`ModelState.load`); 200 only when `qa` is set.
18. **What happens with an invalid body?** Pydantic `PredictRequest` (strict, extra=forbid, lengths) → `RequestValidationError` handler → 400 `bad_request` with detail.
19. **Why 1 worker and a lock?** One model copy in memory; parallel inferences would fight for the same CPU cores.
20. **Why is `predict` a `def` and not `async def`?** FastAPI runs it in a thread pool so the event loop is not blocked by the model.
21. **Why bind to 127.0.0.1?** Only the tunnel on the same laptop must reach it, not the classroom network.

## Evaluation and tunnel — `eval/`, `scripts/`

22. **What does `eval/harness.py` check?** Health, 401 cases, /info fields, validation errors, id echo, offset invariant, 5 s timeout, 150 s cap; projects points with the grading formula.
23. **What is the shadow set and why?** 33 hand-written out-of-domain questions, ~25 % unanswerable, adversarial nulls; used for threshold selection and error analysis.
24. **Which tunnel and why?** ngrok (fixed domain) + Cloudflare Quick Tunnel as backup; latency measured at the dry run.
