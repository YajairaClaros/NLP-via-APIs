"""Extractive QA inference with no-answer detection (SQuAD 2.0 style).

This module is the single code path used by:
  * the web service (app/main.py), one request at a time, on CPU;
  * offline evaluation and threshold tuning (train/evaluate.py), in batches, on GPU.

Pipeline for one (question, context) pair:
  1. Tokenize "question + context" into overlapping windows of at most `max_seq_length`
     tokens (`doc_stride` tokens of overlap), so contexts longer than the model limit are
     fully covered.
  2. Run the encoder once over all windows -> start_logits and end_logits per token.
  3. Null score   = start_logit[CLS] + end_logit[CLS], taking the MINIMUM over windows
     (a window that does not contain the answer always "votes" for no-answer, so we
     keep the window where the model is least sure that there is no answer).
     Span score   = start_logit[i] + end_logit[j] for context tokens i <= j, j - i < max_answer_tokens,
     taking the MAXIMUM over all windows.
  4. Return null if (null_score - best_span_score) > null_threshold, else the best span.
  5. The span is cut from the ORIGINAL context with the tokenizer's character offsets,
     so context[start:start + len(answer)] == answer always holds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

META_FILE = "qa_meta.json"


@dataclass
class QAConfig:
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_tokens: int = 30
    n_best: int = 20
    null_threshold: float = 0.0
    max_question_tokens: int = 128   # longer questions are cut so the context always has room


@dataclass
class Prediction:
    answer: str | None          # returned answer (None = no answer)
    start: int | None           # character offset of `answer` in the context
    score: float                # confidence of the returned output, in [0, 1]
    null_score: float           # min over windows of start[CLS] + end[CLS]
    span_score: float           # best start[i] + end[j] over all windows (-inf if no candidate)
    best_span: str | None       # best non-null span, even when null is returned
    best_span_start: int | None
    num_windows: int

    @property
    def score_diff(self) -> float:
        """null_score - span_score. Null is returned when this is > threshold."""
        return self.null_score - self.span_score


def load_meta(model_dir: str | Path) -> dict:
    path = Path(model_dir) / META_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_meta(model_dir: str | Path, meta: dict) -> None:
    path = Path(model_dir) / META_FILE
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


class QAModel:
    """Fine-tuned encoder + tokenizer + post-processing settings."""

    def __init__(self, model, tokenizer, config: QAConfig, meta: dict | None = None, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.meta = meta or {}
        self.device = device
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required (character offsets are needed).")
        # Only pass the inputs this architecture accepts (DistilBERT/RoBERTa: no token_type_ids).
        self.input_names = [n for n in tokenizer.model_input_names if n != "offset_mapping"]

    @classmethod
    def load(cls, model_dir: str | Path, device: str | None = None, **overrides) -> "QAModel":
        model_dir = Path(model_dir)
        meta = load_meta(model_dir)
        cfg = QAConfig(**{k: meta[k] for k in QAConfig.__dataclass_fields__ if k in meta})
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        return cls(model, tokenizer, cfg, meta, device)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ encoding
    def _clip_question(self, question: str) -> str:
        """Cut a question to `max_question_tokens` tokens.

        truncation="only_second" can only shorten the context: if the question alone did not
        fit in max_seq_length the tokenizer would raise and the request would fail (500).
        """
        question = question.lstrip()   # leading spaces only waste tokens
        enc = self.tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
        limit = min(self.config.max_question_tokens, self.config.max_seq_length // 2)
        if len(enc["input_ids"]) <= limit:
            return question
        return question[: enc["offset_mapping"][limit - 1][1]]

    def encode(self, questions: list[str], contexts: list[str]):
        return self.tokenizer(
            [self._clip_question(q) for q in questions],
            contexts,
            truncation="only_second",          # never cut the question, only the context
            max_length=self.config.max_seq_length,
            stride=self.config.doc_stride,     # overlap between consecutive windows
            return_overflowing_tokens=True,    # one feature per window
            return_offsets_mapping=True,       # token -> (char_start, char_end) in the context
            padding="longest",
        )

    @torch.inference_mode()
    def _logits(self, enc, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        n = len(enc["input_ids"])
        starts, ends = [], []
        for i in range(0, n, batch_size):
            batch = {k: torch.tensor(enc[k][i:i + batch_size], device=self.device) for k in self.input_names}
            out = self.model(**batch)
            starts.append(out.start_logits.float().cpu().numpy())
            ends.append(out.end_logits.float().cpu().numpy())
        return np.concatenate(starts), np.concatenate(ends)

    # ------------------------------------------------------------------ prediction
    def predict(self, questions: list[str], contexts: list[str], batch_size: int = 64) -> list[Prediction]:
        enc = self.encode(questions, contexts)
        start_logits, end_logits = self._logits(enc, batch_size)
        sample_map = enc["overflow_to_sample_mapping"]

        features_per_example: dict[int, list[int]] = {}
        for feat_idx, ex_idx in enumerate(sample_map):
            features_per_example.setdefault(ex_idx, []).append(feat_idx)

        preds = []
        for ex_idx, context in enumerate(contexts):
            feats = features_per_example.get(ex_idx, [])
            preds.append(self._postprocess(enc, feats, start_logits, end_logits, context))
        return preds

    def predict_one(self, question: str, context: str) -> Prediction:
        return self.predict([question], [context])[0]

    def _postprocess(self, enc, feats, start_logits, end_logits, context) -> Prediction:
        cfg = self.config
        null_score = math.inf
        # (char_start, char_end) -> best logit sum for that exact span
        candidates: dict[tuple[int, int], float] = {}

        for f in feats:
            s_log, e_log = start_logits[f], end_logits[f]
            offsets = enc["offset_mapping"][f]
            seq_ids = enc.sequence_ids(f)

            # The [CLS]/<s> token is at index 0 for BERT, DistilBERT and RoBERTa.
            null_score = min(null_score, float(s_log[0] + e_log[0]))

            # Only tokens that belong to the context (sequence id 1) can start or end a span.
            is_ctx = np.array([sid == 1 for sid in seq_ids] + [False] * (len(s_log) - len(seq_ids)))
            if not is_ctx.any():
                continue
            s_masked = np.where(is_ctx, s_log, -np.inf)
            e_masked = np.where(is_ctx, e_log, -np.inf)
            top_s = np.argsort(s_masked)[::-1][: cfg.n_best]
            top_e = np.argsort(e_masked)[::-1][: cfg.n_best]

            for i in top_s:
                if not is_ctx[i]:
                    continue
                for j in top_e:
                    if not is_ctx[j] or j < i or j - i + 1 > cfg.max_answer_tokens:
                        continue
                    c_start, c_end = offsets[i][0], offsets[j][1]
                    # Trim whitespace so the span is clean; offsets stay exact.
                    text = context[c_start:c_end]
                    lead = len(text) - len(text.lstrip())
                    text = text.strip()
                    if not text:
                        continue
                    key = (c_start + lead, c_start + lead + len(text))
                    score = float(s_log[i] + e_log[j])
                    if score > candidates.get(key, -math.inf):
                        candidates[key] = score

        if not feats:
            null_score = 0.0

        if not candidates:
            return Prediction(None, None, 1.0, null_score, -math.inf, None, None, len(feats))

        (b_start, b_end), span_score = max(candidates.items(), key=lambda kv: kv[1])
        best_text = context[b_start:b_end]

        # Confidence: softmax over {no-answer} U {all candidate spans}.
        logits = np.array([null_score] + list(candidates.values()), dtype=np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        p_null, p_span = float(probs[0]), float(probs[1:].max())

        if null_score - span_score > cfg.null_threshold:
            return Prediction(None, None, p_null, null_score, span_score, best_text, b_start, len(feats))
        return Prediction(best_text, b_start, p_span, null_score, span_score, best_text, b_start, len(feats))


def to_jsonable(pred: Prediction) -> dict:
    d = asdict(pred)
    d["score_diff"] = pred.score_diff
    return d
