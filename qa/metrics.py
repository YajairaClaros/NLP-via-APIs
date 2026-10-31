"""SQuAD 2.0 metrics (Exact Match and F1), following the official evaluation script.

Unanswerable items have gold answers == [] and are correct only when the prediction is
null (empty string after normalization).
"""

from __future__ import annotations

import collections
import re
import string


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def exact_match(prediction: str | None, gold: str) -> float:
    return float(normalize_answer(prediction or "") == normalize_answer(gold))


def f1(prediction: str | None, gold: str) -> float:
    pred_toks, gold_toks = _tokens(prediction or ""), _tokens(gold)
    if not pred_toks or not gold_toks:
        # If either is no-answer, F1 is 1 if both are no-answer, else 0.
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred_toks), same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_item(prediction: str | None, gold_answers: list[str]) -> tuple[float, float]:
    """Return (EM, F1) of one prediction against all gold answers (max over golds)."""
    golds = gold_answers or [""]
    return max(exact_match(prediction, g) for g in golds), max(f1(prediction, g) for g in golds)
