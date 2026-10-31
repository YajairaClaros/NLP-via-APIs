"""Post-processing tests with real tokenizers (WordPiece and byte-level BPE) and random weights."""

import json
import math
from pathlib import Path

import pytest

from qa.inference import QAModel
from qa.metrics import score_item

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "docs/brief/public_sample.json").read_text())["items"]


@pytest.fixture(scope="module")
def qa(tiny_model_dir):
    return QAModel.load(tiny_model_dir, device="cpu")


@pytest.mark.parametrize("threshold", [-math.inf, 0.0, math.inf])
def test_offsets_always_match(qa, threshold):
    qa.config.null_threshold = threshold
    for it in SAMPLE:
        p = qa.predict_one(it["question"], it["context"])
        assert (p.answer is None) == (p.start is None)
        if p.answer is not None:
            assert it["context"][p.start:p.start + len(p.answer)] == p.answer
            assert p.answer == p.answer.strip() and p.answer
        assert 0.0 <= p.score <= 1.0
    qa.config.null_threshold = 0.0


def test_threshold_extremes(qa):
    it = SAMPLE[0]
    qa.config.null_threshold = -math.inf
    assert qa.predict_one(it["question"], it["context"]).answer is None      # always null
    qa.config.null_threshold = math.inf
    assert qa.predict_one(it["question"], it["context"]).answer is not None  # never null
    qa.config.null_threshold = 0.0


def test_long_context_uses_several_windows(qa):
    long_item = next(it for it in SAMPLE if it["id"] == "sample-05")
    p = qa.predict_one(long_item["question"], long_item["context"])
    assert p.num_windows > 1


def test_batch_equals_single(qa):
    qs, cs = [it["question"] for it in SAMPLE], [it["context"] for it in SAMPLE]
    batch = qa.predict(qs, cs)
    for it, b in zip(SAMPLE, batch):
        s = qa.predict_one(it["question"], it["context"])
        assert (s.answer, s.start) == (b.answer, b.start)
        assert math.isclose(s.null_score, b.null_score, rel_tol=1e-4, abs_tol=1e-4)


def test_metrics():
    assert score_item("16 attention heads", ["16", "16 attention heads"]) == (1.0, 1.0)
    assert score_item("The 3.5 days.", ["3.5 days"]) == (1.0, 1.0)   # articles/punctuation ignored
    assert score_item(None, []) == (1.0, 1.0)                          # correct null
    assert score_item("something", []) == (0.0, 0.0)                   # answered an unanswerable
    assert score_item(None, ["16"]) == (0.0, 0.0)                      # wrong null
    em, f1 = score_item("about 422 km", ["422 km"])
    assert em == 0.0 and 0 < f1 < 1
