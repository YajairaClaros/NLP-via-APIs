"""Training labels must point at the gold answer (or CLS when it is absent from the window)."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from train.train import make_features_fn

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "docs/brief/public_sample.json").read_text())["items"]


def test_labels_recover_answer(tiny_model_dir):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    batch = {
        "question": [it["question"] for it in SAMPLE],
        "context": [it["context"] for it in SAMPLE],
        "answers": [{"text": [a["text"] for a in it["answers"][:1]],
                     "answer_start": [a["start"] for a in it["answers"][:1]]} for it in SAMPLE],
    }
    feats = make_features_fn(tok, max_len=128, stride=32)(batch)
    # Re-tokenize to get offsets and the window -> example mapping.
    enc = tok([q.lstrip() for q in batch["question"]], batch["context"], truncation="only_second", max_length=128,
              stride=32, return_overflowing_tokens=True, return_offsets_mapping=True)
    found = set()
    for i, (s, e) in enumerate(zip(feats["start_positions"], feats["end_positions"])):
        ex = enc["overflow_to_sample_mapping"][i]
        item = SAMPLE[ex]
        if s == 0 and e == 0:
            continue
        text = item["context"][enc["offset_mapping"][i][s][0]: enc["offset_mapping"][i][e][1]].strip()
        assert text == item["answers"][0]["text"]
        found.add(item["id"])
    answerable = {it["id"] for it in SAMPLE if not it["is_impossible"]}
    assert found == answerable   # every answerable item is labelled in at least one window


def test_train_script_smoke(tmp_path, tiny_bert_dir):
    """Two optimizer steps of the real training script on CPU."""
    import sys
    from train import train as train_mod

    data = tmp_path / "tiny.jsonl"
    with data.open("w") as f:
        for it in SAMPLE:
            a = it["answers"][:1]
            f.write(json.dumps({"id": it["id"], "title": "t", "context": it["context"], "question": it["question"],
                                "answers": {"text": [x["text"] for x in a], "answer_start": [x["start"] for x in a]}}) + "\n")
    sys.argv = ["train", "--model", str(tiny_bert_dir), "--run_name", "smoke", "--train_file", str(data),
                "--max_steps", "2", "--batch_size", "4", "--eval_steps", "1", "--logging_steps", "1",
                "--max_seq_length", "128", "--doc_stride", "32", "--output_root", str(tmp_path),
                "--allow_any_checkpoint"]
    train_mod.main()
    assert (tmp_path / "models/smoke/qa_meta.json").exists()
    assert (tmp_path / "logs/smoke/loss.png").exists()
    rows = (tmp_path / "logs/smoke/train_log.jsonl").read_text().splitlines()
    assert any("loss" in json.loads(r) for r in rows)
