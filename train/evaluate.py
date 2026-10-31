"""Evaluate a fine-tuned model and choose the no-answer threshold.

The model returns null when  (null_score - best_span_score) > threshold.
This script computes, for every example, that difference and the best span, then sweeps
the threshold and reports EM / F1 (overall, HasAns, NoAns) for each value.

    # SQuAD 2.0 dev (11,873 questions) + our own out-of-domain set
    python -m train.evaluate --model_dir models/roberta-base --data squad_v2 eval/shadow_set.json

    # write the best threshold (maximising mean F1 across the datasets) into qa_meta.json
    python -m train.evaluate --model_dir models/roberta-base --data squad_v2 eval/shadow_set.json --write_threshold

    # or set a value by hand
    python -m train.evaluate --model_dir models/roberta-base --data eval/shadow_set.json --set_threshold 1.5

Outputs in logs/<run>/: eval_<dataset>.json (metrics + per-example predictions) and threshold.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qa.inference import QAModel, load_meta, save_meta
from qa.metrics import score_item

ROOT = Path(__file__).resolve().parents[1]


def load_items(spec: str, max_samples: int | None) -> tuple[str, list[dict]]:
    """Return (name, items) with items = {id, question, context, answers: [str]}."""
    if spec == "squad_v2":
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", split="validation")
        items = [{"id": r["id"], "question": r["question"], "context": r["context"],
                  "answers": list(r["answers"]["text"])} for r in ds]
        name = "squad_v2_dev"
    else:
        data = json.loads(Path(spec).read_text(encoding="utf-8"))
        items = [{"id": it["id"], "question": it["question"], "context": it["context"],
                  "answers": [a["text"] for a in it.get("answers", [])]} for it in data["items"]]
        name = Path(spec).stem
    if max_samples:
        items = items[:max_samples]
    return name, items


def run_model(qa: QAModel, items: list[dict], batch_examples: int) -> list[dict]:
    rows = []
    for i in range(0, len(items), batch_examples):
        chunk = items[i:i + batch_examples]
        preds = qa.predict([c["question"] for c in chunk], [c["context"] for c in chunk])
        for it, p in zip(chunk, preds):
            em_span, f1_span = score_item(p.best_span, it["answers"]) if p.best_span else (0.0, 0.0)
            rows.append({
                "id": it["id"], "question": it["question"], "gold": it["answers"],
                "best_span": p.best_span, "best_span_start": p.best_span_start,
                "diff": p.score_diff, "null_score": p.null_score, "span_score": p.span_score,
                "em_span": em_span, "f1_span": f1_span, "is_impossible": len(it["answers"]) == 0,
                "windows": p.num_windows,
            })
        print(f"  {min(i + batch_examples, len(items))}/{len(items)}", end="\r")
    print()
    return rows


def sweep(rows: list[dict], thresholds: np.ndarray) -> dict[str, np.ndarray]:
    diff = np.array([r["diff"] for r in rows])
    imp = np.array([r["is_impossible"] for r in rows])
    em_s = np.array([r["em_span"] for r in rows])
    f1_s = np.array([r["f1_span"] for r in rows])
    pred_null = diff[:, None] > thresholds[None, :]          # shape (items, thresholds)
    null_ok = imp[:, None].astype(float)
    em = np.where(pred_null, null_ok, em_s[:, None])
    f1 = np.where(pred_null, null_ok, f1_s[:, None])
    out = {"em": em.mean(0), "f1": f1.mean(0), "null_rate": pred_null.mean(0)}
    if (~imp).any():
        out["has_ans_em"], out["has_ans_f1"] = em[~imp].mean(0), f1[~imp].mean(0)
    if imp.any():
        out["no_ans_acc"] = em[imp].mean(0)
    return out


def metrics_at(curves: dict, idx: int) -> dict:
    return {k: round(float(v[idx]) * 100, 2) for k, v in curves.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data", nargs="+", default=["squad_v2"], help="squad_v2 and/or json files in public_sample format")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_examples", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--write_threshold", action="store_true", help="Save the best threshold into qa_meta.json")
    p.add_argument("--set_threshold", type=float, default=None, help="Save this threshold into qa_meta.json")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    qa = QAModel.load(model_dir, device=args.device)
    current = qa.config.null_threshold
    run = model_dir.name
    log_dir = ROOT / "logs" / run
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for spec in args.data:
        name, items = load_items(spec, args.max_samples)
        print(f"[{name}] {len(items)} items")
        t0 = time.time()
        rows = run_model(qa, items, args.batch_examples)
        datasets.append((name, rows, time.time() - t0))

    finite = np.concatenate([[r["diff"] for r in rows if np.isfinite(r["diff"])] for _, rows, _ in datasets])
    lo, hi = np.percentile(finite, [1, 99]) if len(finite) else (-5.0, 5.0)
    thresholds = np.unique(np.concatenate([np.linspace(min(lo, -1), max(hi, 1), 401), [0.0, current]]))

    curves = {name: sweep(rows, thresholds) for name, rows, _ in datasets}
    mean_f1 = np.mean([c["f1"] for c in curves.values()], axis=0)
    best_idx = int(np.argmax(mean_f1))
    best_t = float(thresholds[best_idx])
    cur_idx = int(np.argmin(np.abs(thresholds - current)))
    zero_idx = int(np.argmin(np.abs(thresholds)))

    for name, rows, secs in datasets:
        c = curves[name]
        report = {
            "dataset": name, "n": len(rows), "model_dir": str(model_dir), "seconds": round(secs, 1),
            "best_threshold_for_this_dataset": float(thresholds[int(np.argmax(c["f1"]))]),
            "at_best_threshold_for_this_dataset": metrics_at(c, int(np.argmax(c["f1"]))),
            "selected_threshold": best_t, "at_selected_threshold": metrics_at(c, best_idx),
            "at_threshold_0": metrics_at(c, zero_idx),
            "current_threshold": current, "at_current_threshold": metrics_at(c, cur_idx),
            "predictions": [{**r, "pred_null_at_selected": bool(r["diff"] > best_t)} for r in rows],
        }
        out = log_dir / f"eval_{name}.json"
        out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        print(f"\n[{name}] -> {out}")
        for key in ("at_threshold_0", "at_best_threshold_for_this_dataset", "at_selected_threshold"):
            print(f"  {key:38s} {report[key]}")

    print(f"\nSelected threshold (max mean F1 over {len(datasets)} dataset(s)): {best_t:.3f}   (current: {current})")
    plot(thresholds, curves, best_t, log_dir / "threshold.png")

    if args.set_threshold is not None or args.write_threshold:
        value = args.set_threshold if args.set_threshold is not None else best_t
        meta = load_meta(model_dir)
        meta["null_threshold"] = round(float(value), 4)
        meta["threshold_selected_on"] = [n for n, _, _ in datasets]
        save_meta(model_dir, meta)
        print(f"Wrote null_threshold={meta['null_threshold']} to {model_dir / 'qa_meta.json'}")


def plot(thresholds, curves, best_t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for name, c in curves.items():
        ax.plot(thresholds, c["f1"] * 100, label=f"F1 {name}")
        ax.plot(thresholds, c["null_rate"] * 100, "--", alpha=0.6, label=f"% null {name}")
    ax.axvline(best_t, color="k", linestyle=":", label=f"selected = {best_t:.2f}")
    ax.set_xlabel("threshold on (null_score - best_span_score)   [higher -> fewer nulls]")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Plot: {out}")


if __name__ == "__main__":
    main()
