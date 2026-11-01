"""Print every number REPORT.md needs, from the files on this machine.

    python scripts/report_numbers.py                      # all sections
    python scripts/report_numbers.py --skip_challenge     # skip running the model

Reads logs/<run>/train_summary.json and logs/<run>/eval_*.json, measures the model on disk,
reads CPU/RAM, and runs the model on eval/challenge_set.json (decoder investigation).
Nothing is sent anywhere; it only prints.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def section(title: str):
    print(f"\n=== {title} ===")


def train_info(log_dir: Path):
    section("2. Training (train_summary.json)")
    s = load(log_dir / "train_summary.json")
    if not s:
        print("  train_summary.json not found")
        return
    a = s.get("args", {})
    print(f"  model: {a.get('model')}")
    print(f"  train examples: {s.get('train_examples')}  ->  windows: {s.get('train_windows')}  (max_train_samples={a.get('max_train_samples')})")
    print(f"  epochs: {a.get('epochs')}  lr: {a.get('lr')}  batch_size: {a.get('batch_size')}  grad_accum: {a.get('grad_accum')}  warmup_ratio: {a.get('warmup_ratio')}")
    print(f"  max_seq_length/doc_stride: {a.get('max_seq_length')}/{a.get('doc_stride')}  precision: {a.get('precision')}")
    print(f"  training time: {s.get('train_runtime_s', 0) / 60:.1f} min   gpu: {s.get('hardware', {}).get('gpu')}")
    print(f"  final eval loss: {s.get('final_eval', {}).get('eval_loss')}")
    rows = [json.loads(l) for l in (log_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (log_dir / "train_log.jsonl").exists() else []
    losses = [r["loss"] for r in rows if "loss" in r]
    if losses:
        print(f"  train loss: first {losses[0]:.3f}  last {losses[-1]:.3f}")


def metrics_at(preds, t):
    n = len(preds)
    em = f1 = nulls = 0.0
    for p in preds:
        is_null = p["diff"] > t
        nulls += is_null
        if is_null:
            em += p["is_impossible"]; f1 += p["is_impossible"]
        else:
            em += p["em_span"]; f1 += p["f1_span"]
    return round(100 * em / n, 1), round(100 * f1 / n, 1), round(100 * nulls / n, 1)


def eval_info(log_dir: Path):
    section("2-3. Validation and threshold (eval_*.json)")
    files = sorted(log_dir.glob("eval_*.json"))
    if not files:
        print("  no eval_*.json found")
        return
    for f in files:
        r = load(f)
        sel = r.get("at_selected_threshold", {})
        print(f"\n  [{r['dataset']}] n={r['n']}  selected threshold={r.get('selected_threshold'):.3f}")
        print(f"    at selected: EM {sel.get('em')}  F1 {sel.get('f1')}  HasAns F1 {sel.get('has_ans_f1')}  "
              f"NoAns acc {sel.get('no_ans_acc')}  % null {sel.get('null_rate')}")
        z = r.get("at_threshold_0", {})
        print(f"    at threshold 0: EM {z.get('em')}  F1 {z.get('f1')}  % null {z.get('null_rate')}")
        preds = r.get("predictions", [])
        if preds:
            for t in (-6, -4, 0, 4, 6):
                em, f1, nr = metrics_at(preds, t)
                print(f"    threshold {t:+d}: EM {em}  F1 {f1}  % null {nr}")
            always_null = round(100 * sum(p["is_impossible"] for p in preds) / len(preds), 1)
            print(f"    always-null baseline: EM/F1 {always_null}")
            wrong = [p for p in preds if (p["diff"] > r["selected_threshold"]) != p["is_impossible"]
                     or (not p["is_impossible"] and p["diff"] <= r["selected_threshold"] and p["em_span"] < 1)]
            if r["n"] <= 60:
                for p in wrong:
                    got = None if p["diff"] > r["selected_threshold"] else p["best_span"]
                    print(f"    wrong/partial: {p['id']}: {p['question']!r} -> {got!r}  gold={p['gold']}")


def size_info(model_dir: Path):
    section("6. Model size")
    w = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    for f in w:
        print(f"  {f.name}: {f.stat().st_size / 1e6:.0f} MB")
    meta = load(model_dir / "qa_meta.json") or {}
    print(f"  qa_meta: base_model={meta.get('base_model')}  null_threshold={meta.get('null_threshold')}  "
          f"selected_on={meta.get('threshold_selected_on')}")


def machine_info():
    section("6. Laptop CPU and RAM")
    cpu = platform.processor()
    ram_gb = None
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        try:
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM(); m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            ram_gb = m.ullTotalPhys / 2**30
        except Exception:
            pass
    print(f"  CPU: {cpu}   logical cores: {os.cpu_count()}   RAM: {f'{ram_gb:.0f} GB' if ram_gb else 'unknown'}")


def challenge(model_dir: Path):
    section("8. Decoder investigation (challenge_set.json, run locally on CPU)")
    from qa.inference import QAModel
    qa = QAModel.load(model_dir, device="cpu")
    print(f"  params: {qa.num_parameters:,}")
    items = load(ROOT / "eval" / "challenge_set.json")["items"]
    times = []
    for it in items:
        t0 = time.perf_counter()
        p = qa.predict_one(it["question"], it["context"])
        times.append(time.perf_counter() - t0)
        print(f"  [{it['category']}] {it['question']}\n      model: {p.answer!r} (score {p.score:.2f})   best span if forced: {p.best_span!r}   expected: {it['expected_free_text']}")
    long_ctx = max(items, key=lambda i: len(i["context"]))
    t0 = time.perf_counter(); p = qa.predict_one(long_ctx["question"], long_ctx["context"]); dt = time.perf_counter() - t0
    print(f"  CPU time per question: mean {1000 * sum(times) / len(times):.0f} ms; "
          f"longest context ({len(long_ctx['context'])} chars, {p.num_windows} windows): {1000 * dt:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roberta-base")
    ap.add_argument("--skip_challenge", action="store_true")
    a = ap.parse_args()
    log_dir, model_dir = ROOT / "logs" / a.run, ROOT / "models" / a.run
    train_info(log_dir)
    eval_info(log_dir)
    size_info(model_dir)
    machine_info()
    if not a.skip_challenge:
        challenge(model_dir)


if __name__ == "__main__":
    main()
