"""Plot the training / validation loss curve from logs/<run>/train_log.jsonl.

    python -m train.plot_loss logs/roberta-base/train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_loss(log_path: Path, out_path: Path, title: str = "Loss") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(ln) for ln in Path(log_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    dev = [(r["step"], r["eval_loss"]) for r in rows if "eval_loss" in r]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if train:
        ax.plot(*zip(*train), label="train loss", linewidth=1.2)
    if dev:
        ax.plot(*zip(*dev), "o-", label="validation loss", linewidth=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (start + end) / 2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    print(plot_loss(a.log, a.out or a.log.with_name("loss.png"), title=a.log.parent.name))
