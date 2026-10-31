"""Fine-tune a pretrained BASE encoder (DistilBERT / BERT / RoBERTa) for SQuAD 2.0.

Rule V.1: we start from a base checkpoint (e.g. `roberta-base`), never from a checkpoint
already fine-tuned on SQuAD. The QA head (start/end classifier) is newly initialised.

Example (GPU):
    python -m train.train --model roberta-base --epochs 2 --lr 3e-5 --batch_size 16

Outputs:
    checkpoints/<run>/            intermediate checkpoints (git-ignored, for resuming)
    models/<run>/                 final model + tokenizer + qa_meta.json (git-ignored)
    logs/<run>/train_log.jsonl    every logged step (loss, lr, eval_loss)  -> committed
    logs/<run>/loss.png           loss curve                               -> committed
    logs/<run>/train_summary.json args, hardware, runtime                  -> committed
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from qa.inference import save_meta
from train.plot_loss import plot_loss

ROOT = Path(__file__).resolve().parents[1]

# Checkpoints that are already fine-tuned for QA are forbidden (rule V.1).
FORBIDDEN_HINTS = ("squad", "qa", "question")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="roberta-base",
                   help="Base checkpoint: distilbert-base-uncased | bert-base-uncased | roberta-base")
    p.add_argument("--run_name", default=None, help="Folder name under logs/, models/, checkpoints/")
    p.add_argument("--train_file", default=None, help="Optional local SQuAD-format json/jsonl instead of the Hub dataset")
    p.add_argument("--validation_file", default=None)
    p.add_argument("--max_train_samples", type=int, default=None, help="Subsample the train set (None = all ~130k)")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Dev examples used for the eval-loss curve")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (keep 0 on Windows)")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in checkpoints/<run>")
    p.add_argument("--max_steps", type=int, default=-1, help="Debug: stop after N steps")
    p.add_argument("--output_root", default=str(ROOT), help="Where logs/, models/, checkpoints/ are created")
    p.add_argument("--allow_any_checkpoint", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------- data
def load_squad(args):
    if args.train_file:
        files = {"train": args.train_file, "validation": args.validation_file or args.train_file}
        ds = load_dataset("json", data_files=files)
    else:
        ds = load_dataset("rajpurkar/squad_v2")
    train, dev = ds["train"], ds["validation"]
    if args.max_train_samples:
        train = train.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        dev = dev.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(dev))))
    return train, dev


def make_features_fn(tokenizer, max_len: int, stride: int):
    """Turn (question, context, answer) examples into labelled windows.

    Each window gets start_positions / end_positions = token indices of the answer, or
    0 (the CLS token) when the question is unanswerable OR the answer is not fully
    inside this window. That is how the model learns the "no answer" score.
    """

    def prepare(examples):
        enc = tokenizer(
            [q.lstrip() for q in examples["question"]],
            examples["context"],
            truncation="only_second",
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets_all = enc.pop("offset_mapping")
        starts, ends = [], []

        for i, offsets in enumerate(offsets_all):
            seq_ids = enc.sequence_ids(i)
            answers = examples["answers"][sample_map[i]]
            cls = 0
            if len(answers["answer_start"]) == 0:          # unanswerable question
                starts.append(cls); ends.append(cls); continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            ctx_start = seq_ids.index(1)
            ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

            if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
                starts.append(cls); ends.append(cls); continue  # answer not in this window

            ts = ctx_start
            while ts <= ctx_end and offsets[ts][0] <= start_char:
                ts += 1
            te = ctx_end
            while te >= ctx_start and offsets[te][1] >= end_char:
                te -= 1
            starts.append(ts - 1)
            ends.append(te + 1)

        enc["start_positions"] = starts
        enc["end_positions"] = ends
        return enc

    return prepare


# ---------------------------------------------------------------------------- logging
class JsonlLogger(TrainerCallback):
    """Append every Trainer log entry (loss, lr, eval_loss...) to a .jsonl file."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            row = {"step": state.global_step, "epoch": state.epoch, "time": time.time(), **logs}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def pick_precision(choice: str) -> dict:
    if not torch.cuda.is_available() or choice == "fp32":
        return {}
    if choice == "bf16" or (choice == "auto" and torch.cuda.is_bf16_supported()):
        return {"bf16": True}
    return {"fp16": True}


# ---------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if not args.allow_any_checkpoint and any(h in args.model.lower() for h in FORBIDDEN_HINTS):
        raise SystemExit(f"'{args.model}' looks like a QA-fine-tuned checkpoint. Use a BASE checkpoint (rule V.1).")

    set_seed(args.seed)
    run = args.run_name or args.model.split("/")[-1]
    out = Path(args.output_root)
    log_dir, ckpt_dir, model_dir = out / "logs" / run, out / "checkpoints" / run, out / "models" / run

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    train_raw, dev_raw = load_squad(args)

    prepare = make_features_fn(tokenizer, args.max_seq_length, args.doc_stride)
    train_ds = train_raw.map(prepare, batched=True, remove_columns=train_raw.column_names, desc="Tokenizing train")
    dev_ds = dev_raw.map(prepare, batched=True, remove_columns=dev_raw.column_names, desc="Tokenizing dev")
    print(f"train: {len(train_raw)} examples -> {len(train_ds)} windows | dev(loss): {len(dev_raw)} -> {len(dev_ds)}")

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_ratio,          # float in [0, 1) = ratio of total steps
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
        **pick_precision(args.precision),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),  # dynamic padding
        processing_class=tokenizer,
        callbacks=[JsonlLogger(log_dir / "train_log.jsonl")],
    )

    t0 = time.time()
    resume = args.resume and ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    final_eval = trainer.evaluate()
    elapsed = time.time() - t0

    # Final model for serving (weights stay out of git).
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_meta(model_dir, {
        "base_model": args.model,
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
        "max_answer_tokens": 30,
        "null_threshold": 0.0,   # tuned afterwards with train/evaluate.py --write_threshold
        "train_examples": len(train_raw),
        "train_windows": len(train_ds),
    })

    summary = {
        "args": vars(args),
        "train_examples": len(train_raw),
        "train_windows": len(train_ds),
        "train_runtime_s": round(elapsed, 1),
        "train_metrics": result.metrics,
        "final_eval": final_eval,
        "params": sum(p.numel() for p in model.parameters()),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "model_dir": str(model_dir),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_loss(log_dir / "train_log.jsonl", log_dir / "loss.png", title=f"{args.model} on SQuAD 2.0")
    print(f"Done in {elapsed / 60:.1f} min. Model: {model_dir}  Logs: {log_dir}")


if __name__ == "__main__":
    main()
