"""Build eval/shadow_set.json and eval/challenge_set.json from eval/source/*.json.

Answers in the source files are written as plain text; this script finds their character
offsets in the context (and fails loudly if an answer is not found verbatim), producing
files in exactly the same format as docs/brief/public_sample.json.

    python -m eval.build_sets
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def locate(context: str, text: str, item_id: str) -> int:
    start = context.find(text)
    if start < 0:
        raise SystemExit(f"{item_id}: answer {text!r} not found verbatim in its context")
    if context.find(text, start + 1) >= 0:
        print(f"warning: {item_id}: {text!r} occurs more than once; using the first occurrence")
    return start


def build_shadow():
    src = json.loads((HERE / "source" / "shadow_source.json").read_text(encoding="utf-8"))
    items = []
    for it in src["items"]:
        ctx = src["contexts"][it["context"]]
        answers = [{"text": a, "start": locate(ctx, a, it["id"])} for a in it["answers"]]
        items.append({"id": it["id"], "context": ctx, "question": it["question"],
                      "is_impossible": not answers, "answers": answers})
    out = {"version": "1.0", "description": src["description"], "items": items}
    (HERE / "shadow_set.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n_imp = sum(i["is_impossible"] for i in items)
    print(f"shadow_set.json: {len(items)} items, {n_imp} unanswerable ({n_imp / len(items):.0%})")


def build_challenge():
    src = json.loads((HERE / "source" / "challenge_source.json").read_text(encoding="utf-8"))
    items = [{"id": it["id"], "context": src["contexts"][it["context"]], "question": it["question"],
              "category": it["category"], "expected_free_text": it["expected_free_text"]} for it in src["items"]]
    out = {"version": "1.0", "description": src["description"], "items": items}
    (HERE / "challenge_set.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"challenge_set.json: {len(items)} items")


if __name__ == "__main__":
    build_shadow()
    build_challenge()
