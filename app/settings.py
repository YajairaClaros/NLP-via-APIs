"""Service settings, read from environment variables (optionally from an untracked .env file).

QA_API_TOKEN   required. Bearer token for /info and /predict. NEVER hard-code or commit it.
QA_TEAM        team id as registered with the instructor (no spaces), e.g. team-03.
QA_MODEL_DIR   folder with the fine-tuned model, tokenizer and qa_meta.json.
QA_NUM_THREADS optional, number of CPU threads for PyTorch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Real environment variables win over values in .env (override=False).
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    api_token: str
    team: str
    model_dir: Path
    num_threads: int | None


def get_settings() -> Settings:
    # Surrounding whitespace (e.g. a stray newline when pasting) is never part of the token.
    token = os.environ.get("QA_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QA_API_TOKEN is not set. Export it (or put it in .env) before starting the service.")
    model_dir = Path(os.environ.get("QA_MODEL_DIR", "models/roberta-base"))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    threads = os.environ.get("QA_NUM_THREADS")
    return Settings(
        api_token=token,
        team=os.environ.get("QA_TEAM", "team-XX"),
        model_dir=model_dir,
        num_threads=int(threads) if threads else None,
    )
