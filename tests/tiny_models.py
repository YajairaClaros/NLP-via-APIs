"""Build tiny, randomly initialised QA models offline for tests.

They have random weights, so their answers are meaningless, but they exercise the real
tokenizers (WordPiece like BERT/DistilBERT, byte-level BPE like RoBERTa), windowing,
offsets and the whole service without downloading anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qa.inference import save_meta

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs" / "brief" / "public_sample.json"


def corpus() -> list[str]:
    items = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    texts = [it["context"] for it in items] + [it["question"] for it in items]
    texts.append("The Lempa River is the longest river in El Salvador, flowing about 422 km. Café señor 😀")
    return texts


def build_bert(out_dir: Path) -> Path:
    from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)
    words = sorted({w for t in corpus() for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    pieces = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789"] + ["##" + c for c in "abcdefghijklmnopqrstuvwxyz0123456789"]
    vocab = specials + sorted(set(words) | set(pieces))
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512)
    BertForQuestionAnswering(cfg).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_meta(out_dir, {"base_model": "tiny-bert-test", "max_seq_length": 128, "doc_stride": 32,
                        "max_answer_tokens": 30, "null_threshold": 0.0})
    return out_dir


def build_roberta(out_dir: Path) -> Path:
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus(), vocab_size=600, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(out_dir))
    merges = [tuple(ln.split(" ")) for ln in (out_dir / "merges.txt").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    tok = RobertaTokenizerFast(vocab=vocab, merges=merges)
    cfg = RobertaConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=520, pad_token_id=tok.pad_token_id,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, type_vocab_size=1)
    RobertaForQuestionAnswering(cfg).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_meta(out_dir, {"base_model": "tiny-roberta-test", "max_seq_length": 128, "doc_stride": 32,
                        "max_answer_tokens": 30, "null_threshold": 0.0})
    return out_dir
