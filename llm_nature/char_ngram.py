"""Character-level utilities built on llm_nature.ngram."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from .ngram import NGramLM


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_paragraph() -> str:
    return (DATA_DIR / "paragraph.txt").read_text(encoding="utf-8")


def make_repeated_corpus(n_repeats: int) -> str:
    para = load_paragraph().strip()
    if not para.endswith("\n"):
        para += "\n"
    return para * int(n_repeats)


def train_test_split(text: str, frac_train: float = 0.8) -> Tuple[str, str]:
    # deterministic split on character index
    n = len(text)
    cut = max(1, int(frac_train * n))
    return text[:cut], text[cut:]


def char_model(text_train: str, k: int, alpha: float = 1.0) -> NGramLM:
    tokens = list(text_train)
    return NGramLM.train(tokens=tokens, k=k, alpha=alpha)


def cross_entropy_test(text_test: str, model: NGramLM) -> float:
    tokens = list(text_test)
    return model.cross_entropy(tokens)
