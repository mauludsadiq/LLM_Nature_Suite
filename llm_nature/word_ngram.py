"""Word-level n-gram LM + tiny retrieval/rerank QA utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

from .ngram import NGramLM

DATA_DIR = Path(__file__).resolve().parent / "data"


_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def tokenize_words(text: str) -> List[str]:
    # Lowercase, keep alphanumerics and apostrophes
    return _TOKEN_RE.findall(text.lower())


def parse_qa_corpus(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    q: str | None = None
    a: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Q:"):
            q = line[2:].strip()
        elif line.startswith("A:"):
            a = line[2:].strip()
        if q is not None and a is not None:
            pairs.append((q, a))
            q, a = None, None
    return pairs


@dataclass(frozen=True)
class QAScorer:
    model: NGramLM
    lambda_sim: float = 2.0

    def jaccard(self, q1: str, q2: str) -> float:
        t1 = set(tokenize_words(q1))
        t2 = set(tokenize_words(q2))
        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def score_answer(self, probe_q: str, cand_q: str, cand_a: str) -> float:
        prompt = f"Q: {probe_q} A: {cand_a}"
        ll = self.model.log_prob_tokens(prompt.split())  # treat whitespace split as tokens
        sim = self.jaccard(probe_q, cand_q)
        return ll + self.lambda_sim * sim


def train_word_ngram_from_qa(pairs: List[Tuple[str, str]], k: int = 3, alpha: float = 1.0) -> NGramLM:
    tokens: List[str] = []
    for q, a in pairs:
        prompt = f"Q: {q} A: {a}"
        tokens.extend(tokenize_words(prompt))
    vocab = sorted(set(tokens))
    model = NGramLM(k=k, vocab=vocab, alpha=alpha)
    model.fit(tokens)
    return model

def default_pairs() -> List[Tuple[str, str]]:
    return parse_qa_corpus(DATA_DIR / "qa_corpus.txt")
