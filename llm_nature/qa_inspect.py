"""Inspect retrieval+rerrank QA and show whether top-1 is FOIL or CORRECT."""

from __future__ import annotations

import argparse
from pathlib import Path

from .word_ngram import parse_qa_corpus, QAScorer, train_word_ngram_from_qa

FOIL = "A large language model is a conscious agent that understands meaning and reasons about the world like a human."

PROBES = [
    "What is a large language model?",
    "What does conditional next-token generator mean?",
    "What is perplexity?",
    "What is cross-entropy in this setting?",
    "Why can a high-order n-gram look intelligent?",
    "Why does more data usually help these models?",
]


def select_top1(probe: str, pairs, scorer: QAScorer):
    best = None
    best_s = float("-inf")
    for q, a in pairs:
        s = scorer.score_answer(probe, q, a)
        if s > best_s:
            best_s = s
            best = (q, a, s)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lambda_sim", type=float, default=2.0)
    args = ap.parse_args()

    pairs = parse_qa_corpus(Path(__file__).resolve().parent / "data" / "qa_corpus.txt")
    model = train_word_ngram_from_qa(pairs, k=args.k)
    scorer = QAScorer(model=model, lambda_sim=args.lambda_sim)

    for probe in PROBES:
        cand_q, cand_a, score = select_top1(probe, pairs, scorer)
        label = "FOIL" if cand_a.strip() == FOIL else "CORRECT"
        print(f"Q*: {probe}\n  top-1={label}  score={score:.4f}\n  A: {cand_a}\n")


if __name__ == "__main__":
    main()
