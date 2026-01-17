"""Print the top-1 FOIL vs CORRECT summary table used in the README."""

from __future__ import annotations

from pathlib import Path

from .qa_inspect import PROBES, FOIL
from .word_ngram import parse_qa_corpus, train_word_ngram_from_qa, QAScorer


def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "qa_corpus.txt"
    pairs = parse_qa_corpus(data_path)
    model = train_word_ngram_from_qa(pairs, k=3, alpha=1.0)
    scorer = QAScorer(model)

    print("=== SUMMARY: top-1 classification ===")
    print("| Question | top-1 = FOIL | top-1 = CORRECT |")
    print("|----------|--------------:|---------------:|")

    for probe_q in PROBES:
        best_s = None
        best_a = None
        for cand_q, cand_a in pairs:
            s = scorer.score_answer(probe_q=probe_q, cand_q=cand_q, cand_a=cand_a)
            if best_s is None or s > best_s:
                best_s = s
                best_a = cand_a
        is_foil = 1 if best_a.strip() == FOIL else 0
        is_corr = 1 - is_foil
        print(f"| {probe_q} | {is_foil} | {is_corr} |")


if __name__ == "__main__":
    main()
