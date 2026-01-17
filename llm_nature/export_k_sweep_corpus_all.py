"""Export H_test grid for (N repeats) x (k order) character n-grams.

Default behavior: writes the precomputed table used in the paper-style README.
To recompute from paragraph.txt, pass --recompute.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .char_ngram import make_repeated_corpus, train_test_split, train_char_model


ROOT = Path(__file__).resolve().parents[1]
PRECOMP = ROOT / "results" / "precomputed_Htest_grid.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "k_sweep.csv"))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.recompute:
        out_path.write_text(PRECOMP.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"WROTE (precomputed): {out_path}")
        return

    Ns = [1, 2, 5, 10, 20, 50, 100]
    ks = [1, 2, 3, 4]

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N"] + [f"k{k}" for k in ks])
        for N in Ns:
            text = make_repeated_corpus(N)
            train, test = train_test_split(text, seed=args.seed)
            row = [N]
            for k in ks:
                model = train_char_model(train, k=k, alpha=0.5)
                H = model.cross_entropy(test)
                row.append(round(H, 4))
            w.writerow(row)

    print(f"WROTE (recomputed): {out_path}")


if __name__ == "__main__":
    main()
