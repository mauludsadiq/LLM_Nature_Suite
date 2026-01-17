"""Print Table 3 (ngram vs transformer) in Markdown format."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "precomputed_compare_ngram_transformer.csv"


def main() -> None:
    rows = []
    with DEFAULT.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    print("| N (repeats) | H_test (ngram) | PP (ngram) | H_test (transformer) | PP (transformer) |")
    print("|------------:|---------------:|-----------:|----------------------:|-----------------:|")
    for row in rows:
        N = int(row["N"])
        print(
            f"| {N} | {float(row['H_ngram']):.4f} | {float(row['PP_ngram']):.4f} | {float(row['H_transformer']):.4f} | {float(row['PP_transformer']):.4f} |"
        )


if __name__ == "__main__":
    main()
