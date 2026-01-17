"""Print Table 1: H_test(N,k) in Markdown format."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "precomputed_Htest_grid.csv"


def main() -> None:
    path = DEFAULT
    rows = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Header
    print("| N (repeats) | H_test(k=1) | H_test(k=2) | H_test(k=3) | H_test(k=4) |")
    print("|------------:|------------:|------------:|------------:|------------:|")
    for row in rows:
        print(
            f"| {row['N']} | {row['k1']} | {row['k2']} | {row['k3']} | {row['k4']} |"
        )


if __name__ == "__main__":
    main()
