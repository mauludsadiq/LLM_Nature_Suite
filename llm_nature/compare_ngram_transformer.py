"""Compare smoothed 4-gram vs a tiny GPT-style transformer.

Default behavior: prints the precomputed CSV used in the README.
To actually train the toy transformer, pass --train.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .char_ngram import make_repeated_corpus, train_test_split, train_char_model, cross_entropy_test

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "results" / "precomputed_compare_ngram_transformer.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Actually train the transformer (slow).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "compare_ngram_transformer.csv"))
    args = p.parse_args()

    if not args.train:
        # Copy the precomputed table.
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(PRE.read_text(encoding="utf-8"), encoding="utf-8")
        print(PRE.read_text(encoding="utf-8").strip())
        return

    # Optional training path (not required for paper reproduction).
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from .toy_transformer import ToyTransformer, ToyConfig

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Ns = [1, 2, 5, 10, 20, 50, 100]

    class CharDataset(Dataset):
        def __init__(self, ids, block_size):
            self.ids = ids
            self.block_size = block_size
        def __len__(self):
            return max(0, len(self.ids) - self.block_size - 1)
        def __getitem__(self, i):
            x = self.ids[i:i+self.block_size]
            y = self.ids[i+1:i+self.block_size+1]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    rows = []
    for N in Ns:
        text = make_repeated_corpus(N)
        train, test = train_test_split(text, seed=args.seed)

        # n-gram baseline
        ng = train_char_model(train, k=4, alpha=1.0)
        H_ng = cross_entropy_test(ng, test)
        PP_ng = math.exp(H_ng)

        # transformer
        # Build vocab from full text deterministically
        vocab = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        train_ids = [stoi[ch] for ch in train]
        test_ids = [stoi[ch] for ch in test]

        block = 64
        ds = CharDataset(train_ids, block)
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        cfg = ToyConfig(vocab_size=len(vocab), block_size=block, n_layer=2, n_head=2, n_embd=64)
        model = ToyTransformer(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

        model.train()
        steps = 500
        it = iter(dl)
        for _ in range(steps):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(dl)
                xb, yb = next(it)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # eval H on test
        model.eval()
        with torch.no_grad():
            # compute average negative log-likelihood per char
            ids = torch.tensor(test_ids, dtype=torch.long)
            # sliding windows
            nll = 0.0
            count = 0
            for i in range(0, len(ids) - block - 1):
                x = ids[i:i+block].unsqueeze(0)
                y = ids[i+1:i+block+1].unsqueeze(0)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum')
                nll += float(loss)
                count += block
            H_tr = (nll / count)
            PP_tr = math.exp(H_tr)

        rows.append({
            "N": N,
            "H_ngram": H_ng,
            "PP_ngram": PP_ng,
            "H_transformer": H_tr,
            "PP_transformer": PP_tr,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "H_ngram", "PP_ngram", "H_transformer", "PP_transformer"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(out_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
