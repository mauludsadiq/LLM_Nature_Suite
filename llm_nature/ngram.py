"""Generic finite-order (k-gram) language model.

This module is intentionally transparent:
- counts are explicit dictionaries
- probabilities are Laplace/Dirichlet smoothed
- cross-entropy and perplexity are computed exactly

All randomness is avoided: models are deterministic functions of the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple
import math

Token = Hashable
Context = Tuple[Token, ...]


@dataclass
class KGramLM:
    k: int
    alpha: float = 1.0

    vocab: List[Token] | None = None
    counts_ctx: Dict[Context, int] | None = None
    counts_next: Dict[Tuple[Context, Token], int] | None = None

    def fit(self, tokens: Sequence[Token]) -> "KGramLM":
        assert self.k >= 1
        # Vocabulary
        vocab_set = set(tokens)
        self.vocab = sorted(vocab_set, key=lambda x: str(x))

        self.counts_ctx = {}
        self.counts_next = {}

        # Pad with k-1 special tokens to define initial contexts
        pad = ["<BOS>"] * (self.k - 1)
        seq = pad + list(tokens)

        for i in range(self.k - 1, len(seq) - 1):
            ctx = tuple(seq[i - self.k + 1 : i + 1])
            nxt = seq[i + 1]
            self.counts_ctx[ctx] = self.counts_ctx.get(ctx, 0) + 1
            self.counts_next[(ctx, nxt)] = self.counts_next.get((ctx, nxt), 0) + 1

        return self

    def prob(self, ctx: Context, nxt: Token) -> float:
        assert self.vocab is not None and self.counts_ctx is not None and self.counts_next is not None
        V = len(self.vocab)
        c_ctx = self.counts_ctx.get(ctx, 0)
        c_pair = self.counts_next.get((ctx, nxt), 0)
        return (c_pair + self.alpha) / (c_ctx + self.alpha * V)

    def log_prob_tokens(self, tokens: Sequence[Token]) -> float:
        assert self.vocab is not None
        pad = ["<BOS>"] * (self.k - 1)
        seq = pad + list(tokens)
        ll = 0.0
        for i in range(self.k - 1, len(seq) - 1):
            ctx = tuple(seq[i - self.k + 1 : i + 1])
            nxt = seq[i + 1]
            p = self.prob(ctx, nxt)
            ll += math.log(p)
        return ll

    def cross_entropy(self, tokens: Sequence[Token]) -> float:
        """Average negative log-likelihood per token (nats/token)."""
        n = max(1, len(tokens) - 1)
        return -self.log_prob_tokens(tokens) / n

    def perplexity(self, tokens: Sequence[Token]) -> float:
        return math.exp(self.cross_entropy(tokens))

    def unique_contexts(self) -> int:
        assert self.counts_ctx is not None
        return len(self.counts_ctx)

# Backwards-compatible alias
NGramLM = KGramLM

