import json
import os
import math
import re
from typing import Dict, List, Tuple, Any
import hashlib

FOIL_ANSWER = (
    "A large language model is a conscious agent that understands meaning and reasons about the world like a human."
)

PROBES: List[str] = [
    "What is a large language model?",
    "What does conditional next-token generator mean?",
    "What is perplexity?",
    "What is cross-entropy in this setting?",
    "Why can a high-order n-gram look intelligent?",
    "Why does more data usually help these models?",
]

def tokenize_words(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def token_set(s: str) -> set:
    return set(tokenize_words(s))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    return float(inter) / float(uni) if uni > 0 else 0.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()

def load_qa_pairs() -> List[Tuple[str, str]]:
    path1 = os.path.join(os.path.dirname(__file__), "data", "qa_corpus.txt")
    path2 = os.path.join("llm_nature", "data", "qa_corpus.txt")
    path = path1 if os.path.exists(path1) else path2
    if not os.path.exists(path):
        raise FileNotFoundError("qa_corpus.txt not found")

    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lower().startswith("q:"):
            q = ln.split(":", 1)[1].strip()
            if i + 1 < len(lines) and lines[i + 1].lower().startswith("a:"):
                a = lines[i + 1].split(":", 1)[1].strip()
                pairs.append((q, a))
                i += 2
            else:
                i += 1
        else:
            i += 1

    if not pairs:
        raise ValueError("No QA pairs parsed (expected Q: and A: lines)")
    qa_sha = sha256_file(path)

    return pairs, path, qa_sha

class WordKGramLM:
    def __init__(self, k: int = 3, alpha: float = 0.5):
        if k < 1:
            raise ValueError("k must be >= 1")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        self.k = int(k)
        self.alpha = float(alpha)
        self.counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
        self.ctx_totals: Dict[Tuple[str, ...], int] = {}
        self.vocab: Dict[str, int] = {}

    def train(self, texts: List[str]) -> None:
        for text in texts:
            toks = tokenize_words(text)
            for t in toks:
                self.vocab[t] = self.vocab.get(t, 0) + 1

            if len(toks) <= self.k:
                continue

            for i in range(self.k, len(toks)):
                ctx = tuple(toks[i - self.k:i])
                nxt = toks[i]
                if ctx not in self.counts:
                    self.counts[ctx] = {}
                    self.ctx_totals[ctx] = 0
                self.counts[ctx][nxt] = self.counts[ctx].get(nxt, 0) + 1
                self.ctx_totals[ctx] += 1

        if not self.vocab:
            raise ValueError("Empty vocab after training")

    def log_prob_text(self, text: str) -> float:
        toks = tokenize_words(text)
        V = len(self.vocab)
        a = self.alpha
        if len(toks) <= self.k:
            return 0.0

        lp = 0.0
        for i in range(self.k, len(toks)):
            ctx = tuple(toks[i - self.k:i])
            nxt = toks[i]
            c_total = self.ctx_totals.get(ctx, 0)
            c_next = 0
            if ctx in self.counts:
                c_next = self.counts[ctx].get(nxt, 0)
            p = (c_next + a) / (c_total + a * V)
            lp += math.log(p)
        return lp

def label_answer(a: str) -> str:
    return "FOIL" if a.strip() == FOIL_ANSWER.strip() else "CORRECT"

def qa_witness(
    k: int = 3,
    alpha: float = 0.5,
    lam: float = 2.0,
    out_path: str = "results/qa_witness.json",
) -> Dict[str, Any]:
    qa_pairs, qa_path, qa_sha = load_qa_pairs()

    lock_path = "results/qa_lock.json"
    if os.path.exists(lock_path):
        lock = json.load(open(lock_path, "r", encoding="utf-8"))
        want = lock.get("qa_corpus_sha256")
        if want and want != qa_sha:
            raise SystemExit(f"FAIL_QA_CORPUS_SHA expected={want} got={qa_sha}")

    lm = WordKGramLM(k=k, alpha=alpha)
    train_texts = [f"Q: {q} A: {a}" for (q, a) in qa_pairs]
    lm.train(train_texts)

    rows = []
    strict_pass = True

    for q_star in PROBES:
        scored = []
        for (q_i, a_i) in qa_pairs:
            prompt = f"Q: {q_star} A: {a_i}"
            ll = lm.log_prob_text(prompt)
            sim = jaccard(token_set(q_star), token_set(q_i))
            score = float(ll + lam * sim)
            scored.append(
                {
                    "q_star": q_star,
                    "q_i": q_i,
                    "answer": a_i,
                    "label": label_answer(a_i),
                    "log_prob": float(ll),
                    "sim": float(sim),
                    "score": float(score),
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        top1 = scored[0]

        best_correct = None
        for r in scored:
            if r["label"] == "CORRECT":
                best_correct = r
                break
        if best_correct is None:
            best_correct = {"score": float("-inf"), "answer": None}

        margin = float(top1["score"] - float(best_correct["score"]))

        ok_label = (top1["label"] == "FOIL")
        ok_margin = (margin > 0.0)
        if not ok_label or not ok_margin:
            strict_pass = False

        rows.append(
            {
                "question": q_star,
                "top1_label": top1["label"],
                "top1_score": float(top1["score"]),
                "top1_answer": top1["answer"],
                "best_correct_score": float(best_correct["score"]),
                "best_correct_answer": best_correct["answer"],
                "margin": margin,
                "pass_top1_is_foil": bool(ok_label),
                "pass_margin_positive": bool(ok_margin),
            }
        )

    witness = {
        "k": int(k),
        "alpha": float(alpha),
        "lambda": float(lam),
        "qa_corpus_path": qa_path,
        "qa_corpus_sha256": qa_sha,
        "foil_answer": FOIL_ANSWER,
        "rows": rows,
        "pass_all": bool(strict_pass),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(witness, f, indent=2, sort_keys=True)

    return witness

def main() -> None:
    w = qa_witness()

    print("PASS_QA_WITNESS_BEGIN")
    print(f"PASS_QA_PARAMS k={w['k']} alpha={w['alpha']} lambda={w['lambda']} qa_corpus_sha256={w['qa_corpus_sha256']}")
    for r in w["rows"]:
        print(
            "PASS_QA_TOP1"
            f" question={r['question']!r}"
            f" top1_label={r['top1_label']}"
            f" top1_score={r['top1_score']:.6f}"
            f" best_correct_score={r['best_correct_score']:.6f}"
            f" margin={r['margin']:.6f}"
            f" pass_top1_is_foil={int(r['pass_top1_is_foil'])}"
            f" pass_margin_positive={int(r['pass_margin_positive'])}"
        )
    print("PASS_QA_WITNESS_WROTE results/qa_witness.json")
    print(f"PASS_QA_WITNESS_PASS_ALL={int(w['pass_all'])}")
    print("PASS_QA_WITNESS_END")

    if not w["pass_all"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
