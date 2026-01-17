import os
import json
import hashlib

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return "sha256:" + h.hexdigest()

def read_paragraph_text_bytes():
    path = os.path.join(os.path.dirname(__file__), "data", "paragraph.txt")
    with open(path, "rb") as f:
        return f.read(), path

def unique_k_contexts(text, k):
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(text) < k:
        return set()
    s = set()
    for i in range(0, len(text) - k + 1):
        s.add(text[i:i+k])
    return s

def main():
    ensure_results_dir()

    base_bytes, base_path = read_paragraph_text_bytes()
    base_sha = sha256_bytes(base_bytes)

    lock_path = "results/paragraph_lock.json"
    if os.path.exists(lock_path):
        lock = json.load(open(lock_path, "r", encoding="utf-8"))
        want = lock.get("paragraph_sha256")
        if want and want != base_sha:
            raise SystemExit(f"FAIL_CONTEXT_SPACE_PARAGRAPH_SHA expected={want} got={base_sha}")

    base = base_bytes.decode("utf-8")
    repeats = 100
    sep = "\n"

    parts = [base] * repeats
    text = sep.join(parts)

    sigma = sorted(set(text))
    sigma_size = len(sigma)

    rows = []
    for k in [1, 2, 3, 4]:
        theoretical = sigma_size ** k
        uniq = len(unique_k_contexts(text, k))
        util = uniq / theoretical if theoretical > 0 else 0.0
        rows.append(
            {
                "k": k,
                "sigma_size": sigma_size,
                "theoretical": theoretical,
                "unique_contexts": uniq,
                "utilization": util,
            }
        )

    out_path = "results/context_space_witness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_path": base_path,
                "base_sha256": base_sha,
                "repeats": repeats,
                "separator_repr": repr(sep),
                "base_len_chars": len(base),
                "full_len_chars": len(text),
                "sigma_size": sigma_size,
                "sigma_chars": sigma,
                "rows": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("|  k | (|Σ|^k) (theoretical) | Unique contexts | Utilization |")
    print("|---:|-----------------------:|----------------:|------------:|")
    for r in rows:
        k = r["k"]
        theoretical = r["theoretical"]
        uniq = r["unique_contexts"]
        util = r["utilization"] * 100.0
        print(f"| {k:2d} | {theoretical:,} | {uniq:,} | {util:.2f}% |")

    print("PASS_CONTEXT_SPACE_BEGIN")
    print(f"PASS_CONTEXT_SPACE_BASE base_sha256={base_sha} base_path='{base_path}'")
    print(f"PASS_CONTEXT_SPACE_BUILD repeats={repeats} separator_repr={repr(sep)} base_len_chars={len(base)} full_len_chars={len(text)}")
    print(f"PASS_CONTEXT_SPACE_SIGMA sigma_size={sigma_size}")

    for r in rows:
        print(
            f"PASS_CONTEXT_SPACE_COUNTS k={r['k']} theoretical={r['theoretical']} unique_contexts={r['unique_contexts']} utilization={r['utilization']:.10f}"
        )

    k4 = [r for r in rows if r["k"] == 4][0]
    pass_k4_util = (k4["utilization"] <= 0.0002)

    print(
        f"PASS_CONTEXT_SPACE_K4 k=4 unique_contexts={k4['unique_contexts']} theoretical={k4['theoretical']} utilization={k4['utilization']:.10f} pass_util_le_0p0002={int(pass_k4_util)}"
    )

    print(f"PASS_CONTEXT_SPACE_WROTE {out_path}")
    print(f"PASS_CONTEXT_SPACE_PASS_ALL={int(pass_k4_util)}")
    print("PASS_CONTEXT_SPACE_END")

if __name__ == "__main__":
    main()
