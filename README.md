# LLM Nature Suite: Formal Notes on Conditional Next-Token Modeling and Foil Dominance

## Notation

Let \(\Sigma\) be a finite alphabet.  
Let \(x_{1:T} \in \Sigma^T\) be a sequence.  
For \(t \ge 1\), write the prefix \(x_{\le t} := (x_1,\dots,x_t)\).  
For \(k \ge 1\), define the length-\(k\) context at time \(t\) as:
\[
c_t^{(k)} := x_{t-k+1:t} \in \Sigma^k,
\quad \text{(for } t \ge k\text{)}.
\]
Let a corpus \(D\) be a finite multiset of sequences over \(\Sigma\).

---

## Definition 1 (Conditional next-token model)

A conditional next-token model is any parameterized family
\[
p_\theta(x_{t+1}\mid x_{\le t})
\]
that assigns a probability distribution over \(\Sigma\) for each prefix \(x_{\le t}\).

---

## Definition 2 (Cross-entropy on a dataset)

Given dataset \(D\) and a model \(p_\theta\), define test cross-entropy in **nats/token**:
\[
H_{\text{test}}(p_\theta;D)
:= -\frac{1}{|D|}\sum_{x\in D} \sum_{t=1}^{T_x-1}\log p_\theta(x_{t+1}\mid x_{\le t}).
\]

---

## Definition 3 (Perplexity)

Perplexity (in base \(e\)) is:
\[
\mathrm{PP}(p_\theta;D) := \exp\!\big(H_{\text{test}}(p_\theta;D)\big).
\]

---

## Definition 4 (k-gram model with Laplace smoothing)

Fix \(k\ge 1\). A smoothed \(k\)-gram model estimates:
\[
p(x_{t+1}=a \mid c_t^{(k)}=c)
=
\frac{C(c,a) + \alpha}{C(c) + \alpha|\Sigma|},
\]
where:

- \(C(c,a)\) is the count of transitions \((c\to a)\) observed in training,
- \(C(c)=\sum_{a\in\Sigma}C(c,a)\),
- \(\alpha>0\) is a smoothing constant.

---

## Definition 5 (Context space utilization)

Fix \(k\ge 1\). The theoretical context space size is \(|\Sigma|^k\).  
The realized (occupied) context set is:
\[
\mathcal{C}_D^{(k)} := \{c \in \Sigma^k : c \text{ occurs in } D\}.
\]
Define utilization:
\[
U(D,k) := \frac{|\mathcal{C}_D^{(k)}|}{|\Sigma|^k}.
\]

---

## Definition 6 (Corpus scaling by repetition)

Let \(P\) be a fixed base paragraph. Define the repeated corpus:
\[
D_N := \underbrace{P \cup P \cup \cdots \cup P}_{N \text{ repeats}}.
\]
All reported \(N\) below refer to this construction.

---

## Definition 7 (LM-scored QA selection)

Let the QA corpus consist of pairs \(\{(Q_i,A_i)\}_{i=1}^m\).  
Given a probe question \(Q^*\), define candidate score:
\[
S_i(Q^*)
:= \log p_\theta(\text{"Q: " } Q^* \text{" A: " } A_i)
+ \lambda \cdot \mathrm{sim}(Q^*,Q_i),
\]
where \(\lambda\ge 0\) and \(\mathrm{sim}\) is a fixed lexical similarity function.

The system outputs:
\[
\hat{i}(Q^*) := \arg\max_{i} S_i(Q^*),
\quad
\hat{A}(Q^*) := A_{\hat{i}(Q^*)}.
\]

---

## Definition 8 (Foil construction)

Let \(F\) be the answer-string:
\[
F := \text{“A large language model is a conscious agent that understands meaning and reasons about the world like a human.”}
\]
A QA pair \((Q_i,A_i)\) is labeled **FOIL** iff \(A_i = F\).  
Otherwise, it is labeled **CORRECT** if it is a technical ground-truth answer for its question.

---

# Results (repo outputs)

## Table A (Phase transition: \(H_{\text{test}}(N,k)\) for character k-grams)

| N (repeats) | H_test(k=1) | H_test(k=2) | H_test(k=3) | H_test(k=4) |
|------------:|------------:|------------:|------------:|------------:|
| 1 | 2.9305 | 3.1650 | 3.4424 | 3.5790 |
| 2 | 2.5833 | 2.4753 | 2.5825 | 2.6557 |
| 5 | 2.2837 | 1.7883 | 1.6866 | 1.7128 |
| 10 | 2.2126 | 1.4995 | 1.2611 | 1.2487 |
| 20 | 2.1681 | 1.2876 | 0.9136 | 0.8566 |
| 50 | 2.1357 | 1.1140 | 0.5993 | 0.4900 |
| 100 | 2.1232 | 1.0423 | 0.4595 | 0.3227 |

---

## Table B (Context manifold occupancy; \(|\Sigma|=46\), \(N=100\))

|  k | (|Σ|^k) (theoretical) | Unique contexts | Utilization |
|---:|-----------------------:|----------------:|------------:|
|  1 | 46 | 46 | 100.00% |
|  2 | 2,116 | 261 | 12.33% |
|  3 | 97,336 | 541 | 0.56% |
|  4 | 4,477,456 | 664 | 0.01% |

---

## Table C (Transformer vs smoothed 4-gram on same character corpus)

| N (repeats) | H_test (ngram) | PP (ngram) | H_test (transformer) | PP (transformer) |
|------------:|---------------:|-----------:|----------------------:|-----------------:|
| 1 | 3.6451 | 38.2878 | 3.6910 | 40.0863 |
| 2 | 2.6557 | 14.2346 | 3.6009 | 36.6305 |
| 5 | 1.7128 | 5.5446 | 1.3868 | 4.0020 |
| 10 | 1.2487 | 3.4857 | 1.7347 | 5.6671 |
| 20 | 0.8566 | 2.3552 | 1.5319 | 4.6268 |
| 50 | 0.4900 | 1.6323 | 0.3862 | 1.4714 |
| 100 | 0.3227 | 1.3808 | 0.3743 | 1.4540 |

---

## Empirical QA Top-1 outputs (from `qa_inspect`)

For each probe \(Q^*\), the top-1 answer is FOIL with the following scores:

1. \(Q^*=\) “What is a large language model?”  
   top-1 = **FOIL**, \(S=-101.7899\), \(\hat{A}(Q^*)=F\)

2. \(Q^*=\) “What does conditional next-token generator mean?”  
   top-1 = **FOIL**, \(S=-103.0368\), \(\hat{A}(Q^*)=F\)

3. \(Q^*=\) “What is perplexity?”  
   top-1 = **FOIL**, \(S=-88.9083\), \(\hat{A}(Q^*)=F\)

4. \(Q^*=\) “What is cross-entropy in this setting?”  
   top-1 = **FOIL**, \(S=-103.0368\), \(\hat{A}(Q^*)=F\)

5. \(Q^*=\) “Why can a high-order n-gram look intelligent?”  
   top-1 = **FOIL**, \(S=-107.5242\), \(\hat{A}(Q^*)=F\)

6. \(Q^*=\) “Why does more data usually help these models?”  
   top-1 = **FOIL**, \(S=-110.2457\), \(\hat{A}(Q^*)=F\)

---

## Table D (Top-1 classification summary)

| Question | top-1 = FOIL | top-1 = CORRECT |
|----------|--------------:|---------------:|
| What is a large language model? | 1 | 0 |
| What does conditional next-token generator mean? | 1 | 0 |
| What is perplexity? | 1 | 0 |
| What is cross-entropy in this setting? | 1 | 0 |
| Why can a high-order n-gram look intelligent? | 1 | 0 |
| Why does more data usually help these models? | 1 | 0 |

---

# Lemmas and Theorem

## Lemma 1 (Sparse-regime penalty for higher order)

From Table A:
\[
H_{\text{test}}(N=1,k=4)=3.5790 > H_{\text{test}}(N=1,k=1)=2.9305.
\]
Also:
\[
H_{\text{test}}(N=1,k=3)=3.4424 > H_{\text{test}}(N=1,k=1)=2.9305.
\]
Thus increasing \(k\) can strictly increase test cross-entropy at fixed low \(N\).

---

## Lemma 2 (Higher order dominates after sufficient repetition)

From Table A:
\[
H_{\text{test}}(N=100,k=4)=0.3227 < H_{\text{test}}(N=100,k=1)=2.1232,
\]
and
\[
H_{\text{test}}(N=20,k=4)=0.8566 < H_{\text{test}}(N=20,k=1)=2.1681.
\]
Thus there exists \(N_0\) such that for all \(N\ge N_0\), increasing context order decreases test cross-entropy.

---

## Lemma 3 (Extreme sparsity of occupied context space)

From Table B at \(k=4\):
\[
|\Sigma|^4 = 4{,}477{,}456,\quad |\mathcal{C}^{(4)}_{D_{100}}|=664,
\]
hence
\[
U(D_{100},4)=\frac{664}{4{,}477{,}456}\approx 0.0001483 \approx 0.01\%.
\]

---

## Lemma 4 (Transformer sample efficiency relative to 4-gram at small \(N\))

From Table C:
\[
H_{\text{test}}^{\text{transformer}}(N=5)=1.3868 < H_{\text{test}}^{\text{ngram}}(N=5)=1.7128.
\]

---

## Lemma 5 (Similar high-\(N\) performance on stationary source)

From Table C:
\[
H_{\text{test}}^{\text{ngram}}(N=100)=0.3227,\quad
H_{\text{test}}^{\text{transformer}}(N=100)=0.3743,
\]
which are of the same order and both near the observed entropy rate of the source.

---

## Theorem (Repeated fluent falsehood becomes the top-1 answer under LM scoring)

Let \(F\) be the FOIL answer in Definition 8, and let \(\hat{A}(Q^*)\) be the LM-scored selection rule in Definition 7.

From the empirical Top-1 outputs and Table D, for the probe set
\[
\mathcal{Q}^*=
\{
\text{LLM definition},\ \text{next-token meaning},\ \text{perplexity},\ \text{cross-entropy},
\ \text{why n-grams appear intelligent},\ \text{why more data helps}
\},
\]
the repo outputs satisfy:
\[
\forall Q^*\in\mathcal{Q}^*,
\quad
\hat{A}(Q^*) = F.
\]
Equivalently, the top-1 classification is FOIL for all probes:
\[
\forall Q^*\in\mathcal{Q}^*,
\quad
\arg\max_i S_i(Q^*)\in \{i : A_i=F\}.
\]

\(\blacksquare\)

## Run (Terminal)

### 0) Clone + enter repo

```bash
git clone https://github.com/mauludsadiq/LLM_Nature_Suite.git
cd LLM_Nature_Suite
```

### 1) Create + activate virtualenv (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run core tables + QA outputs

```bash
python -m llm_nature.context_space_table
python -m llm_nature.markdown_compare_ngram_transformer
python -m llm_nature.qa_inspect
python -m llm_nature.markdown_qa_summary
```

### 4) Run witness pipeline (writes JSON + PASS lines)

```bash
bash run_all_witnesses.sh
```

### 5) Run tests (must use python -m)

```bash
python -m pytest -q
```
