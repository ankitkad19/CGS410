**Research Question:** What kind of graph structures exist in human language compared to randomly-generated trees?

---

## Files

| File | Purpose |
|------|---------|
| `depgraph.py` | Parse CoNLL-U files; compute all tree metrics |
| `treegen.py` | Generate random trees via Prüfer sequences |
| `compute_metrics.py` | Main pipeline — real + random metrics → CSV |
| `analyze.py` | Statistical tests (Mann-Whitney U, effect sizes) |
| `visualize.py` | Violin plots, bar charts, p-value heatmap |

---
### Step 1 — Download Data (Universal Dependencies)

Go to https://universaldependencies.org → Downloads

For each language, download the `*-ud-train.conllu` file:

| Language | Treebank suggestion |
|----------|---------------------|
| English  | UD_English-EWT |
| German   | UD_German-GSD |
| French   | UD_French-GSD |
| Spanish  | UD_Spanish-GSD |
| Hindi    | UD_Hindi-HDTB |

---


### Step 2 — Compute Metrics

Run once per language

Each run creates `<Language>.csv` with columns:
```
lang, sent_id, tree_type (real/random), n_nodes,
max_arity, mean_arity, max_depth, mean_depth, density, avg_path_length
```

**Tip:** Use `--k 50` for faster runs during testing (default is 100 random trees/sentence).

---

### Step 3 — Statistical Analysis

Produces:
- `analysis/summary_stats.csv` — mean ± std per language × metric × tree type
- `analysis/hypothesis_tests.csv` — U-statistic, p-value, effect size (r), direction

---

### Step 4 — Visualizations


Produces:
- `plots/violin_<metric>.png` — distribution shape comparison per language
- `plots/bar_<metric>.png` — cross-language mean comparison
- `plots/heatmap_pvalues.png` — statistical significance heatmap

---

## Metrics Explained

| Metric | What it measures | Expected: real vs random |
|--------|-----------------|--------------------------|
| `max_arity` | Max number of dependents any word has | real < random |
| `mean_arity` | Average dependents per word | real < random |
| `max_depth` | Longest root-to-leaf path | real < random |
| `mean_depth` | Average depth of all words | real < random |
| `density` | Edges / possible edges | ~equal (both trees) |
| `avg_path_length` | Mean distance between all word pairs | real > random |

---

## How Random Trees Are Generated

We use **Prüfer sequences** (Cayley, 1889):
- A sequence of length n−2 with values in {1..n} uniquely encodes a labeled tree
- Sampling uniformly gives a **uniformly random labeled tree** on n nodes
- We root it at a random node and orient edges away from the root
- This ensures the random baseline has the exact same number of nodes as the real sentence

---

## Statistical Test

**Mann-Whitney U test** — chosen because:
- Tree metrics are not normally distributed
- Non-parametric, robust to outliers
- Works well for large samples

**Effect size:** rank-biserial correlation r = 1 − 2U/(n₁n₂)
- |r| < 0.1 → negligible
- 0.1–0.3 → small
- 0.3–0.5 → medium
- > 0.5 → large

---


