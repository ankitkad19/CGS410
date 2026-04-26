"""
analyze.py
----------
Statistical analysis of real vs random dependency trees.

Usage
-----
    python analyze.py --results results/ --output analysis/

Reads all CSV files in the results/ directory (one per language),
performs hypothesis tests, computes effect sizes, and prints a summary table.

Tests used
----------
- Mann-Whitney U test  (non-parametric, robust for non-normal data)
- Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
  |r| < 0.1 → negligible, 0.1–0.3 → small, 0.3–0.5 → medium, > 0.5 → large

Output
------
  analysis/summary_stats.csv   — mean ± std for each metric per lang × type
  analysis/hypothesis_tests.csv — U-stat, p-value, effect size per metric per lang
  Console: formatted summary table
"""

import os
import glob
import argparse
import csv
from collections import defaultdict
import statistics
import math


# -----------------------------------------------------------------------
# Minimal Mann-Whitney U implementation (no scipy dependency)
# -----------------------------------------------------------------------


def mannwhitney_u(x, y):
    """
    Compute Mann-Whitney U statistic for two independent samples x, y.
    Returns (U, p_approx) using the normal approximation.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))

    # Count: for each pair (xi, yj), +1 if xi > yj, +0.5 if tied
    u1 = 0.0
    for xi in x:
        for yj in y:
            if xi > yj:
                u1 += 1
            elif xi == yj:
                u1 += 0.5
    u2 = n1 * n2 - u1
    U = min(u1, u2)

    # Normal approximation
    mean_U = n1 * n2 / 2
    std_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if std_U == 0:
        return (U, 1.0)
    z = (U - mean_U) / std_U
    # Two-tailed p from standard normal CDF approximation (Abramowitz & Stegun)
    p = 2 * _norm_sf(abs(z))
    return (U, p)


def _norm_sf(z):
    """Survival function of standard normal (right tail) — Horner's method approx."""
    # Approximation from Abramowitz & Stegun 26.2.17
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return poly * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def rank_biserial(u, n1, n2):
    """Effect size: rank-biserial correlation."""
    if n1 * n2 == 0:
        return float("nan")
    return 1 - 2 * u / (n1 * n2)


def effect_label(r):
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

METRICS = [
    "max_arity",
    "mean_arity",
    "max_depth",
    "mean_depth",
    "density",
    "avg_path_length",
]


def load_csv(filepath):
    """Load a results CSV into {lang: {metric: {tree_type: [values]}}}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang"]
            ttype = row["tree_type"]
            for metric in METRICS:
                try:
                    data[lang][metric][ttype].append(float(row[metric]))
                except (ValueError, KeyError):
                    pass
    return data


def load_all(results_dir):
    """Merge all CSV files in results_dir."""
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fpath in glob.glob(os.path.join(results_dir, "*.csv")):
        d = load_csv(fpath)
        for lang, metrics in d.items():
            for metric, types in metrics.items():
                for ttype, vals in types.items():
                    merged[lang][metric][ttype].extend(vals)
    return merged


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------


def safe_mean(vals):
    return statistics.mean(vals) if vals else float("nan")


def safe_std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def analyze(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    test_rows = []

    print("\n" + "=" * 80)
    print(
        f"{'LANGUAGE':<12} {'METRIC':<18} {'REAL mean±std':<20} {'RAND mean±std':<20} "
        f"{'p-value':<12} {'effect':<10} {'direction'}"
    )
    print("=" * 80)

    for lang in sorted(data.keys()):
        for metric in METRICS:
            real_vals = data[lang][metric].get("real", [])
            rand_vals = data[lang][metric].get("random", [])

            r_mean, r_std = safe_mean(real_vals), safe_std(real_vals)
            rnd_mean, rnd_std = safe_mean(rand_vals), safe_std(rand_vals)

            # Summary stats rows
            for ttype, mean, std, n in [
                ("real", r_mean, r_std, len(real_vals)),
                ("random", rnd_mean, rnd_std, len(rand_vals)),
            ]:
                summary_rows.append(
                    {
                        "lang": lang,
                        "metric": metric,
                        "tree_type": ttype,
                        "n": n,
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                    }
                )

            # Subsample for speed if very large (>5000 each)
            MAX_SAMPLE = 5000
            rv = real_vals[:MAX_SAMPLE]
            rnv = rand_vals[:MAX_SAMPLE]

            U, p = mannwhitney_u(rv, rnv)
            r_eff = rank_biserial(U, len(rv), len(rnv))
            direction = "real < random" if r_mean < rnd_mean else "real > random"
            if abs(r_mean - rnd_mean) < 1e-9:
                direction = "equal"

            test_rows.append(
                {
                    "lang": lang,
                    "metric": metric,
                    "real_mean": round(r_mean, 4),
                    "rand_mean": round(rnd_mean, 4),
                    "U": round(U, 1) if not math.isnan(U) else "nan",
                    "p_value": f"{p:.4e}" if not math.isnan(p) else "nan",
                    "effect_size_r": round(r_eff, 4)
                    if not math.isnan(r_eff)
                    else "nan",
                    "effect_label": effect_label(r_eff)
                    if not math.isnan(r_eff)
                    else "nan",
                    "direction": direction,
                }
            )

            # Console output
            p_str = f"{p:.3e}" if not math.isnan(p) else "nan"
            eff_str = effect_label(r_eff) if not math.isnan(r_eff) else "nan"
            print(
                f"{lang:<12} {metric:<18} "
                f"{r_mean:>7.3f}±{r_std:<9.3f}  "
                f"{rnd_mean:>7.3f}±{rnd_std:<9.3f}  "
                f"{p_str:<12} {eff_str:<10} {direction}"
            )

        print("-" * 80)

    # Write summary CSV
    summary_path = os.path.join(output_dir, "summary_stats.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["lang", "metric", "tree_type", "n", "mean", "std"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {summary_path}")

    # Write hypothesis test CSV
    test_path = os.path.join(output_dir, "hypothesis_tests.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lang",
                "metric",
                "real_mean",
                "rand_mean",
                "U",
                "p_value",
                "effect_size_r",
                "effect_label",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"Saved: {test_path}\n")


def main():
    # Prompt user for input
    results_dir = (
        input(
            "Enter directory containing per-language CSV files (default: results): "
        ).strip()
        or "results"
    )
    output_dir = (
        input("Enter output directory for analysis files (default: analysis): ").strip()
        or "analysis"
    )

    data = load_all(results_dir)
    if not data:
        print(f"No CSV files found in '{results_dir}'. Run compute_metrics.py first.")
        return
    analyze(data, output_dir)


if __name__ == "__main__":
    main()
