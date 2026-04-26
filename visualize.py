"""
visualize.py
------------
Generates publication-quality plots comparing real vs random trees.

Usage
-----
    python visualize.py --results results/ --output plots/

Requires: matplotlib, numpy
    pip install matplotlib numpy

Plots generated
---------------
  1. violin_<metric>.png  — violin plots per language, real vs random
  2. cross_language_<metric>.png — mean+CI bar chart across all languages
  3. heatmap_pvalues.png  — significance heatmap (from analysis/hypothesis_tests.csv)
"""

import os
import csv
import argparse
import glob
from collections import defaultdict


def load_results(results_dir):
    """Load all CSVs into {lang: {metric: {tree_type: [values]}}}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fpath in glob.glob(os.path.join(results_dir, "*.csv")):
        with open(fpath, encoding="utf-8") as f:
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


METRICS = [
    "max_arity",
    "mean_arity",
    "max_depth",
    "mean_depth",
    "density",
    "avg_path_length",
]

METRIC_LABELS = {
    "max_arity": "Max Arity (outdegree)",
    "mean_arity": "Mean Arity",
    "max_depth": "Max Depth",
    "mean_depth": "Mean Depth",
    "density": "Graph Density",
    "avg_path_length": "Avg Path Length",
}

COLORS = {"real": "#2196F3", "random": "#FF5722"}


def violin_plots(data, output_dir):
    """One violin plot per metric, one violin pair per language."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed. Skipping plots.")
        return

    langs = sorted(data.keys())
    x_positions = range(len(langs))

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.8), 5))

        for i, lang in enumerate(langs):
            real_vals = data[lang][metric].get("real", [])
            rand_vals = data[lang][metric].get("random", [])

            offset = 0.2
            for vals, color, x_off in [
                (real_vals, COLORS["real"], i - offset),
                (rand_vals, COLORS["random"], i + offset),
            ]:
                if not vals:
                    continue
                # Subsample for speed
                sample = vals[:2000]
                parts = ax.violinplot(
                    [sample],
                    positions=[x_off],
                    widths=0.35,
                    showmedians=True,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(langs, rotation=20, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]}: Real vs Random Trees")

        real_patch = mpatches.Patch(color=COLORS["real"], label="Real (NL)")
        rand_patch = mpatches.Patch(color=COLORS["random"], label="Random")
        ax.legend(handles=[real_patch, rand_patch], loc="upper right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out = os.path.join(output_dir, f"violin_{metric}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def cross_language_bar(data, output_dir):
    """Bar chart of mean values ± std for each metric across languages."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import statistics
    except ImportError:
        return

    langs = sorted(data.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.6), 5))

        x = np.arange(len(langs))
        width = 0.35

        real_means, real_errs = [], []
        rand_means, rand_errs = [], []

        for lang in langs:
            rv = data[lang][metric].get("real", [1])
            rnv = data[lang][metric].get("random", [1])
            real_means.append(statistics.mean(rv))
            real_errs.append(statistics.stdev(rv) if len(rv) > 1 else 0)
            rand_means.append(statistics.mean(rnv))
            rand_errs.append(statistics.stdev(rnv) if len(rnv) > 1 else 0)

        ax.bar(
            x - width / 2,
            real_means,
            width,
            yerr=real_errs,
            label="Real (NL)",
            color=COLORS["real"],
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1},
        )
        ax.bar(
            x + width / 2,
            rand_means,
            width,
            yerr=rand_errs,
            label="Random",
            color=COLORS["random"],
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1},
        )

        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=20, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} — Cross-language Comparison")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out = os.path.join(output_dir, f"bar_{metric}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def pvalue_heatmap(analysis_dir, output_dir):
    """Heatmap of -log10(p) for each (language, metric) cell."""
    test_csv = os.path.join(analysis_dir, "hypothesis_tests.csv")
    if not os.path.exists(test_csv):
        print(
            f"hypothesis_tests.csv not found in {analysis_dir}. Run analyze.py first."
        )
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import math
    except ImportError:
        return

    # Load p-values
    table = defaultdict(dict)
    with open(test_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["p_value"])
                table[row["lang"]][row["metric"]] = p
            except (ValueError, KeyError):
                pass

    langs = sorted(table.keys())
    metrics = METRICS

    matrix = np.zeros((len(langs), len(metrics)))
    for i, lang in enumerate(langs):
        for j, metric in enumerate(metrics):
            p = table[lang].get(metric, 1.0)
            p = max(p, 1e-300)  # prevent log(0)
            matrix[i, j] = -math.log10(p)

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.4 + 1, len(langs) * 0.8 + 1))
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)

    # Annotate cells
    for i in range(len(langs)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            text = f"{val:.1f}" if val < 300 else ">300"
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log₁₀(p-value)", rotation=270, labelpad=15)

    # Reference line for p = 0.05 → -log10 ≈ 1.3
    ax.set_title(
        "Statistical Significance: Real vs Random\n"
        "(higher = more significant; >1.3 ≈ p<0.05)"
    )

    fig.tight_layout()
    out = os.path.join(output_dir, "heatmap_pvalues.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    # Prompt user for input
    results_dir = (
        input("Enter results CSV directory (default: results): ").strip() or "results"
    )
    analysis_dir = (
        input("Enter analysis CSV directory (default: analysis): ").strip()
        or "analysis"
    )
    output_dir = (
        input("Enter output directory for plots (default: plots): ").strip() or "plots"
    )

    os.makedirs(output_dir, exist_ok=True)
    data = load_results(results_dir)

    if not data:
        print("No data found. Run compute_metrics.py first.")
        return

    violin_plots(data, output_dir)
    cross_language_bar(data, output_dir)
    pvalue_heatmap(analysis_dir, output_dir)
    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()
