"""
compute_metrics.py
------------------
Main pipeline for Project LE2.

Usage
-----
    python compute_metrics.py --conllu path/to/file.conllu --lang English --output results/
    python compute_metrics.py --conllu data/*.conllu --lang English --output results/

For each sentence in the CoNLL-U file:
  1. Parse real dependency tree
  2. Generate K random trees with same number of nodes
  3. Compute metrics for both
  4. Write results to a long-format CSV

Output CSV columns
------------------
lang, sent_id, tree_type (real/random), n_nodes,
max_arity, mean_arity, max_depth, mean_depth, density, avg_path_length
"""

import os
import csv
import argparse
import glob
from depgraph import load_treebank
from treegen import generate_random_trees

# Number of random trees to generate per sentence
K_RANDOM = 100

FIELDNAMES = [
    "lang",
    "sent_id",
    "tree_type",
    "n_nodes",
    "max_arity",
    "mean_arity",
    "max_depth",
    "mean_depth",
    "density",
    "avg_path_length",
]


def process_treebank(conllu_path, lang, k=K_RANDOM):
    """
    Process one CoNLL-U file.

    Returns
    -------
    list of dicts, one per (sentence × tree_type) row
    """
    trees = load_treebank(conllu_path)
    rows = []

    for sent_id, tree in enumerate(trees, start=1):
        if tree.n < 2:
            continue  # skip trivial sentences

        # --- Real tree ---
        m = tree.all_metrics()
        rows.append(
            {
                "lang": lang,
                "sent_id": sent_id,
                "tree_type": "real",
                **m,
            }
        )

        # --- Random trees ---
        rand_trees = generate_random_trees(tree.n, k=k)
        for rt in rand_trees:
            rm = rt.all_metrics()
            rows.append(
                {
                    "lang": lang,
                    "sent_id": sent_id,
                    "tree_type": "random",
                    **rm,
                }
            )

        if sent_id % 500 == 0:
            print(f"  [{lang}] processed {sent_id} sentences...")

    return rows


def main():

    # Prompt user for input
    file_path = input("Enter path to CoNLL-U file: ").strip()
    lang = input("Enter language label (e.g. English, Hindi, French): ").strip()
    output_dir = "Outputs"
    k = 100
    k = int(k) if k else K_RANDOM

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{lang}.csv")

    print(f"\n=== Processing {lang} ===")

    all_rows = []
    print(f"  File: {file_path}")
    rows = process_treebank(file_path, lang, k=k)
    all_rows.extend(rows)
    print(f"  → {len(rows)} rows written")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved: {out_path}  ({len(all_rows)} rows total)\n")


if __name__ == "__main__":
    main()
