"""
treegen.py
----------
Generates random labeled trees with n nodes using Prüfer sequences.
A Prüfer sequence of length n-2 uniquely encodes a labeled tree on n nodes.
Sampling a Prüfer sequence uniformly at random gives a uniformly random
labeled tree (Cayley's formula).

We then root the random tree at a random node and orient all edges away
from the root, mirroring the structure of a dependency tree.
"""

import random
from depgraph import DependencyTree


def prufer_to_tree(prufer_seq, n_nodes):
    """
    Convert a Prüfer sequence to a rooted DependencyTree.

    Parameters
    ----------
    prufer_seq : list of int  — sequence of length n_nodes - 2
                               with values in [1, n_nodes]
    n_nodes    : int          — number of nodes (words) in the tree

    Returns
    -------
    DependencyTree with nodes 1..n_nodes, rooted at a random node.
    """
    if n_nodes == 1:
        t = DependencyTree()
        t.add_node(1, f"w1", 0, "root")
        return t

    if n_nodes == 2:
        t = DependencyTree()
        t.add_node(1, "w1", 0, "root")
        t.add_node(2, "w2", 1, "dep")
        return t

    # --- Step 1: decode Prüfer → undirected edge list ---
    degree = [1] * (n_nodes + 1)   # 1-indexed, degree starts at 1
    for v in prufer_seq:
        degree[v] += 1

    edges = []
    seq = list(prufer_seq)
    for v in seq:
        # Find smallest leaf (degree == 1)
        for u in range(1, n_nodes + 1):
            if degree[u] == 1:
                edges.append((u, v))
                degree[u] -= 1
                degree[v] -= 1
                break

    # Last edge connects the two remaining nodes with degree 1
    remaining = [u for u in range(1, n_nodes + 1) if degree[u] == 1]
    if len(remaining) == 2:
        edges.append((remaining[0], remaining[1]))

    # --- Step 2: build adjacency list ---
    adj = {i: [] for i in range(1, n_nodes + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # --- Step 3: choose a random root and orient edges via BFS ---
    root = random.randint(1, n_nodes)
    heads = {}
    visited = set([root])
    queue = [root]
    heads[root] = 0   # root has no head

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                heads[neighbor] = node
                queue.append(neighbor)

    # --- Step 4: pack into DependencyTree ---
    tree = DependencyTree()
    for idx in range(1, n_nodes + 1):
        head = heads.get(idx, 0)
        tree.add_node(idx, f"w{idx}", head, "dep" if head != 0 else "root")

    return tree


def random_prufer_tree(n_nodes, seed=None):
    """
    Generate one uniformly random labeled tree with n_nodes nodes.

    Parameters
    ----------
    n_nodes : int
    seed    : optional random seed for reproducibility

    Returns
    -------
    DependencyTree
    """
    if seed is not None:
        random.seed(seed)

    if n_nodes <= 2:
        return prufer_to_tree([], n_nodes)

    seq = [random.randint(1, n_nodes) for _ in range(n_nodes - 2)]
    return prufer_to_tree(seq, n_nodes)


def generate_random_trees(n_nodes, k=100):
    """
    Generate k random trees all having n_nodes nodes.

    Parameters
    ----------
    n_nodes : int  — must match the real sentence length
    k       : int  — how many random trees to generate per sentence

    Returns
    -------
    list of DependencyTree
    """
    return [random_prufer_tree(n_nodes) for _ in range(k)]
