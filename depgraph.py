"""
depgraph.py
-----------
Parses CoNLL-U files into dependency tree objects and computes
structural metrics: arity, depth, density, average path length.
"""

import math
from collections import defaultdict, deque


class DependencyTree:
    """
    Represents a single sentence as a rooted dependency tree.

    Attributes
    ----------
    nodes   : list of int  — word indices (1-based, 0 = root sentinel)
    heads   : dict {node: head}  — head of each word (0 means root)
    labels  : dict {node: label} — dependency relation label
    words   : dict {node: str}   — surface form
    """

    def __init__(self):
        self.nodes = []          # word indices (1..n)
        self.heads = {}          # node -> head index
        self.labels = {}         # node -> dep label
        self.words = {}          # node -> word string
        self._children = None    # cached children dict

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_node(self, idx, word, head, label):
        self.nodes.append(idx)
        self.heads[idx] = head
        self.labels[idx] = label
        self.words[idx] = word
        self._children = None   # invalidate cache

    @property
    def n(self):
        return len(self.nodes)

    @property
    def root(self):
        """Return the node whose head is 0 (the root word)."""
        for node, head in self.heads.items():
            if head == 0:
                return node
        return None

    @property
    def children(self):
        """Build children dict lazily and cache it."""
        if self._children is None:
            self._children = defaultdict(list)
            for node, head in self.heads.items():
                if head != 0:
                    self._children[head].append(node)
        return self._children

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    def arity(self):
        """
        Returns a dict of per-node outdegree (number of dependents).
        Useful for computing max and mean arity.
        """
        degrees = {node: len(self.children[node]) for node in self.nodes}
        return degrees

    def max_arity(self):
        return max(self.arity().values(), default=0)

    def mean_arity(self):
        deg = list(self.arity().values())
        return sum(deg) / len(deg) if deg else 0.0

    def depth_of(self, node):
        """Depth of a single node (root = 0)."""
        d = 0
        cur = node
        visited = set()
        while self.heads.get(cur, 0) != 0:
            cur = self.heads[cur]
            d += 1
            if cur in visited:
                break          # guard against malformed trees
            visited.add(cur)
        return d

    def all_depths(self):
        return {node: self.depth_of(node) for node in self.nodes}

    def max_depth(self):
        return max(self.all_depths().values(), default=0)

    def mean_depth(self):
        depths = list(self.all_depths().values())
        return sum(depths) / len(depths) if depths else 0.0

    def density(self):
        """
        Graph density = E / (V*(V-1)).
        For a tree E = V-1, so density = 1/(V) exactly.
        But we include it so real vs random trees of DIFFERENT
        actual edge counts (e.g. forests) can still be compared.
        """
        n = self.n
        if n <= 1:
            return 0.0
        e = len(self.heads)   # each word has exactly one head entry
        return e / (n * (n - 1))

    def avg_path_length(self):
        """
        Mean shortest-path length between all pairs of nodes,
        treating the tree as an undirected graph.
        Uses BFS from each node — O(n^2) but fine for sentences.
        """
        if self.n <= 1:
            return 0.0

        # Build undirected adjacency list
        adj = defaultdict(list)
        for node, head in self.heads.items():
            if head != 0:
                adj[node].append(head)
                adj[head].append(node)

        total, count = 0, 0
        for source in self.nodes:
            dist = {source: 0}
            q = deque([source])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for node in self.nodes:
                if node != source and node in dist:
                    total += dist[node]
                    count += 1

        return total / count if count else 0.0

    def all_metrics(self):
        """Return a dict of all scalar metrics for this tree."""
        return {
            "n_nodes":        self.n,
            "max_arity":      self.max_arity(),
            "mean_arity":     self.mean_arity(),
            "max_depth":      self.max_depth(),
            "mean_depth":     self.mean_depth(),
            "density":        self.density(),
            "avg_path_length": self.avg_path_length(),
        }

    def __repr__(self):
        return f"DependencyTree(n={self.n}, root={self.root})"


# ======================================================================
# CoNLL-U parser
# ======================================================================

def parse_conllu(filepath):
    """
    Generator that yields one DependencyTree per sentence in a CoNLL-U file.

    CoNLL-U columns (1-indexed):
      1  ID    2  FORM  3  LEMMA  4  UPOS  5  XPOS
      6  FEATS 7  HEAD  8  DEPREL 9  DEPS  10 MISC

    Skips multi-word tokens (lines like "1-2") and empty nodes ("1.1").
    """
    tree = None
    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                # Sentence boundary
                if tree is not None and tree.n > 0:
                    yield tree
                tree = DependencyTree()
                continue

            if tree is None:
                tree = DependencyTree()

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            tok_id = parts[0]
            # Skip multi-word tokens and empty nodes
            if "-" in tok_id or "." in tok_id:
                continue

            try:
                idx = int(tok_id)
                word = parts[1]
                head = int(parts[6]) if parts[6] != "_" else 0
                label = parts[7]
                tree.add_node(idx, word, head, label)
            except ValueError:
                continue

    # Yield last tree if file doesn't end with blank line
    if tree is not None and tree.n > 0:
        yield tree


def load_treebank(filepath):
    """Load all trees from a CoNLL-U file into a list."""
    return list(parse_conllu(filepath))
