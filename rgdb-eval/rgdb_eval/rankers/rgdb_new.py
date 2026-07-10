"""Post-rewrite RGDB contender: sparse typed-PPR with optional refraction."""
from __future__ import annotations
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from ..dataset import TypedGraph
from ..embeddings import embed_texts


class NewRgdbRanker:
    def __init__(self, graph: TypedGraph, vocab_mode: str = "refraction"):
        if vocab_mode not in ("refraction", "uniform"):
            raise ValueError(vocab_mode)
        self.graph = graph
        self.vocab_mode = vocab_mode
        self.name = f"rgdb-new-{vocab_mode}"

        adj = [[] for _ in range(graph.num_nodes)]
        for (s, d, rel_id) in graph.edges:
            adj[s].append((d, 0.0, rel_id))
        self._g = core.build_graph(graph.num_nodes, adj)

        n = len(graph.relations)
        if vocab_mode == "uniform":
            self._vocab = core.uniform_vocab(n)
        else:
            rel_vecs = embed_texts(list(graph.relations))
            sim = (rel_vecs @ rel_vecs.T).clip(0.0, 1.0).astype(np.float32)
            self._vocab = core.vocab_from_matrix(list(graph.relations), sim.ravel().tolist())

    def rank(self, seeds, query_relation, k):
        if not seeds:
            return []
        rel_id = self.graph.relation_to_id.get(query_relation) if query_relation else None
        seed_pairs = [(int(s), 1.0 / len(seeds)) for s in seeds]
        totals = dict(core.propagate(self._g, self._vocab, seed_pairs, rel_id, 4, 1e-4))
        seed_set = set(seeds)
        items = [(node, sc) for node, sc in totals.items() if node not in seed_set]
        items.sort(key=lambda x: -x[1])
        return [int(n) for n, _ in items[:k]]
