"""Pre-rewrite RGDB contender using the current angle-bin bindings."""
from __future__ import annotations
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from ..dataset import TypedGraph

N_BINS = 16


class CurrentRgdbRanker:
    name = "rgdb-current"

    def __init__(self, graph: TypedGraph, relation_bins: dict[str, int] | None = None):
        self.graph = graph
        if relation_bins is None:
            relation_bins = {r: (i % N_BINS) for i, r in enumerate(graph.relations)}
        self.relation_bins = relation_bins
        # Build adjacency in binding format: adj[u] = [(dst, attenuation, angle_bin)]
        adj: list[list[tuple[int, float, int]]] = [[] for _ in range(graph.num_nodes)]
        for (s, d, rel_id) in graph.edges:
            bin_ = rel_id % N_BINS
            adj[s].append((d, 0.1, bin_))
        self._g = core.build_graph(graph.num_nodes, adj)

    def rank(self, seeds, query_relation, k):
        if not seeds:
            return []
        bin_ = self.relation_bins.get(query_relation, 0) if query_relation else 0
        acc = np.zeros(self.graph.num_nodes, dtype=np.float64)
        for s in seeds:
            acc += np.asarray(core.propagate_light(self._g, s, bin_), dtype=np.float64)
        order = np.argsort(-acc)
        return [int(i) for i in order[:k]]
