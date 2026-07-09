"""Untyped personalized PageRank baseline (scipy sparse)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from ..dataset import TypedGraph


class PPRRanker:
    name = "untyped-ppr"

    def __init__(self, graph: TypedGraph, damping: float = 0.85, iters: int = 30):
        self.graph = graph
        self.damping = damping
        self.iters = iters
        n = graph.num_nodes
        if graph.edges:
            rows = np.fromiter((s for (s, _, _) in graph.edges), dtype=np.int64)
            cols = np.fromiter((d for (_, d, _) in graph.edges), dtype=np.int64)
            data = np.ones(len(graph.edges), dtype=np.float64)
            adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            adj = sp.csr_matrix((n, n), dtype=np.float64)
        # Row-normalize to a transition matrix (dangling rows stay zero).
        out = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
        self.trans = sp.diags(inv) @ adj  # row-stochastic where out>0

    def rank(self, seeds, query_relation, k):
        n = self.graph.num_nodes
        if not seeds:
            return []
        restart = np.zeros(n, dtype=np.float64)
        restart[seeds] = 1.0 / len(seeds)
        scores = restart.copy()
        tt = self.trans.T.tocsr()
        for _ in range(self.iters):
            scores = self.damping * (tt @ scores) + (1 - self.damping) * restart
        order = np.argsort(-scores)
        return [int(i) for i in order[:k]]
