"""Vector-similarity contenders."""
from __future__ import annotations
from typing import Callable
import numpy as np

from ..dataset import TypedGraph


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class VectorRanker:
    name = "vector-only"

    def __init__(self, graph: TypedGraph, node_vecs: np.ndarray,
                 query_vec_fn: Callable[[str | None], np.ndarray]):
        self.graph = graph
        self.node_vecs = node_vecs
        self.query_vec_fn = query_vec_fn

    def _scores(self, query_relation):
        q = _unit(np.asarray(self.query_vec_fn(query_relation), dtype=np.float64))
        return self.node_vecs @ q

    def rank(self, seeds, query_relation, k):
        scores = self._scores(query_relation)
        order = np.argsort(-scores)
        return [int(i) for i in order[:k]]


class VectorTwoHopRanker(VectorRanker):
    name = "vector+2hop"

    def _candidates(self, seeds: list[int]) -> list[int]:
        frontier = set(seeds)
        seen = set(seeds)
        for _ in range(2):
            nxt = set()
            for u in frontier:
                for (d, _r) in self.graph.out_neighbors(u):
                    if d not in seen:
                        seen.add(d)
                        nxt.add(d)
            frontier = nxt
        return list(seen)

    def rank(self, seeds, query_relation, k):
        scores = self._scores(query_relation)
        cands = self._candidates(seeds)
        cands.sort(key=lambda i: -scores[i])
        return cands[:k]
