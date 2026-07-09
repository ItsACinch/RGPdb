"""Deterministic synthetic probe graphs for controlled refraction ablations."""
from __future__ import annotations
import random

from .dataset import TypedGraph, Question


def make_probe(kind: str, seed: int, n_relations: int = 6,
               depth: int = 3, noise_nodes: int = 50):
    if kind not in ("coherent", "incoherent"):
        raise ValueError(f"unknown probe kind: {kind}")
    rng = random.Random(seed)

    # Nodes 0..depth are the planted path; the rest are noise.
    n_nodes = depth + 1 + noise_nodes
    relations = [f"r{i}" for i in range(n_relations)]
    edges: list[tuple[int, int, int]] = []

    # Planted path 0 -> 1 -> ... -> depth.
    # Successor of a node on the path is always the next id, so `min(out_neighbors)`
    # in the tests deterministically follows the planted edge.
    if kind == "coherent":
        path_rel = rng.randrange(n_relations)
        for i in range(depth):
            edges.append((i, i + 1, path_rel))
    else:  # incoherent: cycle through distinct relations
        for i in range(depth):
            edges.append((i, i + 1, i % n_relations))

    # Noise edges among the higher-id nodes, none re-entering the planted path.
    noise_start = depth + 1
    for u in range(noise_start, n_nodes):
        for _ in range(2):
            v = rng.randrange(noise_start, n_nodes)
            if v != u:
                edges.append((u, v, rng.randrange(n_relations)))

    names = [f"n{i}" for i in range(n_nodes)]
    g = TypedGraph(num_nodes=n_nodes, entity_names=names,
                   relations=relations, edges=edges)
    q = Question(text=f"{kind} probe", topic_id=0, answer_ids=[depth],
                 relation=relations[edges[0][2]] if kind == "coherent" else None,
                 hop=depth)
    return g, [q]
