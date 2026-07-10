"""PIVOTAL: does ARRIVAL-depth weighting reproduce the SHORTEST-PATH-depth mask win?

The 0.419 3-hop MRR came from masking candidates to shortest-path distance == 3.
The proposed `depth_weights` kernel change instead weights mass by the depth at
which it ARRIVES during the walk. These are not the same thing: with inverse edges,
a node at distance 1 also receives mass at depth 3 (movie -> actor -> movie -> actor).

If arrival-depth weighting reproduces the win, depth_weights is O(1) and free.
If it does not, the fix needs per-query BFS distance -- O(ball), which the
neighborhood-growth numbers (6641 nodes at depth 3) make prohibitive.

Reimplements propagate() in Python with per-depth accumulators, first asserting it
matches the Rust kernel exactly at c=[1,1,1,1].

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_arrival_depth.py
"""
from __future__ import annotations
import os
from collections import defaultdict
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, mrr

DATA = "data/MetaQA"
LIMIT = 300
HOP = 3
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
REFL = 0.85
FLOOR = 0.05


def build_trained(graph):
    n = len(graph.relations); rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = f"{DATA}/qa_train_{hop}hop_qtype.txt"
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
            for a, b in zip(ids, ids[1:]):
                counts[a][b] += 1
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M


def propagate_depth(adj, sim, seed, qrel, max_depth=MAX_DEPTH, min_int=MIN_INTENSITY):
    """Mirror of propagate_single, but returns node -> per-arrival-depth mass."""
    per = defaultdict(lambda: np.zeros(max_depth + 1, dtype=np.float64))
    per[seed][0] += 1.0
    frontier = {(seed, qrel): 1.0}
    for d in range(max_depth):
        nxt = defaultdict(float)
        for (u, r_in), mass in frontier.items():
            if mass < min_int:
                continue
            out = adj[u]
            if not out:
                continue
            p = 1.0 / len(out)          # attenuation == 0 everywhere in this KB
            for (v, rel) in out:
                s = 1.0 if r_in is None else float(sim[r_in][rel])
                t = mass * REFL * p * s
                if t < min_int:         # prune on FLOW magnitude, not on c_k * flow
                    continue
                per[v][d + 1] += t
                nxt[(v, rel)] += t
        frontier = dict(nxt)
        if not frontier:
            break
    return per


def main():
    g = load_kb(f"{DATA}/kb.txt")
    n = len(g.relations)
    adj = defaultdict(list)
    for (s, d, r) in g.edges:
        adj[s].append((d, r))
    adj = {u: v for u, v in adj.items()}
    adj = defaultdict(list, adj)

    trained = build_trained(g)
    G = core.build_graph(g.num_nodes, [[(d, 0.0, r) for (d, r) in adj[u]] for u in range(g.num_nodes)])
    trained_v = core.vocab_from_matrix(list(g.relations), trained.ravel().tolist())

    qt = [l.strip() for l in open(f"{DATA}/qa_test_{HOP}hop_qtype.txt", encoding="utf-8")]
    rows = []
    for i, line in enumerate(open(f"{DATA}/qa_test_{HOP}hop.txt", encoding="utf-8")):
        if len(rows) >= LIMIT:
            break
        topic, answers = parse_qa_line(line)
        if topic not in g.name_to_id:
            continue
        gold = {g.name_to_id[a] for a in answers if a in g.name_to_id}
        if not gold:
            continue
        sched = [g.relation_to_id[r] for r in qtype_to_relation_sequence(qt[i]) if r in g.relation_to_id]
        if len(sched) != HOP:
            continue
        rows.append((g.name_to_id[topic], gold, sched))

    # --- correctness: python mirror must match the Rust kernel at c = all-ones ---
    topic0, _, sched0 = rows[0]
    per = propagate_depth(adj, trained, topic0, sched0[0])
    rust = dict(core.propagate(G, trained_v, [(topic0, 1.0)], sched0[0], MAX_DEPTH, MIN_INTENSITY))
    mx = max(abs(per[k].sum() - rust.get(k, 0.0)) for k in set(per) | set(rust))
    print(f"python mirror vs rust kernel, max abs diff at c=[1,1,1,1]: {mx:.2e}")
    assert mx < 1e-5, "python mirror does not match the Rust kernel"

    # --- where does mass at each arrival depth actually land? ---
    dist_of_arrival = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))  # arrival depth x shortest-path dist
    def sp_dist(seed):
        dist = {seed: 0}; fr = [seed]
        for k in range(1, MAX_DEPTH + 1):
            nx = []
            for u in fr:
                for (v, _r) in adj[u]:
                    if v not in dist:
                        dist[v] = k; nx.append(v)
            fr = nx
        return dist

    schemes = {
        "c=[1,1,1,1]  (today)":        np.array([1, 1, 1, 1, 1.0]),
        "c=[0,0,1,0]  terminal-3":     np.array([0, 0, 0, 1, 0.0]),
        "c=[0,0,1,1]  depth>=3":       np.array([0, 0, 0, 1, 1.0]),
        "c=[-1,-1,1,1] subtract near": np.array([0, -1, -1, 1, 1.0]),
    }
    res = {k: [] for k in schemes}
    masked = []

    for topic, gold, sched in rows:
        per = propagate_depth(adj, trained, topic, sched[0])
        dist = sp_dist(topic)
        for v, vec in per.items():
            dv = dist.get(v, MAX_DEPTH + 1)
            for k in range(MAX_DEPTH + 1):
                if vec[k] > 0:
                    dist_of_arrival[k][min(dv, MAX_DEPTH + 1)] += vec[k]
        for name, c in schemes.items():
            sc = {v: float(vec @ c) for v, vec in per.items() if v != topic}
            rk = [v for v, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:20]
            res[name].append((rk, gold))
        # reference: shortest-path mask on the unweighted score
        D = {v for v, dd in dist.items() if dd == HOP}
        sc = {v: float(per[v].sum()) for v in D if v != topic}
        rk = [v for v, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:20]
        masked.append((rk, gold))

    print(f"\n{len(rows)} 3-hop questions, trained matrix, seeded with first-hop relation\n")
    print("| readout | hits@1 | hits@20 | mrr |")
    print("|---|---|---|---|")
    for name in schemes:
        it = res[name]
        print(f"| {name} | {mean(hits_at_k(r,gs,1) for r,gs in it):.3f} "
              f"| {mean(hits_at_k(r,gs,20) for r,gs in it):.3f} "
              f"| {mean(mrr(r,gs) for r,gs in it):.3f} |")
    print(f"| shortest-path==3 mask (the 0.419 evidence) | "
          f"{mean(hits_at_k(r,gs,1) for r,gs in masked):.3f} | "
          f"{mean(hits_at_k(r,gs,20) for r,gs in masked):.3f} | "
          f"{mean(mrr(r,gs) for r,gs in masked):.3f} |")

    print("\nWhere does mass arriving at depth k actually live? (rows=arrival depth, cols=shortest-path dist)")
    hdr = "  arrival |" + "".join(f"  d={j}" for j in range(MAX_DEPTH + 2))
    print(hdr)
    for k in range(1, MAX_DEPTH + 1):
        tot = dist_of_arrival[k].sum()
        if tot == 0:
            continue
        cells = "".join(f" {100*dist_of_arrival[k][j]/tot:5.1f}" for j in range(MAX_DEPTH + 2))
        print(f"   k={k}   |{cells}")
    print("\n(row k, column d = % of depth-k arrival mass landing on nodes whose shortest-path distance is d)")


if __name__ == "__main__":
    main()
