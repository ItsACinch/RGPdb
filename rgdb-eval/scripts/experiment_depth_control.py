"""What actually fixes 3-hop? Cross depth-control against three relation models.

The gate experiment showed a GOLD schedule + depth control hits 0.927 MRR. But a gold
schedule needs the answer's reasoning chain, which no deployment has. This asks the
deployable question:

  rows: relation model    uniform (none) | trained matrix (global marginal, needs only
                          the first-hop relation) | gold schedule (needs the full chain)
  cols: depth control     none | exactly-k graph-distance mask (needs only k)

The trained matrix is what ships today (option A). The interesting cell is
"trained + depth mask": if it lands near the gold schedule, depth-aware scoring is the
whole fix and no chain prediction is needed. If it stays near uniform, the relation
model must be query-conditioned and the deep-hop problem is genuinely hard.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_depth_control.py
"""
from __future__ import annotations
import os
from collections import defaultdict
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr

DATA = "data/MetaQA"
LIMIT = 1000
HOP = 3
FLOOR = 0.05
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4


def build_transition_matrix(graph) -> np.ndarray:
    """T[a][b] = normalized frequency that gold relation b follows a (from qa_train)."""
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n), dtype=np.float64)
    pairs = 0
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
                    pairs += 1
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        m = counts[a].max()
        if m > 0:
            M[a] = np.maximum(M[a], (counts[a] / m).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    print(f"trained matrix from {pairs} gold relation pairs")
    return M


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    start_id = n
    names_plus = list(graph.relations) + ["__START__"]

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    out_any: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
        out_any[s].append(d)
    g = core.build_graph(graph.num_nodes, adj)

    uniform_v = core.uniform_vocab(n)
    trained = build_transition_matrix(graph)
    trained_v = core.vocab_from_matrix(list(graph.relations), trained.ravel().tolist())

    with open(os.path.join(DATA, f"qa_test_{HOP}hop_qtype.txt"), encoding="utf-8") as f:
        qtypes = [ln.strip() for ln in f]

    rows = []
    with open(os.path.join(DATA, f"qa_test_{HOP}hop.txt"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            topic, answers = parse_qa_line(line)
            if topic not in graph.name_to_id:
                continue
            gold = {graph.name_to_id[a] for a in answers if a in graph.name_to_id}
            if not gold:
                continue
            sched = [graph.relation_to_id[r]
                     for r in qtype_to_relation_sequence(qtypes[i])
                     if r in graph.relation_to_id]
            if len(sched) != HOP:
                continue
            rows.append((graph.name_to_id[topic], gold, sched))
            if len(rows) >= LIMIT:
                break
    print(f"{len(rows)} 3-hop questions\n")

    def hard_matrix(sched):
        m = [0.0] * ((n + 1) * (n + 1))
        m[start_id * (n + 1) + sched[0]] = 1.0
        for a, b in zip(sched, sched[1:]):
            m[a * (n + 1) + b] = 1.0
        return m

    def exact_distance_set(topic, k):
        seen = {topic}
        frontier = [topic]
        for _ in range(k):
            nxt = []
            for u in frontier:
                for v in out_any[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        return set(frontier)

    def ranked(scores, cands, seed):
        items = [(c, scores.get(c, 0.0)) for c in cands if c != seed]
        items.sort(key=lambda x: (-x[1], x[0]))
        return [c for c, _ in items]

    cells = {k: [] for k in
             ["uniform|none", "uniform|depth",
              "trained|none", "trained|depth",
              "gold|none", "gold|depth"]}

    for topic, gold, sched in rows:
        su = dict(core.propagate(g, uniform_v, [(topic, 1.0)], None, MAX_DEPTH, MIN_INTENSITY))
        # trained matrix is seeded with the FIRST-HOP relation only -- what an intent
        # classifier can supply at query time. No chain knowledge.
        st = dict(core.propagate(g, trained_v, [(topic, 1.0)], sched[0], MAX_DEPTH, MIN_INTENSITY))
        vg = core.vocab_from_matrix(names_plus, hard_matrix(sched))
        sg = dict(core.propagate(g, vg, [(topic, 1.0)], start_id, MAX_DEPTH, MIN_INTENSITY))

        D = exact_distance_set(topic, HOP)
        for tag, sc in (("uniform", su), ("trained", st), ("gold", sg)):
            allc = [k for k in sc if k != topic]
            cells[f"{tag}|none"].append((ranked(sc, allc, topic), gold))
            cells[f"{tag}|depth"].append((ranked(sc, D, topic), gold))

    def stat(items):
        return (mean(hits_at_k(r, gs, 1) for r, gs in items),
                mean(hits_at_k(r, gs, 20) for r, gs in items),
                mean(recall_at_k(r, gs, 20) for r, gs in items),
                mean(mrr(r, gs) for r, gs in items))

    labels = {
        "uniform|none": "uniform, no depth control",
        "uniform|depth": "uniform + exactly-k mask",
        "trained|none": "trained matrix, no depth control",
        "trained|depth": "trained matrix + exactly-k mask",
        "gold|none": "gold schedule, no depth control",
        "gold|depth": "gold schedule + exactly-k mask",
    }
    lines = ["# What fixes 3-hop? depth control x relation model (MetaQA, 1000 q)\n",
             "\n`trained` uses only the first-hop relation (intent-classifiable).",
             "`gold` uses the full reasoning chain (not deployable).",
             "`exactly-k mask` uses only k (shortest-path distance), no relation info.\n",
             "\n| variant | hits@1 | hits@20 | recall@20 | mrr |", "|---|---|---|---|---|"]
    res = {}
    for key, lab in labels.items():
        h1, h20, r20, m = stat(cells[key])
        res[key] = m
        lines.append(f"| {lab} | {h1:.3f} | {h20:.3f} | {r20:.3f} | {m:.3f} |")
    md = "\n".join(lines) + "\n"

    with open("results/metaqa-depth-control.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    print("=" * 72)
    print("3-hop MRR grid")
    print(f"{'':<18}{'no depth':>12}{'+ depth mask':>16}")
    for tag in ("uniform", "trained", "gold"):
        print(f"{tag:<18}{res[tag+'|none']:>12.3f}{res[tag+'|depth']:>16.3f}")
    print("-" * 72)
    print(f"depth control alone (uniform)      : {res['uniform|none']:.3f} -> {res['uniform|depth']:.3f}")
    print(f"trained matrix alone               : {res['uniform|none']:.3f} -> {res['trained|none']:.3f}")
    print(f"DEPLOYABLE combo (trained + depth) : {res['trained|depth']:.3f}")
    print(f"ceiling (gold chain + depth)       : {res['gold|depth']:.3f}")
    print(f"reference: untyped PPR unmasked was 0.279")
    print("=" * 72)


if __name__ == "__main__":
    main()
