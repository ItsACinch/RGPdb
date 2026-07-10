"""Diagnostic: is the 3-hop MRR gap a DISTANCE artifact or a RELATION-model failure?

The gold-schedule gate (experiment_gold_schedule.py) found that a perfect per-hop
relation schedule more than doubles 3-hop recall@20 (0.578 vs PPR's 0.262) yet still
LOSES on MRR (0.254 vs 0.279). Two explanations:

  (a) relation-model failure -- the schedule doesn't really help, recall is a fluke
  (b) distance artifact -- propagate() accumulates intensity at EVERY visited node,
      so a hop-1 neighbour always outranks a hop-3 answer regardless of relation

This isolates them by controlling for distance: restrict the candidate set to nodes
lying exactly k chain-steps from the topic (the BFS terminal set of the gold chain),
then rank within that set. Both contenders get the SAME mask and the same code path.

  - If gold-schedule >> uniform under the mask, the relation model carries real signal
    and the MRR loss is purely a distance/accumulation artifact.
  - If they are comparable, the relation model adds nothing and C is dead outright.

NOT a deployable number. The mask uses gold knowledge of k and of the chain; it is an
upper bound whose only job is to attribute the cause.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_distance_artifact.py
"""
from __future__ import annotations
import os
from collections import defaultdict
from statistics import mean

from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr

DATA = "data/MetaQA"
TEST_LIMIT = 1000
HOP = 3
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    start_id = n
    names = list(graph.relations) + ["__START__"]

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    by_src_rel: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
        by_src_rel[(s, r)].append(d)
    g = core.build_graph(graph.num_nodes, adj)
    uniform = core.uniform_vocab(n)

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
            if len(rows) >= TEST_LIMIT:
                break
    print(f"{len(rows)} 3-hop questions with a full gold chain")

    def hard_matrix(sched):
        m = [0.0] * ((n + 1) * (n + 1))
        m[start_id * (n + 1) + sched[0]] = 1.0
        for a, b in zip(sched, sched[1:]):
            m[a * (n + 1) + b] = 1.0
        return m

    def terminal_set(topic, sched):
        frontier = {topic}
        for r in sched:
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(by_src_rel.get((u, r), ()))
            frontier = nxt
            if not frontier:
                break
        return frontier

    # Untyped out-adjacency, for the pure graph-distance mask.
    out_any: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for (s, d, _r) in graph.edges:
        out_any[s].append(d)

    def exact_distance_set(topic, k):
        """Nodes whose SHORTEST-path distance from topic is exactly k (relations ignored).

        This is what a deployable distance fix could actually use: it needs k, but not
        the gold chain. Stricter than the chain-terminal mask in one direction (a gold
        answer reachable by a 1-hop shortcut is excluded) and looser in another.
        """
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

    def ranked_from(scores: dict, cands, seed) -> list[int]:
        items = [(c, scores.get(c, 0.0)) for c in cands if c != seed]
        items.sort(key=lambda x: (-x[1], x[0]))
        return [c for c, _ in items]

    masked_u, masked_g, open_u, open_g = [], [], [], []
    dist_u, dist_g = [], []
    tsizes, gold_in_t, dsizes, gold_in_d = [], [], [], []

    for topic, gold, sched in rows:
        su = dict(core.propagate(g, uniform, [(topic, 1.0)], None, MAX_DEPTH, MIN_INTENSITY))
        vg = core.vocab_from_matrix(names, hard_matrix(sched))
        sg = dict(core.propagate(g, vg, [(topic, 1.0)], start_id, MAX_DEPTH, MIN_INTENSITY))

        T = terminal_set(topic, sched)
        tsizes.append(len(T))
        gold_in_t.append(len(T & gold) / len(gold))

        D = exact_distance_set(topic, HOP)
        dsizes.append(len(D))
        gold_in_d.append(len(D & gold) / len(gold))

        masked_u.append((ranked_from(su, T, topic), gold))
        masked_g.append((ranked_from(sg, T, topic), gold))
        dist_u.append((ranked_from(su, D, topic), gold))
        dist_g.append((ranked_from(sg, D, topic), gold))

        allc_u = [k for k in su if k != topic]
        allc_g = [k for k in sg if k != topic]
        open_u.append((ranked_from(su, allc_u, topic), gold))
        open_g.append((ranked_from(sg, allc_g, topic), gold))

    def report(label, items):
        return (label,
                mean(hits_at_k(r, gs, 1) for r, gs in items),
                mean(hits_at_k(r, gs, 20) for r, gs in items),
                mean(recall_at_k(r, gs, 20) for r, gs in items),
                mean(mrr(r, gs) for r, gs in items))

    stats = [
        report("uniform, no mask", open_u),
        report("gold-schedule, no mask", open_g),
        report("uniform + graph-distance==3 mask", dist_u),
        report("gold-schedule + graph-distance==3 mask", dist_g),
        report("uniform + gold-chain-terminal mask", masked_u),
        report("gold-schedule + gold-chain-terminal mask", masked_g),
    ]

    lines = ["# Distance artifact vs relation model (MetaQA 3-hop)\n",
             "\nThe two masks differ in how much gold knowledge they use:\n",
             "\n- **graph-distance==3**: nodes whose shortest-path distance from the topic is",
             "  exactly 3, relations ignored. Needs only `k` -- a deployable fix could use it.",
             f"  Mean candidates {mean(dsizes):.1f}; covers {mean(gold_in_d):.3f} of gold answers.",
             "\n- **gold-chain-terminal**: nodes reached by walking the exact gold relation chain.",
             "  Uses the answer's reasoning path; a strict upper bound, not deployable.",
             f"  Mean candidates {mean(tsizes):.1f}; covers {mean(gold_in_t):.3f} of gold answers.\n",
             "\nRecall@20 is bounded by each mask's gold coverage.\n",
             "\n| variant | hits@1 | hits@20 | recall@20 | mrr |",
             "|---|---|---|---|---|"]
    for label, h1, h20, r20, m in stats:
        lines.append(f"| {label} | {h1:.3f} | {h20:.3f} | {r20:.3f} | {m:.3f} |")
    md = "\n".join(lines) + "\n"

    out = "results/metaqa-distance-artifact.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}\n")
    print(md)

    ou, og = stats[0][4], stats[1][4]
    du, dg = stats[2][4], stats[3][4]
    tu, tg = stats[4][4], stats[5][4]
    print("=" * 72)
    print(f"no mask                 : uniform {ou:.3f}  gold-schedule {og:.3f}")
    print(f"graph-distance==3 mask  : uniform {du:.3f}  gold-schedule {dg:.3f}"
          f"   (gold coverage {mean(gold_in_d):.3f})")
    print(f"gold-chain-terminal mask: uniform {tu:.3f}  gold-schedule {tg:.3f}"
          f"   (gold coverage {mean(gold_in_t):.3f})")
    print("-" * 72)
    print(f"distance effect (uniform, no-mask -> distance mask): {ou:.3f} -> {du:.3f}")
    print(f"relation effect (under the distance mask)          : {du:.3f} -> {dg:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
