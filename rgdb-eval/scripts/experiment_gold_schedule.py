"""Option C validation gate: the gold-schedule upper bound.

Gives the kernel PERFECT knowledge of each question's reasoning chain, by building a
per-question relation matrix that rewards exactly the expected relation at each hop.
No kernel change needed: the kernel's `r_in` is the relation traversed on the previous
hop, so seeding with a synthetic __START__ relation and setting

    M[__START__][r1] = M[r1][r2] = M[r2][r3] = 1.0   (everything else = floor)

makes hop k score `sim(expected[k-1], edge)` -- i.e. an exact per-hop schedule.
Two variants:
  soft (floor=0.05): off-schedule hops are penalized 20x but still carry mass
  hard (floor=0.00): off-schedule hops carry NO mass -- the strict ceiling

GATE (docs/superpowers/specs/2026-07-10-deferred-query-conditioned-refraction.md):
if 3-hop MRR with a gold schedule does not clearly beat untyped-ppr's 0.279, ABANDON C.
Perfect chain knowledge is the best case; if the best case loses, predicting the chain
imperfectly certainly will.

VALIDITY CHECK (runs first): follow the gold relation chain from the topic entity by
plain BFS. If that does not reach the gold answers, the qtype->relation mapping is
wrong and every number below is meaningless. Must be high before trusting the gate.

DIAGNOSTIC: separates the two possible failure modes --
  (a) relation model is wrong -> deep recall (Hits@20) does NOT improve either
  (b) k-hop distance artifact -> recall improves but MRR/Hits@1 do not, because
      diffusion accumulates mass at every visited node, so nearer intermediates
      outrank the k-hop answer

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_gold_schedule.py
"""
from __future__ import annotations
import os
from collections import defaultdict
from statistics import mean

from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.dataset import Question
from rgdb_eval.metaqa import (
    load_kb, parse_qa_line, query_relation_from_qtype, qtype_to_relation_sequence,
)
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr, K_VALUES
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker
from rgdb_eval.report import evaluate, to_markdown

DATA = "data/MetaQA"
TEST_LIMIT = 1000
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FRONTIER_CAP = 100_000  # guards the validity BFS against a pathological blow-up


def read_qtypes(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f]


def schedule_matrix(n_real: int, sched: list[int], floor: float) -> list[float]:
    """(n_real+1)^2 row-major matrix implementing a per-hop expected-relation schedule.

    Relation id `n_real` is the synthetic __START__, used only as the query relation;
    no graph edge carries it, so it can never be matched as an `r_out`.
    """
    n = n_real + 1
    start = n_real
    m = [floor] * (n * n)
    m[start * n + sched[0]] = 1.0
    for a, b in zip(sched, sched[1:]):
        m[a * n + b] = 1.0
    return m


def load_with_schedules(qpath: str, tpath: str, hop: int, graph) -> list:
    """Mirrors load_questions()'s filtering, but keeps the raw line index so the
    line-aligned qtype file stays in step, and returns the full gold chain."""
    qtypes = read_qtypes(tpath)
    rows = []
    with open(qpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            topic, answers = parse_qa_line(line)
            if topic not in graph.name_to_id:
                continue
            answer_ids = [graph.name_to_id[a] for a in answers if a in graph.name_to_id]
            if not answer_ids:
                continue
            qt = qtypes[i] if i < len(qtypes) else ""
            # `relation` must match load_questions exactly: the FIRST-hop relation,
            # kept even when the full chain is unresolvable. Baselines depend on it.
            first = query_relation_from_qtype(qt)
            relation = first if first in graph.relation_to_id else None
            sched = [graph.relation_to_id[r]
                     for r in qtype_to_relation_sequence(qt)
                     if r in graph.relation_to_id]
            rows.append((Question(text=line.split("\t")[0],
                                  topic_id=graph.name_to_id[topic],
                                  answer_ids=answer_ids, relation=relation, hop=hop),
                         sched))
            if len(rows) >= TEST_LIMIT:
                break
    return rows


def chain_reachability(graph, rows) -> tuple[float, float]:
    """Follow the gold relation chain from the topic by BFS.

    Returns (hit_rate, recall) over questions with a resolved schedule. If this is
    not high, the schedule derivation is broken and the gate below means nothing.
    """
    by_src_rel: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (s, d, r) in graph.edges:
        by_src_rel[(s, r)].append(d)

    hits, recalls = [], []
    for q, sched in rows:
        if not sched:
            continue
        frontier = {q.topic_id}
        for r in sched:
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(by_src_rel.get((u, r), ()))
                if len(nxt) > FRONTIER_CAP:
                    break
            frontier = nxt
            if not frontier:
                break
        gold = set(q.answer_ids)
        hits.append(1.0 if frontier & gold else 0.0)
        recalls.append(len(frontier & gold) / len(gold))
    if not hits:
        return 0.0, 0.0
    return mean(hits), mean(recalls)


def score_gold(g, names, n_real, per_hop, floor: float, label: str) -> dict:
    k_max = max(K_VALUES)
    start_id = n_real
    row = {"ranker": label}
    for hop, rows in sorted(per_hop.items()):
        items = []
        for q, sched in rows:
            gold = set(q.answer_ids)
            if not sched:
                items.append(([], gold))  # unresolvable schedule scores zero, not skipped
                continue
            vocab = core.vocab_from_matrix(names, schedule_matrix(n_real, sched, floor))
            totals = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], start_id,
                                         MAX_DEPTH, MIN_INTENSITY))
            totals.pop(q.topic_id, None)  # uniform seed exclusion, as report.evaluate does
            ranked = [n for n, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
            items.append((ranked, gold))
        s = {}
        for k in K_VALUES:
            s[f"hits@{k}"] = mean(hits_at_k(r, gs, k) for r, gs in items)
            s[f"recall@{k}"] = mean(recall_at_k(r, gs, k) for r, gs in items)
        s["mrr"] = mean(mrr(r, gs) for r, gs in items)
        s["n"] = len(items)
        row[f"hop{hop}"] = s
        print(f"  scored {label} hop{hop}")
    return row


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n_real = len(graph.relations)
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n_real} relations")

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    names = list(graph.relations) + ["__START__"]

    per_hop: dict[int, list] = {}
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_test_{hop}hop.txt")
        tpath = os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt")
        if not (os.path.exists(qpath) and os.path.exists(tpath)):
            continue
        rows = load_with_schedules(qpath, tpath, hop, graph)
        per_hop[hop] = rows
        resolved = sum(1 for _, s in rows if s)
        lens = sorted({len(s) for _, s in rows if s})
        print(f"hop{hop}: {len(rows)} questions, {resolved} with a gold schedule, chain lengths {lens}")

    # ---- VALIDITY: does the gold chain actually reach the gold answers? ----
    print("\nvalidity check (BFS along the gold relation chain):")
    valid = True
    for hop, rows in sorted(per_hop.items()):
        hit, rec = chain_reachability(graph, rows)
        flag = "ok" if hit >= 0.95 else "SUSPECT"
        if hit < 0.95:
            valid = False
        print(f"  hop{hop}: chain reaches a gold answer {hit:.3f} of the time, recall {rec:.3f}  [{flag}]")
    if not valid:
        print("\n!! The gold chain does not reliably reach the gold answers.")
        print("!! The qtype->relation mapping is wrong; the gate below is NOT meaningful.")

    # ---- Contenders ----
    questions = [q for rows in per_hop.values() for q, _ in rows]
    baselines = [PPRRanker(graph), NewRgdbRanker(graph, vocab_mode="uniform")]
    rows_all = []
    for b in baselines:
        rows_all.append(evaluate(b, questions))
        print(f"  scored {b.name}")
    rows_all.append(score_gold(g, names, n_real, per_hop, 0.05, "rgdb-gold-schedule-soft"))
    rows_all.append(score_gold(g, names, n_real, per_hop, 0.00, "rgdb-gold-schedule-hard"))

    md = "# Option C gate: gold-schedule upper bound (MetaQA)\n" + to_markdown(rows_all)
    out = "results/metaqa-gold-schedule.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}\n")
    print(md)

    # ---- The gate ----
    by = {r["ranker"]: r for r in rows_all}
    ppr = by["untyped-ppr"]["hop3"]
    soft = by["rgdb-gold-schedule-soft"]["hop3"]
    hard = by["rgdb-gold-schedule-hard"]["hop3"]
    best = max(soft["mrr"], hard["mrr"])
    print("=" * 72)
    print(f"GATE  3-hop MRR: gold-soft {soft['mrr']:.3f}  gold-hard {hard['mrr']:.3f}"
          f"   vs  untyped-ppr {ppr['mrr']:.3f}")
    print(f"GATE  {'PASS -> C is worth designing' if best > ppr['mrr'] else 'FAIL -> ABANDON C'}")

    # ---- Diagnostic: which failure mode? ----
    print("-" * 72)
    print(f"3-hop Hits@1  : soft {soft['hits@1']:.3f}  hard {hard['hits@1']:.3f}  ppr {ppr['hits@1']:.3f}")
    print(f"3-hop Hits@20 : soft {soft['hits@20']:.3f}  hard {hard['hits@20']:.3f}  ppr {ppr['hits@20']:.3f}")
    print(f"3-hop recall@20: soft {soft['recall@20']:.3f}  hard {hard['recall@20']:.3f}  ppr {ppr['recall@20']:.3f}")
    best_h20 = max(soft["hits@20"], hard["hits@20"])
    if best_h20 > ppr["hits@20"] and best <= ppr["mrr"]:
        print("\nDIAGNOSIS: the relation model WORKS -- a gold schedule finds the answer more")
        print("           often than PPR (higher Hits@20) -- but diffusion accumulates mass at")
        print("           EVERY visited node, so nearer intermediates outrank the k-hop answer.")
        print("           This is the DISTANCE artifact, not a relation-model failure.")
        print("           Fixing it needs terminal-node bias / exactly-k-hop restriction,")
        print("           which is ORTHOGONAL to C. C alone would not deliver the MRR win.")
    elif best_h20 <= ppr["hits@20"]:
        print("\nDIAGNOSIS: even perfect chain knowledge does not improve retrieval at all ->")
        print("           the relation model is not the bottleneck. Abandon C outright.")
    else:
        print("\nDIAGNOSIS: a gold schedule improves both recall and ranking -> C has headroom.")
    print("=" * 72)


if __name__ == "__main__":
    main()
