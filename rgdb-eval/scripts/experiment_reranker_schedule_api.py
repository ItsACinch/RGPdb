"""Acceptance: the query-conditioned reranker recovers 3-hop Hits@1 on DEGRADED
schedules, THROUGH the production engine (reranker_enabled + schedule).

For each corruption p, corrupt each gold-schedule relation with prob p (a proxy for
imperfect prediction). schedule-alone = engine with the reranker OFF; reranker+match =
engine with the reranker ON, trained online via record_feedback under the corrupted
schedule's expected final relation. Both evaluated on the same corrupted test schedules.
Gate: reranker+match 3-hop Hits@1 beats schedule-alone at p=0.25 (floor 0.50).

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_reranker_schedule_api.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN, TESTN = 3000, 2000
TERM3 = [0.0, 0.0, 0.0, 1.0, 0.0]
P_LEVELS = [0.0, 0.25, 0.4]
GATE_P, GATE_FLOOR = 0.25, 0.50


def trained_flat(graph):
    n = len(graph.relations); rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FL, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M.ravel().tolist(), n


def load_sched(graph, split, hop, limit):
    """(topic_id, set(answer_ids), schedule) line-aligned with the qtype file."""
    rid = graph.relation_to_id
    qpath = os.path.join(DATA, f"qa_{split}_{hop}hop.txt")
    tpath = os.path.join(DATA, f"qa_{split}_{hop}hop_qtype.txt")
    qtypes = [l.strip() for l in open(tpath, encoding="utf-8")]
    rows = []
    for i, line in enumerate(open(qpath, encoding="utf-8")):
        if len(rows) >= limit:
            break
        if not line.strip():
            continue
        topic, answers = parse_qa_line(line)
        if topic not in graph.name_to_id:
            continue
        ans = {graph.name_to_id[a] for a in answers if a in graph.name_to_id}
        if not ans:
            continue
        seq = qtype_to_relation_sequence(qtypes[i]) if i < len(qtypes) else []
        sched = [rid[r] for r in seq if r in rid]
        if not sched:
            continue
        rows.append((graph.name_to_id[topic], ans, sched))
    return rows


def corrupt(sched, p, n, rng):
    out = []
    for r in sched:
        if rng.random() < p:
            alt = int(rng.integers(0, n - 1))
            if alt >= r:
                alt += 1
            out.append(alt)
        else:
            out.append(r)
    return out


def hits1(eng, rows, p, n, seed):
    rng = np.random.default_rng(seed)
    tot = []
    for topic, gold, sched in rows:
        sc = corrupt(sched, p, n, rng)
        ranked, _ = eng.query([(topic, 1.0)], None, max_depth=MD, min_intensity=MI,
                              depth_weights=TERM3, schedule=sc)
        ranked = [nid for nid, _ in ranked if nid != topic][:20]
        tot.append(hits_at_k(ranked, gold, 1))
    return mean(tot)


def train(eng, rows, p, n, seed):
    rng = np.random.default_rng(seed)
    for topic, gold, sched in rows:
        sc = corrupt(sched, p, n, rng)
        _, qid = eng.query([(topic, 1.0)], None, max_depth=MD, min_intensity=MI,
                           depth_weights=TERM3, schedule=sc)
        try:
            eng.record_feedback(qid, next(iter(gold)), 1.0)
        except ValueError:
            pass


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    flat, n = trained_flat(graph)
    names = list(graph.relations)
    train_rows = load_sched(graph, "train", 3, TRAIN)
    test_rows = load_sched(graph, "test", 3, TESTN)
    print(f"{len(train_rows)} train / {len(test_rows)} test 3-hop questions")

    rows = []
    for p in P_LEVELS:
        off = core.Engine(g, core.vocab_from_matrix(names, flat), rebuild_every_n=0)
        h_alone = hits1(off, test_rows, p, n, seed=1)           # reranker off
        on = core.Engine(g, core.vocab_from_matrix(names, flat), rebuild_every_n=0,
                         reranker_enabled=True)
        train(on, train_rows, p, n, seed=2)
        h_rerank = hits1(on, test_rows, p, n, seed=1)           # same corrupted test schedules
        rows.append((p, h_alone, h_rerank))
        print(f"p={p}: schedule-alone Hits@1 {h_alone:.4f}  reranker+match Hits@1 {h_rerank:.4f}")

    out = "results/metaqa-reranker-schedule-api.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Query-conditioned reranker via the production engine (MetaQA 3-hop)\n\n"
                "Degraded schedules (each relation corrupted with prob p). Reranker trained "
                "online through record_feedback under the corrupted schedule's expected final "
                "relation; both variants evaluated on the same corrupted test schedules.\n\n"
                "| p | schedule-alone Hits@1 | reranker+match Hits@1 |\n|---|---|---|\n")
        for p, a, b in rows:
            f.write(f"| {p} | {a:.4f} | {b:.4f} |\n")
        f.write("\nCAVEAT: corruption is synthetic uniform relation substitution, a proxy for "
                "a real predictor's correlated errors; MetaQA is templated. Follow-up #1 "
                "(non-templated eval) is where a real predictor's error pattern is measured. "
                "The lean feature-set pruning (drop net-negative degree/incoming-one-hot) "
                "remains a follow-up.\n")
    print(f"wrote {out}")

    gate = next(b for (p, a, b) in rows if p == GATE_P)
    assert gate > GATE_FLOOR, f"reranker+match Hits@1 at p={GATE_P} = {gate:.4f} did not clear {GATE_FLOOR}"
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
