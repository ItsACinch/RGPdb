"""Acceptance: predicted schedules THROUGH the production `schedule` param beat baseline.

Trains the reference SchedulePredictor, then for each MetaQA 3-hop test question scores
via core.propagate(..., depth_weights=terminal(3), schedule=predicted) and asserts
3-hop MRR > 0.381 (the deployed terminal(3) baseline; expected ~0.878). This re-confirms
the measure-first gate through the real production API, not the ad-hoc __START__ matrix.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_schedule_api.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr
from rgdb_eval.schedule_predictor import SchedulePredictor

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
BASELINE = 0.381


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def trained_vocab(graph):
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
    return core.vocab_from_matrix(list(graph.relations), M.ravel().tolist())


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = trained_vocab(graph)

    # Train the reference predictor on all train hops (question text -> qtype).
    q_train, y_train = [], []
    for hop in (1, 2, 3):
        qs = read_lines(os.path.join(DATA, f"qa_train_{hop}hop.txt"))
        ts = read_lines(os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt"))
        for line, t in zip(qs, ts):
            q_train.append(line.split("\t")[0]); y_train.append(t)
    predictor = SchedulePredictor(graph.relation_to_id).fit(q_train, y_train)

    term3 = [0.0, 0.0, 0.0, 1.0, 0.0]
    # Question objects carry `.text` (the question string incl. the [entity] span),
    # which the predictor masks internally.
    qs = load_questions(os.path.join(DATA, "qa_test_3hop.txt"), 3, graph,
                        limit=None, qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"))

    items_pred, items_base = [], []
    for q in qs:
        sched = predictor.predict_schedule(q.text)
        ranked_p = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], None, MD, MI, term3, sched))
        ranked_p.pop(q.topic_id, None)
        rp = [n for n, _ in sorted(ranked_p.items(), key=lambda kv: -kv[1])][:20]
        items_pred.append((rp, set(q.answer_ids)))
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        ranked_b = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel, MD, MI, term3, None))
        ranked_b.pop(q.topic_id, None)
        rb = [n for n, _ in sorted(ranked_b.items(), key=lambda kv: -kv[1])][:20]
        items_base.append((rb, set(q.answer_ids)))

    mrr_p = mean(mrr(r, gs) for r, gs in items_pred)
    mrr_b = mean(mrr(r, gs) for r, gs in items_base)
    h1_p = mean(hits_at_k(r, gs, 1) for r, gs in items_pred)
    print(f"3-hop predicted-schedule (via production API): MRR {mrr_p:.4f}  Hits@1 {h1_p:.4f}")
    print(f"3-hop baseline terminal(3), no schedule:       MRR {mrr_b:.4f}")

    out = "results/metaqa-schedule-api.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Query-conditioned schedule via the production API (MetaQA 3-hop)\n\n"
                "Predicted schedule fed through `core.propagate(..., schedule=...)` with "
                "`depth_weights=terminal(3)`.\n\n"
                f"| condition | MRR | Hits@1 |\n|---|---|---|\n"
                f"| baseline terminal(3), no schedule | {mrr_b:.4f} | - |\n"
                f"| predicted-schedule (reference predictor) | {mrr_p:.4f} | {h1_p:.4f} |\n\n"
                "CAVEAT: MetaQA's 15 fixed 3-hop templates make the reference predictor "
                "near-perfect; this is NOT evidence that open-ended question->schedule "
                "prediction is easy. The mechanism's robustness (degradation sweep in "
                "experiment_predicted_schedule.py) is the transferable evidence.\n")
    print(f"wrote {out}")

    assert mrr_p > BASELINE, f"predicted-schedule MRR {mrr_p:.4f} did not beat baseline {BASELINE}"
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
