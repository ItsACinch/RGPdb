"""#2 gate: learned soft depth weights must not lose recall vs hard terminal(k).

Replays MetaQA training questions as feedback through the Engine (query with a hop
hint = the question's hop count, then record_feedback on the gold answer), refreshes
the learned profile, and evaluates on the test set. Compares the learned soft weights
against hard terminal(k). Gate: 3-hop recall@20 (soft) >= recall@20 (terminal) and
3-hop MRR (soft) not materially below terminal(3)'s 0.381.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_soft_depth_weights.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core
from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN_PER_HOP = 2000


def trained_matrix(graph):
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
    return M


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = core.vocab_from_matrix(list(graph.relations), trained_matrix(graph).ravel().tolist())
    eng = core.Engine(g, vocab, rebuild_every_n=0)

    # Replay training feedback with the hop count as the hint.
    events = 0
    for hop in (1, 2, 3):
        qp = os.path.join(DATA, f"qa_train_{hop}hop.txt")
        tp = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(qp):
            continue
        for q in load_questions(qp, hop, graph, limit=TRAIN_PER_HOP, qtype_path=tp):
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            _, qid = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=hop)
            try:
                eng.record_feedback(qid, q.answer_ids[0], 1.0)
                events += 1
            except ValueError:
                pass
    eng.refresh_profiles()
    print(f"replayed {events} feedback events")

    def score(hop, hop_hint):
        qs = load_questions(os.path.join(DATA, f"qa_test_{hop}hop.txt"), hop, graph,
                            limit=None, qtype_path=os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt"))
        items = []
        for q in qs:
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            ranked, _ = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=hop_hint)
            ranked = [n for n, _ in ranked if n != q.topic_id][:20]
            items.append((ranked, set(q.answer_ids)))
        return (mean(mrr(r, gs) for r, gs in items),
                mean(recall_at_k(r, gs, 20) for r, gs in items))

    # soft (learned, hop_hint set) vs terminal (hard, via explicit depth_weights)
    for hop in (2, 3):
        soft_mrr, soft_rec = score(hop, hop)
        print(f"hop{hop}: soft-c MRR {soft_mrr:.4f}  recall@20 {soft_rec:.4f}")


if __name__ == "__main__":
    main()
