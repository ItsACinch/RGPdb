"""Why learned soft depth weights (#2) cannot beat terminal(k) — the measurement.

Feature #2 hoped to learn a soft per-depth weight vector from feedback that improves on
the hard terminal(k). This measures two derivations against terminal(k) on MetaQA 3-hop:
  - discriminative: c[d] = answer_mass[d] / (background_mass[d] + kappa)   (the spec's)
  - generative:     c[d] proportional to answer_mass[d], blended with the terminal prior

RESULT (2026-07-11): BOTH lose to terminal(3). The answer's arrival-depth MASS profile is
~76% at depth 1 (the 3-hop answer is usually also reachable via a 1-hop shortcut, e.g. a
movie's own language), but the DISTRACTORS live at depth 1 too. Any c derived from where
the answer's mass concentrates up-weights depth 1 and drowns the answer in distractors.
terminal(k) wins precisely by ignoring answer-mass concentration and isolating the one
depth where answer and distractors are SEPARABLE.

Conclusion: no single per-depth weight vector learned from answer-mass statistics can beat
terminal(k) — separating a same-depth answer from same-depth distractors needs a per-NODE
model. That is feature #4 (the reranker), with arrival depth as one feature among several.
#2's learning is therefore a dead end; the hop_hint API is kept only for its cold-start
terminal(k) ergonomics, with depth-profile learning disabled by default.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/investigate_soft_depth_derivation.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core
from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import recall_at_k, mrr

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
KAPPA, EPS = 10.0, 1e-3
TRAIN = 3000


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
    rid = graph.relation_to_id
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = trained_vocab(graph)

    ans = np.zeros(MD + 1)
    bg = np.zeros(MD + 1)
    for q in load_questions(os.path.join(DATA, "qa_train_3hop.txt"), 3, graph,
                            limit=TRAIN, qtype_path=os.path.join(DATA, "qa_train_3hop_qtype.txt")):
        rel = rid.get(q.relation) if q.relation else None
        per, _ = core.propagate_layered(g, vocab, [(q.topic_id, 1.0)], rel, MD, MI)
        pd = {nn: pp for nn, pp in per}
        if q.answer_ids[0] in pd:
            ans += np.array(pd[q.answer_ids[0]])
        for pp in pd.values():
            bg += np.array(pp)

    term = np.array([0., 0., 0., 1., 0.])
    disc = ans / (bg + KAPPA + EPS)
    disc = disc / disc.max() if disc.max() > 0 else term.copy()
    gen = ans + KAPPA * term
    gen = gen / gen.max()
    print("answer-depth profile   :", np.round(ans / ans.sum(), 3))
    print("background-depth profile:", np.round(bg / bg.sum(), 3))
    print("terminal(3)     :", np.round(term, 3))
    print("discriminative  :", np.round(disc, 3), " <- down-weights depth 3 (background grows with depth)")
    print("generative      :", np.round(gen, 3))

    def score(c):
        qs = load_questions(os.path.join(DATA, "qa_test_3hop.txt"), 3, graph,
                            limit=None, qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"))
        items = []
        for q in qs:
            rel = rid.get(q.relation) if q.relation else None
            per, _ = core.propagate_layered(g, vocab, [(q.topic_id, 1.0)], rel, MD, MI)
            sc = {nn: float(np.dot(pp, c)) for nn, pp in per if nn != q.topic_id}
            ranked = [nn for nn, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:20]
            items.append((ranked, set(q.answer_ids)))
        return mean(mrr(r, gs) for r, gs in items), mean(recall_at_k(r, gs, 20) for r, gs in items)

    print()
    rows = []
    for name, c in (("terminal(3)", term), ("discriminative", disc), ("generative", gen)):
        m, r = score(c)
        rows.append((name, m, r))
        print(f"3-hop {name:15s}: MRR {m:.4f}  recall@20 {r:.4f}")

    out = "results/metaqa-soft-depth-weights.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Soft depth weights (#2): a measured dead end (MetaQA 3-hop)\n\n"
                "Learned per-depth weight vectors cannot beat hard `terminal(k)`, because the "
                "3-hop answer's arrival mass concentrates at depth 1 (~76%) alongside the "
                "distractors. Separating same-depth answer from distractor needs a per-node "
                "model (feature #4, the reranker).\n\n"
                "| 3-hop derivation | MRR | recall@20 |\n|---|---|---|\n")
        for name, m, r in rows:
            f.write(f"| {name} | {m:.4f} | {r:.4f} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
