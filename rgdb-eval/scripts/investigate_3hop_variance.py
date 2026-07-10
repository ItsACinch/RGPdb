"""Is 3-hop terminal(3) MRR 0.398 (n=300) vs 0.370 (n=1000) sampling noise or a
systematic first-slice-is-easier effect? Score ALL 3-hop test questions once, then
look at cumulative MRR, contiguous slices, and bootstrap variance.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/investigate_3hop_variance.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import mrr

DATA = "data/MetaQA"
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05
HOP = 3


def build_trained(graph) -> np.ndarray:
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
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


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = core.vocab_from_matrix(list(graph.relations), build_trained(graph).ravel().tolist())

    # ALL 3-hop test questions (limit None => everything the loader accepts).
    qs = load_questions(os.path.join(DATA, f"qa_test_{HOP}hop.txt"), HOP, graph,
                        limit=None, qtype_path=os.path.join(DATA, f"qa_test_{HOP}hop_qtype.txt"))
    w = [0.0] * (MAX_DEPTH + 1)
    w[HOP] = 1.0
    k_max = 20

    rr = np.empty(len(qs), dtype=np.float64)  # per-question reciprocal rank, in FILE ORDER
    for i, q in enumerate(qs):
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        totals = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel, MAX_DEPTH, MIN_INTENSITY, w))
        totals.pop(q.topic_id, None)
        ranked = [nid for nid, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
        rr[i] = mrr(ranked, set(q.answer_ids))
    N = len(rr)
    print(f"scored ALL {N} 3-hop test questions with terminal(3)\n")

    # 1) Cumulative MRR at increasing n (file order) — does it converge, and where do
    #    the exploratory 300 and acceptance 1000 land?
    print("cumulative MRR over the FIRST n questions (file order):")
    for cut in (300, 500, 1000, 2000, 5000, N):
        c = min(cut, N)
        print(f"  first {c:5d}: {rr[:c].mean():.4f}")

    # 2) Contiguous non-overlapping slices — is the head systematically higher?
    print("\ncontiguous 1000-question slices (is the first slice an outlier?):")
    for start in range(0, N, 1000):
        end = min(start + 1000, N)
        if end - start < 200:
            continue
        print(f"  [{start:5d}:{end:5d}] (n={end-start:4d}): {rr[start:end].mean():.4f}")

    # 3) Bootstrap: sampling distribution of MRR at n=300 and n=1000, so I can set a
    #    defensible floor with a real margin instead of eyeballing one.
    rng = np.random.default_rng(20260710)
    for size in (300, 1000):
        boot = np.array([rng.choice(rr, size=size, replace=True).mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\nbootstrap MRR at n={size} (2000 resamples): "
              f"mean {boot.mean():.4f}, sd {boot.std():.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    print(f"\nfull-set MRR (the most reliable point estimate): {rr.mean():.4f}")
    print(f"exploratory first-300: {rr[:300].mean():.4f}   acceptance first-1000: {rr[:1000].mean():.4f}")


if __name__ == "__main__":
    main()
