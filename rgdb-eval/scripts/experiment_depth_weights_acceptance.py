"""Acceptance: depth-weighted terminal(k) readout meets the spec's success criteria.

For each hop k in {1,2,3}, score MetaQA test questions with the trained transition
matrix, seeded with the first-hop relation, using depth_weights = terminal(k) via the
native binding. Also scores the SAME trained matrix with uniform weights
(depth_weights=None) to isolate the lift that depth-weighting alone provides.

Evaluates the FULL test set per hop (not a fixed slice). 3-hop MRR at n=1000 has a
bootstrap sd of ~0.011, so a first-1000-question estimate swings ~+/-0.02 by luck of
the slice; the full-set estimate is stable to ~+/-0.006. See
scripts/investigate_3hop_variance.py for the analysis that set the floors below.

Floors are the converged full-set values minus a margin, NOT the (noisier, and by
chance higher) 300/1000-question exploratory numbers:
    hop  full-set MRR   floor   margin
     1      0.987        0.98    ~0.007  (near ceiling, tiny variance)
     2      0.612        0.58    ~0.032
     3      0.381        0.36    ~0.021
For context, untyped-PPR's 3-hop MRR is 0.279 and the same trained matrix WITHOUT
depth weighting is 0.235 -- so terminal(3)'s 0.381 is a +37% / +62% win, respectively.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_depth_weights_acceptance.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, mrr, K_VALUES

DATA = "data/MetaQA"
LIMIT = None  # full test set per hop -> stable, reproducible gate (no slice roulette)
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05

# Floors: converged full-set MRR minus margin (see module docstring / the investigation
# script). Deliberately below the true values so slice/version variance cannot flake the
# gate, while still far above every baseline (untyped-PPR 3-hop = 0.279).
THRESHOLDS = {1: 0.98, 2: 0.58, 3: 0.36}


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


def terminal(k: int) -> list[float]:
    v = [0.0] * (MAX_DEPTH + 1)
    v[k] = 1.0
    return v


def score(g, vocab, graph, qs, weights) -> tuple[float, float]:
    """(MRR, Hits@1) over qs with the given depth_weights (None = uniform)."""
    k_max = max(K_VALUES)
    items = []
    for q in qs:
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        totals = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel,
                                     MAX_DEPTH, MIN_INTENSITY, weights))
        totals.pop(q.topic_id, None)
        ranked = [nid for nid, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
        items.append((ranked, set(q.answer_ids)))
    return (mean(mrr(r, gs) for r, gs in items),
            mean(hits_at_k(r, gs, 1) for r, gs in items))


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n} relations")

    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = core.vocab_from_matrix(list(graph.relations), build_trained(graph).ravel().tolist())

    rows = []
    failures = []
    for hop in (1, 2, 3):
        qs = load_questions(
            os.path.join(DATA, f"qa_test_{hop}hop.txt"), hop, graph, limit=LIMIT,
            qtype_path=os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt"),
        )
        m_term, h1_term = score(g, vocab, graph, qs, terminal(hop))      # depth-weighted
        m_uni, h1_uni = score(g, vocab, graph, qs, None)                 # same matrix, uniform
        rows.append((hop, len(qs), h1_term, m_term, m_uni))
        status = "PASS" if m_term >= THRESHOLDS[hop] else "FAIL"
        if m_term < THRESHOLDS[hop]:
            failures.append((hop, m_term, THRESHOLDS[hop]))
        print(f"  hop{hop} (n={len(qs)}): terminal({hop}) MRR {m_term:.4f} "
              f"(>= {THRESHOLDS[hop]}) Hits@1 {h1_term:.4f} | uniform MRR {m_uni:.4f}  [{status}]")

    md = ["# Depth-weighted terminal(k) acceptance (MetaQA, full test set)\n",
          "\nTrained transition matrix, seeded with the first-hop relation. `terminal(k)`",
          "is the depth-weighted readout; `uniform` is the SAME matrix with no depth",
          "weighting (depth_weights=None) -- the isolated lift. Full test set per hop, so",
          "these are stable estimates, not slice-dependent.\n",
          "\nBaseline reference (from results/metaqa-depth-control.md): untyped-PPR 3-hop",
          "MRR 0.279; the trained matrix without depth control 0.235. terminal(3)'s 0.381",
          "beats both by +37% / +62%.\n",
          "\n| hop | n | terminal(k) Hits@1 | terminal(k) MRR | uniform MRR | floor |",
          "|---|---|---|---|---|---|"]
    for hop, n_q, h1, m_term, m_uni in rows:
        md.append(f"| {hop} | {n_q} | {h1:.4f} | {m_term:.4f} | {m_uni:.4f} | {THRESHOLDS[hop]} |")
    out = "results/metaqa-depth-weights.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwrote {out}")

    assert not failures, f"depth-weight acceptance failed: {failures}"
    print("\nACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
