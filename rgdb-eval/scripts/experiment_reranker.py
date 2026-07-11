"""#4 disposition: measure the cold/terminal(3) 3-hop baseline the reranker is gated
against, and document why this script cannot exercise the reranker from Python.

Background: the online reranker (rgdb/src/reranker.rs) was found to regress 3-hop
Hits@1 while improving recall@20 -- global feature preferences cannot supply the
question-specific target relation that separates a correct same-depth answer from a
distractor (see docs/superpowers/specs/2026-07-11-feedback-learned-ranking-design.md,
Component #4 "MEASURED OUTCOME" note, and the raw acceptance-gate run in
.superpowers/sdd/task-8-report.md: Hits@1 0.1585 vs terminal(3) baseline 0.203,
recall@20 0.5339 vs terminal(3) baseline 0.902 -- that 0.902 figure was later found to
be a mislabeled Hits@20, not recall@20; terminal(3) recall@20 is ~0.531).

Disposition (rgdb/src/engine.rs): the reranker is now gated behind
`EngineConfig.reranker_enabled`, default **false** -- neither applied in `query()` nor
trained in `record_feedback()` unless explicitly turned on, mirroring the existing
`depth_profile_learning` gate. This is a Rust-only `EngineConfig` field: the Python
`core.Engine` constructor (rgdb-python/src/lib.rs) does not expose a `reranker_enabled`
kwarg, so a Python caller can only ever get the reranker-off engine -- which is exactly
the point of the gate (default behavior cannot regress). Adding that binding is out of
scope for this script.

So this script measures ONE engine: the trained-matrix, reranker-off (default
EngineConfig) engine, replaying 3-hop training feedback (which now also leaves the
reranker and depth-profile learner untouched, since both gates default off) and
reporting 3-hop test Hits@1 / recall@20. This is the cold/terminal(3) baseline that the
gated-off reranker cannot regress below. It does NOT reproduce the reranker-ENABLED
numbers -- those require the Rust flag and are documented in the spec instead.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_reranker.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core
from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN = 4000


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
    # core.Engine has no reranker_enabled kwarg -- it always builds with
    # EngineConfig::default(), which (as of this change) has reranker_enabled = false.
    # This IS the "reranker off" engine; there is no Python-reachable way to get the
    # reranker-on one.
    eng = core.Engine(g, vocab, rebuild_every_n=0)

    tp = os.path.join(DATA, "qa_train_3hop.txt")
    tt = os.path.join(DATA, "qa_train_3hop_qtype.txt")
    events = 0
    for q in load_questions(tp, 3, graph, limit=TRAIN, qtype_path=tt):
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        _, qid = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=3)
        try:
            eng.record_feedback(qid, q.answer_ids[0], 1.0)
            events += 1
        except ValueError:
            pass
    eng.refresh_profiles()
    print(f"replayed {events} 3-hop feedback events "
          f"(inert: reranker_enabled=false and depth_profile_learning=false by default)")

    qs = load_questions(os.path.join(DATA, "qa_test_3hop.txt"), 3, graph,
                        limit=None, qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"))
    items = []
    for q in qs:
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        ranked, _ = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=3)
        ranked = [n for n, _ in ranked if n != q.topic_id][:20]
        items.append((ranked, set(q.answer_ids)))
    h1 = mean(hits_at_k(r, gs, 1) for r, gs in items)
    r20 = mean(recall_at_k(r, gs, 20) for r, gs in items)
    print(f"3-hop cold/terminal(3), reranker off (default): Hits@1 {h1:.4f}, recall@20 {r20:.4f}")

    out = "results/metaqa-reranker.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "# Reranker gate (MetaQA 3-hop)\n\n"
            "`EngineConfig.reranker_enabled` defaults to **false** (rgdb/src/engine.rs): the\n"
            "reranker is neither applied in `query()` nor trained in `record_feedback()` unless\n"
            "explicitly turned on. This mirrors `depth_profile_learning`.\n\n"
            "| variant | Hits@1 | recall@20 | source |\n"
            "|---|---|---|---|\n"
            f"| cold / terminal(3), reranker off (this script, default `core.Engine`) "
            f"| {h1:.4f} | {r20:.4f} | measured here |\n"
            "| + online reranker enabled (Rust-only; not reachable from `core.Engine`) "
            "| 0.176 | (+10% recall@20) | design spec Component #4 MEASURED OUTCOME |\n\n"
            "The reranker-enabled row is **not reproduced by this script** -- the Python\n"
            "`core.Engine` binding (rgdb-python/src/lib.rs) does not expose a `reranker_enabled`\n"
            "kwarg, so every Python-constructed engine gets the gated-off (default) `EngineConfig`.\n"
            "Enabling the reranker requires building an `RgdbEngine` from Rust with\n"
            "`EngineConfig { reranker_enabled: true, .. }`. The regression when it IS enabled\n"
            "(3-hop Hits@1 0.208 -> 0.176, while recall@20 improves ~10%) is documented in\n"
            "`docs/superpowers/specs/2026-07-11-feedback-learned-ranking-design.md` (Component #4\n"
            "MEASURED OUTCOME) and in the raw acceptance-gate run captured in\n"
            "`.superpowers/sdd/task-8-report.md`. Root cause: the reranker learns a single global\n"
            "feature-preference vector shared across all queries, so it cannot supply the\n"
            "question-specific target relation that separates a correct same-depth answer from a\n"
            "distractor -- the same wall the learned soft depth weights (#2) hit. Disposition: keep\n"
            "the reranker as an inert seam for a future query-conditioned extension (#3); do not\n"
            "enable it by default.\n"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
