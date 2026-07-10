"""Acceptance: does the online feedback loop learn what offline training learned?

Replays MetaQA *training* questions through the engine as feedback events
(query -> record_feedback(gold answer)), then evaluates the resulting matrix on
the *test* questions. Compares against the uniform cold start and the offline
trained matrix.

Run: .venv/Scripts/python.exe rgdb-eval/scripts/experiment_online_learning.py
"""
from __future__ import annotations
import os
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker
from rgdb_eval.report import evaluate, to_markdown

DATA = "data/MetaQA"
TRAIN_EVENTS = 4000   # feedback events to replay
TEST_LIMIT = 1000     # test questions per hop


def replay_feedback(graph) -> list[float]:
    """Drive the engine with training questions; return the learned flat matrix."""
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)

    n = len(graph.relations)
    uniform = core.vocab_from_matrix(list(graph.relations), [1.0] * (n * n))
    eng = core.Engine(g, uniform, rebuild_every_n=0)  # manual refresh at the end

    # cold start must be exactly uniform
    assert all(abs(x - 1.0) < 1e-6 for x in eng.matrix()), "cold start not uniform"

    events = 0
    skipped = 0
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_train_{hop}hop.txt")
        qtype = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(qpath):
            continue
        per_hop = TRAIN_EVENTS // 3
        for q in load_questions(qpath, hop, graph, limit=per_hop, qtype_path=qtype):
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            _, qid = eng.query([(q.topic_id, 1.0)], rel, 4, 1e-4)
            try:
                eng.record_feedback(qid, q.answer_ids[0], 1.0)
                events += 1
            except ValueError:
                skipped += 1  # unreachable within max_depth
    eng.refresh()
    print(f"replayed {events} feedback events ({skipped} skipped as unreachable)")
    return eng.matrix()


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {len(graph.relations)} relations")

    learned = replay_feedback(graph)
    n = len(graph.relations)
    learned_np = np.asarray(learned, dtype=np.float32).reshape(n, n)
    off_diag = learned_np[~np.eye(n, dtype=bool)]
    print(f"learned matrix: off-diagonal mean {off_diag.mean():.3f}, min {off_diag.min():.3f}")

    questions = []
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_test_{hop}hop.txt")
        qtype = os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt")
        if os.path.exists(qpath):
            questions += load_questions(qpath, hop, graph, limit=TEST_LIMIT, qtype_path=qtype)

    contenders = [
        PPRRanker(graph),
        NewRgdbRanker(graph, vocab_mode="uniform"),
        NewRgdbRanker(graph, sim_matrix=learned_np, name="rgdb-new-online-learned"),
    ]
    rows = []
    for c in contenders:
        rows.append(evaluate(c, questions))
        print(f"  scored {c.name}")

    md = "# Online-learning acceptance (MetaQA)\n" + to_markdown(rows)
    out = "results/metaqa-online-learning.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}\n")
    print(md)

    # Directional acceptance: the loop must beat the uniform cold start at 2-hop.
    by_name = {r["ranker"]: r for r in rows}
    uni2 = by_name["rgdb-new-uniform"]["hop2"]["mrr"]
    on2 = by_name["rgdb-new-online-learned"]["hop2"]["mrr"]
    on1 = by_name["rgdb-new-online-learned"]["hop1"]["mrr"]
    print(f"2-hop MRR: uniform {uni2:.3f} -> online-learned {on2:.3f} (offline reference 0.275)")
    print(f"1-hop MRR: online-learned {on1:.3f} (offline reference 0.992)")
    assert on2 > uni2, f"online learning did not improve 2-hop MRR ({on2:.3f} <= {uni2:.3f})"
    assert on1 >= 0.90, f"online learning damaged 1-hop MRR ({on1:.3f})"


if __name__ == "__main__":
    main()
