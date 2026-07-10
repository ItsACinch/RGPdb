"""Experiment: does a data-weighted relation-transition matrix fix multi-hop refraction?

Builds a DIRECTED transition matrix T[r_prev][r_next] from the gold relation
sequences in MetaQA's qa_train qtypes (i.e. "learned" from the training set),
and compares a `refraction-trained` contender against PPR, typed-uniform, and
the name-embedded refraction. Vector contenders are omitted (they need the slow
43k-entity embedding and are irrelevant to the transition-matrix question).

Run: .venv/Scripts/python.exe rgdb-eval/scripts/experiment_trained_refraction.py
"""
from __future__ import annotations
import os
import numpy as np

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker
from rgdb_eval.report import evaluate, to_markdown

DATA = "data/MetaQA"
LIMIT = 1000
FLOOR = 0.05  # unseen transitions: heavy penalty, not a full block


def build_transition_matrix(graph) -> np.ndarray:
    """T[a][b] = normalized frequency that gold relation b follows a in training."""
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n), dtype=np.float64)
    total_pairs = 0
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(qpath):
            continue
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                seq = qtype_to_relation_sequence(line)
                ids = [rid[r] for r in seq if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
                    total_pairs += 1
    # Row-normalize off-diagonal to max=1 (the typical next relation after `a`
    # gets full weight), floor unseen transitions, force diagonal=1 (staying on
    # a relation is never penalized — needed for the 1-hop win).
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        row = counts[a].copy()
        m = row.max()
        if m > 0:
            norm = row / m
            M[a] = np.maximum(M[a], norm.astype(np.float32))
    np.fill_diagonal(M, 1.0)
    print(f"transition matrix built from {total_pairs} gold relation pairs; "
          f"off-diagonal mean {M[~np.eye(n, dtype=bool)].mean():.3f}")
    return M


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, "
          f"{len(graph.relations)} relations")

    T = build_transition_matrix(graph)

    contenders = [
        PPRRanker(graph),
        NewRgdbRanker(graph, vocab_mode="uniform"),
        NewRgdbRanker(graph, vocab_mode="refraction"),          # name-embedded
        NewRgdbRanker(graph, sim_matrix=T, name="rgdb-new-trained"),
    ]

    rows_by_ranker = {c.name: c for c in contenders}
    all_rows = []
    for name, ranker in rows_by_ranker.items():
        # evaluate across all hops at once (report buckets by hop)
        questions = []
        for hop in (1, 2, 3):
            qpath = os.path.join(DATA, f"qa_test_{hop}hop.txt")
            qtype = os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt")
            if os.path.exists(qpath):
                questions += load_questions(qpath, hop, graph, limit=LIMIT,
                                            qtype_path=qtype)
        all_rows.append(evaluate(ranker, questions))
        print(f"  scored {name}")

    md = "# Trained-refraction experiment (MetaQA, 1000 q/hop)\n" + to_markdown(all_rows)
    out = "results/metaqa-trained-refraction.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}\n")
    print(md)


if __name__ == "__main__":
    main()
