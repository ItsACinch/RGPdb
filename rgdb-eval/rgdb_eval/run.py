"""CLI: evaluate all available contenders on MetaQA and write a markdown table."""
from __future__ import annotations
import argparse
import os

from .metaqa import load_kb, load_questions
from .embeddings import embed_texts
from .rankers.ppr import PPRRanker
from .rankers.vector import VectorRanker, VectorTwoHopRanker
from .report import evaluate, to_markdown


def build_rankers(graph, node_vecs, question_vec):
    rankers = [
        VectorRanker(graph, node_vecs, lambda _r: question_vec),
        VectorTwoHopRanker(graph, node_vecs, lambda _r: question_vec),
        PPRRanker(graph),
    ]
    try:
        from .rankers.rgdb_current import CurrentRgdbRanker
        rankers.append(CurrentRgdbRanker(graph))
    except Exception as exc:  # bindings not installed -> skip, but say so
        print(f"[warn] rgdb-current contender skipped: {exc}")
    return rankers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/MetaQA")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="results/metaqa.md")
    args = ap.parse_args()

    graph = load_kb(os.path.join(args.data, "kb.txt"))
    node_vecs = embed_texts(graph.entity_names)

    all_questions = []
    for hop in (1, 2, 3):
        path = os.path.join(args.data, f"qa_test_{hop}hop.txt")
        if os.path.exists(path):
            all_questions += load_questions(path, hop, graph, limit=args.limit)

    # One question embedding per question would be ideal; for a per-run table we
    # score each contender question-by-question, so embed lazily per question.
    # Here we pass a closure that re-embeds the current question text.
    rows = []
    for ranker in build_rankers(graph, node_vecs, question_vec=node_vecs[0]):
        # Rebind the query vector per question for vector contenders.
        if isinstance(ranker, VectorRanker):
            def qvec(_rel, _cache={}):
                return _cache.get("v", node_vecs[0])
            ranker.query_vec_fn = qvec
        rows.append(_evaluate_with_question_embeddings(ranker, all_questions))

    md = to_markdown(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# MetaQA retrieval results\n" + md)
    print(f"wrote {args.out}")


def _evaluate_with_question_embeddings(ranker, questions):
    # For vector contenders, set the per-question embedding before ranking.
    from .rankers.vector import VectorRanker as _VR
    if isinstance(ranker, _VR):
        texts = [q.text for q in questions]
        qvecs = embed_texts(texts)
        idx = {"i": 0}
        def fn(_rel):
            return qvecs[idx["i"]]
        ranker.query_vec_fn = fn
        # evaluate() ranks in question order, so advance the pointer in lockstep.
        orig_rank = ranker.rank
        def ranked(seeds, rel, k):
            r = orig_rank(seeds, rel, k)
            idx["i"] = min(idx["i"] + 1, len(qvecs) - 1)
            return r
        ranker.rank = ranked
    return evaluate(ranker, questions)
