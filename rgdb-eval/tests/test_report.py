from rgdb_eval import TypedGraph, Question
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.report import evaluate, to_markdown


def _line_graph():
    return TypedGraph(
        num_nodes=4, entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"], edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_evaluate_and_markdown():
    g = _line_graph()
    qs = [Question(text="q", topic_id=0, answer_ids=[1], relation="r", hop=1)]
    res = evaluate(PPRRanker(g), qs)
    assert res["ranker"] == "untyped-ppr"
    assert 0.0 <= res["hop1"]["mrr"] <= 1.0
    md = to_markdown([res])
    assert "untyped-ppr" in md and "mrr" in md.lower()


def test_evaluate_excludes_seed():
    # PPR ranks the seed (node 0) highest; only seed exclusion makes the real
    # answer (node 1) the top-1, so hits@1 must be 1.0.
    g = _line_graph()
    qs = [Question(text="q", topic_id=0, answer_ids=[1], relation="r", hop=1)]
    res = evaluate(PPRRanker(g), qs)
    assert res["hop1"]["hits@1"] == 1.0
