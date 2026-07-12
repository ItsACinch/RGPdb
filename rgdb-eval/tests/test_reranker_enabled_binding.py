"""Engine(reranker_enabled=...) binding: default off (backward-compatible), can flip on."""
from rgdb_embeddings import _rgdb_core as core


def _chain():
    adj = [[(1, 0.0, 0)], [(2, 0.0, 1)], []]
    return core.build_graph(3, adj), core.uniform_vocab(2)


def test_default_construction_still_works():
    g, v = _chain()
    eng = core.Engine(g, v, rebuild_every_n=0)  # no reranker_enabled -> default False
    ranked, qid = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9)
    assert qid > 0 and len(ranked) >= 1


def test_reranker_enabled_kwarg_accepted():
    g, v = _chain()
    eng = core.Engine(g, v, rebuild_every_n=0, reranker_enabled=True)
    # cold reranker is identity, so this behaves like a normal query; just confirm it runs.
    ranked, qid = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 1])
    assert qid > 0 and len(ranked) >= 1
    assert eng.record_feedback(qid, 2, 1.0) is None  # trains the reranker, returns None on success
