"""schedule kwarg on the native bindings (query-conditioned schedule seam)."""
import math
from rgdb_embeddings import _rgdb_core as core


def _ab_chain():
    # 0 -(A=0)-> 1 -(B=1)-> 2
    adj = [[(1, 0.0, 0)], [(2, 0.0, 1)], []]
    return core.build_graph(3, adj), core.uniform_vocab(2)


def test_schedule_none_matches_no_arg():
    g, v = _ab_chain()
    a = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, None))
    b = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None))  # schedule omitted
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k]


def test_schedule_matches_relation_per_hop():
    g, v = _ab_chain()
    t = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, [0, 1]))
    assert math.isclose(t[2], 0.7225, rel_tol=1e-4)
    t2 = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, [0, 0]))
    assert math.isclose(t2[2], 0.036125, rel_tol=1e-4)  # 2nd hop floored


def test_engine_query_honors_schedule():
    g, v = _ab_chain()
    eng = core.Engine(g, v, rebuild_every_n=0)
    # With schedule [A,A] the B-hop is floored, so node 2's score is far below the
    # matching-schedule case; a bare query (no schedule) uses the vocab.
    ranked_bad, _ = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 0])
    ranked_ok, _ = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 1])
    s_bad = dict(ranked_bad)[2]
    s_ok = dict(ranked_ok)[2]
    assert s_ok > s_bad * 5, f"schedule must reach the engine: ok={s_ok} bad={s_bad}"
