"""depth_weights kwarg on the native propagate() / Engine.query() bindings."""
import math
import pytest
from rgdb_embeddings import _rgdb_core as core


def _chain():
    # 0 -> 1 -> 2 -> 3, single relation, no attenuation.
    adj = [[(1, 0.0, 0)], [(2, 0.0, 0)], [(3, 0.0, 0)], []]
    g = core.build_graph(4, adj)
    v = core.uniform_vocab(1)
    return g, v


def test_uniform_weights_match_none():
    g, v = _chain()
    a = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, None))
    b = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, 1.0, 1.0, 1.0]))
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k]  # exact


def test_terminal_scores_only_arrival_depth():
    g, v = _chain()
    t = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [0.0, 0.0, 0.0, 1.0]))
    assert math.isclose(t.get(3, 0.0), 0.614125, rel_tol=1e-4)
    assert t.get(1, 0.0) == 0.0


def test_wrong_length_raises_value_error():
    g, v = _chain()
    with pytest.raises(ValueError):
        core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, 1.0])  # needs length 4


def test_negative_weight_raises_value_error():
    g, v = _chain()
    with pytest.raises(ValueError):
        core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, -1.0, 1.0, 1.0])
