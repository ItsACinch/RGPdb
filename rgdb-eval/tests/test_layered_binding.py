"""propagate_layered native binding: per-depth intensities + dominant incoming relation."""
import math
from rgdb_embeddings import _rgdb_core as core


def _chain():
    adj = [[(1, 0.0, 0)], [(2, 0.0, 0)], [(3, 0.0, 0)], []]
    return core.build_graph(4, adj), core.uniform_vocab(1)


def test_layered_matches_scalar_under_uniform():
    g, v = _chain()
    per_depth, _dom = core.propagate_layered(g, v, [(0, 1.0)], 0, 3, 0.0)
    layered = {n: prof for n, prof in per_depth}
    scalar = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 0.0, None))
    for n, prof in layered.items():
        assert math.isclose(sum(prof), scalar.get(n, 0.0), rel_tol=1e-5, abs_tol=1e-6)


def test_layered_depth_profile_of_chain():
    g, v = _chain()
    per_depth, _dom = core.propagate_layered(g, v, [(0, 1.0)], 0, 3, 1e-6)
    layered = {n: prof for n, prof in per_depth}
    assert math.isclose(layered[3][3], 0.614125, rel_tol=1e-4)
    assert layered[3][1] == 0.0
