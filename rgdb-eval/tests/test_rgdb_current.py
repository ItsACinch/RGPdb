import pytest
from rgdb_eval import TypedGraph

core = pytest.importorskip("rgdb_embeddings._rgdb_core")
from rgdb_eval.rankers.rgdb_current import CurrentRgdbRanker


def line_graph():
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_current_rgdb_ranks_reachable_nodes():
    r = CurrentRgdbRanker(line_graph())
    ranked = r.rank(seeds=[0], query_relation="r", k=4)
    # Node 1 (1 hop) should outrank node 3 (3 hops) by intensity.
    assert ranked.index(1) < ranked.index(3)
