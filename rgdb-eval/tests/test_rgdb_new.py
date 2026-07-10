import pytest
from rgdb_eval import TypedGraph

core = pytest.importorskip("rgdb_embeddings._rgdb_core")
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker


def line_graph():
    return TypedGraph(
        num_nodes=4, entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"], edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_new_rgdb_excludes_seed_and_orders_by_distance():
    r = NewRgdbRanker(line_graph(), vocab_mode="uniform")
    ranked = r.rank(seeds=[0], query_relation="r", k=4)
    assert 0 not in ranked            # seed excluded
    assert ranked.index(1) < ranked.index(3)
