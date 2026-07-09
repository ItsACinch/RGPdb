from rgdb_eval import TypedGraph
from rgdb_eval.rankers.ppr import PPRRanker


def line_graph() -> TypedGraph:
    # 0 -> 1 -> 2 -> 3, single relation
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_ppr_ranks_closer_nodes_higher():
    r = PPRRanker(line_graph())
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    # From seed 0, node 1 must outrank node 3 (closer on the chain).
    assert ranked.index(1) < ranked.index(3)


def test_ppr_returns_at_most_k():
    r = PPRRanker(line_graph())
    assert len(r.rank(seeds=[0], query_relation=None, k=2)) == 2
