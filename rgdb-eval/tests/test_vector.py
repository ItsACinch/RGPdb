import numpy as np
from rgdb_eval import TypedGraph
from rgdb_eval.rankers.vector import VectorRanker, VectorTwoHopRanker


def graph4():
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0)],  # 3 is unreachable from 0
    )


def fixed_vecs():
    # node 2 is identical to the query; node 3 also close but unreachable
    return np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.9, 0.1]], dtype=np.float64
    )


def test_vector_ranks_by_cosine():
    q = np.array([1.0, 0.0])
    r = VectorRanker(graph4(), fixed_vecs(), lambda _rel: q)
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    assert ranked[0] in (0, 2)  # cosine-identical to query


def test_two_hop_excludes_unreachable():
    q = np.array([0.9, 0.1])
    r = VectorTwoHopRanker(graph4(), fixed_vecs(), lambda _rel: q)
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    # node 3 is closest to q but not within 2 hops of seed 0 -> excluded
    assert 3 not in ranked
