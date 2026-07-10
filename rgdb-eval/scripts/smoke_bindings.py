"""Confirm the native rgdb bindings import and a trivial propagation runs."""
from rgdb_embeddings import _rgdb_core as core


def main() -> None:
    adjacency = [[(1, 0.0, 0)], [(2, 0.0, 0)], []]  # 0 -> 1 -> 2, relation 0
    g = core.build_graph(3, adjacency)
    assert g.num_nodes == 3 and g.num_edges == 2
    vocab = core.uniform_vocab(1)
    totals = dict(core.propagate(g, vocab, [(0, 1.0)], None, 4, 1e-6))
    assert abs(totals[0] - 1.0) < 1e-6, totals
    assert abs(totals[1] - 0.85) < 1e-4, totals  # reflection default 0.85
    print("bindings smoke ok:", sorted(totals.items()))


if __name__ == "__main__":
    main()
