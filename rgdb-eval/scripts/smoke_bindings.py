"""Confirm the native rgdb bindings import and a trivial propagation runs."""
from rgdb_embeddings import _rgdb_core as core


def main() -> None:
    # 0 -> 1 -> 2 chain; attenuation 0.1, angle_bin 0
    adjacency = [[(1, 0.1, 0)], [(2, 0.1, 0)], []]
    g = core.build_graph(3, adjacency)
    assert g.num_nodes == 3, g.num_nodes
    assert g.num_edges == 2, g.num_edges
    intensities = core.propagate_light(g, 0)
    assert len(intensities) == 3, len(intensities)
    assert intensities[0] > 0.0, "source should have positive intensity"
    print("bindings smoke ok:", list(intensities))


if __name__ == "__main__":
    main()
