"""Core graph structures mirroring the Rust RGDB implementation.

Uses CSR (Compressed Sparse Row) format for memory-efficient graph storage,
matching the Rust Graph struct in src/graph.rs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

N_ANGLE_BINS = 16


@dataclass
class NodeProps:
    """Physical properties of a node (mirrors Rust NodeProps)."""
    luminance: float = 1.0
    reflection: float = 1.0
    refraction_index: float = 1.0
    default_angle_bin: int = 0
    directional_luminance: np.ndarray = field(
        default_factory=lambda: np.ones(N_ANGLE_BINS, dtype=np.float32)
    )

    @classmethod
    def uniform(cls, luminance: float = 1.0) -> NodeProps:
        return cls(
            luminance=luminance,
            directional_luminance=np.full(N_ANGLE_BINS, luminance, dtype=np.float32),
        )

    @classmethod
    def directional(cls, primary_bin: int, luminance: float = 1.0, spread: float = 0.1) -> NodeProps:
        """Create node that emits primarily in one direction with falloff."""
        dl = np.full(N_ANGLE_BINS, spread, dtype=np.float32)
        dl[primary_bin % N_ANGLE_BINS] = luminance
        return cls(
            luminance=luminance,
            default_angle_bin=primary_bin % N_ANGLE_BINS,
            directional_luminance=dl,
        )


@dataclass
class EdgeProps:
    """Physical properties of an edge (mirrors Rust EdgeProps)."""
    attenuation: float = 0.0
    angle_bin: int = 0
    is_portal: bool = False


class Graph:
    """CSR-format graph with node and edge properties.

    Mirrors the Rust Graph struct. Nodes are integers 0..n-1.
    """

    def __init__(self, num_nodes: int):
        self._num_nodes = num_nodes
        self._node_props: list[NodeProps] = [NodeProps.uniform() for _ in range(num_nodes)]
        # Build as adjacency list first, compact to CSR on freeze
        self._adj: list[list[tuple[int, EdgeProps]]] = [[] for _ in range(num_nodes)]
        self._frozen = False
        # CSR arrays (populated on freeze)
        self._row_ptr: Optional[np.ndarray] = None
        self._col_idx: Optional[np.ndarray] = None
        self._edge_props_list: Optional[list[EdgeProps]] = None
        # Optional labels
        self._node_labels: dict[int, str] = {}
        self._label_to_id: dict[str, int] = {}

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        if self._frozen:
            return len(self._col_idx)
        return sum(len(adj) for adj in self._adj)

    def set_node_props(self, node_id: int, props: NodeProps) -> None:
        self._node_props[node_id] = props

    def set_node_label(self, node_id: int, label: str) -> None:
        self._node_labels[node_id] = label
        self._label_to_id[label] = node_id

    def get_node_id(self, label: str) -> Optional[int]:
        return self._label_to_id.get(label)

    def get_node_label(self, node_id: int) -> Optional[str]:
        return self._node_labels.get(node_id)

    def add_edge(self, src: int, dst: int, props: Optional[EdgeProps] = None) -> None:
        if self._frozen:
            raise RuntimeError("Graph is frozen (CSR built). Cannot add edges.")
        if props is None:
            props = EdgeProps()
        self._adj[src].append((dst, props))

    def freeze(self) -> None:
        """Compact adjacency list into CSR format for fast propagation."""
        row_ptr = np.zeros(self._num_nodes + 1, dtype=np.int64)
        col_idx_list = []
        edge_props_list = []

        for u in range(self._num_nodes):
            neighbors = self._adj[u]
            row_ptr[u + 1] = row_ptr[u] + len(neighbors)
            for v, ep in neighbors:
                col_idx_list.append(v)
                edge_props_list.append(ep)

        self._row_ptr = row_ptr
        self._col_idx = np.array(col_idx_list, dtype=np.int32) if col_idx_list else np.array([], dtype=np.int32)
        self._edge_props_list = edge_props_list
        self._frozen = True

    def neighbors(self, node_id: int):
        """Yield (neighbor_id, EdgeProps) for a node."""
        if self._frozen:
            start = self._row_ptr[node_id]
            end = self._row_ptr[node_id + 1]
            for i in range(start, end):
                yield int(self._col_idx[i]), self._edge_props_list[i]
        else:
            yield from self._adj[node_id]

    def node_props(self, node_id: int) -> NodeProps:
        return self._node_props[node_id]
