"""GraphBuilder -- high-level API for constructing RGDB graphs from DataFrames.

Designed for Jupyter notebook workflows where you build a graph from query results
and then run propagation to discover relationships.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional
from .core import Graph, NodeProps, EdgeProps, N_ANGLE_BINS


# Angle bin assignments for automotive service domain relationships
RELATION_BINS = {
    # Temporal / sequential
    "followed_by": 4,       # Causes / leads_to
    "preceded_by": 8,       # PartOf (reverse temporal)

    # Ownership / assignment
    "owns_vehicle": 6,      # Contains
    "serviced_at": 6,       # Contains
    "assigned_to": 6,       # Contains
    "belongs_to": 8,        # PartOf

    # Similarity
    "same_vehicle": 10,     # SimilarTo
    "same_customer": 10,    # SimilarTo
    "same_code": 10,        # SimilarTo

    # Taxonomy
    "is_a": 0,              # IsA
    "type_of": 0,           # IsA
    "has_code": 0,          # IsA

    # Association
    "related_to": 2,        # RelatedTo (default)
    "co_occurs": 2,         # RelatedTo

    # Causal
    "resolves": 14,         # Enables
    "warns_about": 4,       # Causes
    "declined": 12,         # OppositeOf (refused service)

    # Preferences / intent (for recommendation graphs)
    "prefers_make": 10,     # SimilarTo (brand affinity)
    "prefers_type": 10,     # SimilarTo (vehicle type affinity)
    "purchased": 4,         # Causes (purchase action)
    "interested_in": 4,     # Causes (lead VOI)
}


def relation_to_bin(relation: str) -> int:
    """Map a relation string to an angle bin. Defaults to bin 2 (RelatedTo)."""
    return RELATION_BINS.get(relation, 2)


class GraphBuilder:
    """Build an RGDB graph from labeled nodes and edges.

    Handles node ID allocation, label tracking, and DataFrame-based bulk loading.

    Example:
        builder = GraphBuilder()
        builder.add_nodes_from_df(df_customers, id_col="contact_id", type_name="customer")
        builder.add_nodes_from_df(df_vins, id_col="vin", type_name="vehicle")
        builder.add_edges_from_df(df_services, src_col="contact_id", dst_col="vin",
                                  relation="serviced_at")
        graph = builder.build()
    """

    def __init__(self):
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        self._node_types: dict[int, str] = {}
        self._node_props_overrides: dict[int, NodeProps] = {}
        self._edges: list[tuple[int, int, EdgeProps]] = []
        self._next_id = 0

    @property
    def num_nodes(self) -> int:
        return self._next_id

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def _get_or_create_node(self, label: str, type_name: str = "default") -> int:
        """Get existing node ID or allocate a new one."""
        if label in self._label_to_id:
            return self._label_to_id[label]
        node_id = self._next_id
        self._next_id += 1
        self._label_to_id[label] = node_id
        self._id_to_label[node_id] = label
        self._node_types[node_id] = type_name
        return node_id

    def add_node(self, label: str, type_name: str = "default",
                 props: Optional[NodeProps] = None) -> int:
        """Add a single node. Returns node ID."""
        node_id = self._get_or_create_node(label, type_name)
        if props is not None:
            self._node_props_overrides[node_id] = props
        return node_id

    def add_nodes_from_df(self, df: pd.DataFrame, id_col: str,
                          type_name: str = "default",
                          luminance: float = 1.0) -> dict[str, int]:
        """Add nodes from a DataFrame column. Returns label->id mapping."""
        mapping = {}
        for val in df[id_col].dropna().unique():
            label = f"{type_name}:{val}"
            node_id = self._get_or_create_node(label, type_name)
            if luminance != 1.0:
                self._node_props_overrides[node_id] = NodeProps.uniform(luminance)
            mapping[str(val)] = node_id
        return mapping

    def add_edge(self, src_label: str, dst_label: str, relation: str = "related_to",
                 attenuation: float = 0.1, bidirectional: bool = False) -> None:
        """Add an edge between two labeled nodes."""
        src_id = self._label_to_id.get(src_label)
        dst_id = self._label_to_id.get(dst_label)
        if src_id is None or dst_id is None:
            return  # Skip edges to unknown nodes
        angle_bin = relation_to_bin(relation)
        ep = EdgeProps(attenuation=attenuation, angle_bin=angle_bin)
        self._edges.append((src_id, dst_id, ep))
        if bidirectional:
            self._edges.append((dst_id, src_id, ep))

    def add_edges_from_df(self, df: pd.DataFrame, src_col: str, dst_col: str,
                          src_type: str = "default", dst_type: str = "default",
                          relation: str = "related_to", attenuation: float = 0.1,
                          bidirectional: bool = False) -> int:
        """Add edges from DataFrame columns. Creates nodes if they don't exist.

        Returns number of edges added.
        """
        angle_bin = relation_to_bin(relation)
        count = 0
        for _, row in df[[src_col, dst_col]].dropna().iterrows():
            src_label = f"{src_type}:{row[src_col]}"
            dst_label = f"{dst_type}:{row[dst_col]}"
            src_id = self._get_or_create_node(src_label, src_type)
            dst_id = self._get_or_create_node(dst_label, dst_type)
            ep = EdgeProps(attenuation=attenuation, angle_bin=angle_bin)
            self._edges.append((src_id, dst_id, ep))
            if bidirectional:
                self._edges.append((dst_id, src_id, ep))
            count += 1
        return count

    def build(self):
        """Construct the frozen CSR graph. Uses native Rust backend if available."""
        from . import HAS_NATIVE

        if HAS_NATIVE:
            return self._build_native()
        return self._build_python()

    def _build_native(self):
        """Build using native Rust backend (fast)."""
        from _rgdb_core import build_graph

        # Convert edges to adjacency list format: adj[u] = [(dst, attenuation, angle_bin), ...]
        adj = [[] for _ in range(self._next_id)]
        for src, dst, ep in self._edges:
            adj[src].append((dst, ep.attenuation, ep.angle_bin))

        # Build node property arrays
        luminances = [self._node_props_overrides.get(i, NodeProps.uniform()).luminance
                      for i in range(self._next_id)]

        graph = build_graph(self._next_id, adj, node_luminances=luminances)

        # Store labels on the graph object for compatibility with query functions
        graph._node_labels = dict(self._id_to_label)
        graph._label_to_id = dict(self._label_to_id)
        graph._is_native = True
        return graph

    def _build_python(self) -> Graph:
        """Build using pure Python backend (fallback)."""
        graph = Graph(self._next_id)

        for node_id in range(self._next_id):
            if node_id in self._node_props_overrides:
                graph.set_node_props(node_id, self._node_props_overrides[node_id])
            label = self._id_to_label.get(node_id, str(node_id))
            graph.set_node_label(node_id, label)

        for src, dst, ep in self._edges:
            graph.add_edge(src, dst, ep)

        graph.freeze()
        return graph

    def get_label(self, node_id: int) -> Optional[str]:
        return self._id_to_label.get(node_id)

    def get_id(self, label: str) -> Optional[int]:
        return self._label_to_id.get(label)

    def get_type(self, node_id: int) -> Optional[str]:
        return self._node_types.get(node_id)

    def summary(self) -> str:
        """Return a human-readable summary of the graph."""
        from collections import Counter
        type_counts = Counter(self._node_types.values())
        lines = [
            f"Graph: {self.num_nodes:,} nodes, {self.num_edges:,} edges",
            "Node types:",
        ]
        for t, c in type_counts.most_common():
            lines.append(f"  {t}: {c:,}")
        return "\n".join(lines)
