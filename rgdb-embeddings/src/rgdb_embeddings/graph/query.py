"""Query helpers -- top-k influence, distance, and path tracing.

Returns results as pandas DataFrames for easy use in Jupyter notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from .core import Graph
from .propagation import LightParams, propagate_light


@dataclass
class QueryResult:
    """Single result from a propagation query."""
    node_id: int
    label: str
    intensity: float
    node_type: Optional[str] = None
    distance: Optional[float] = None


def query_top_k(
    graph: Graph,
    source: int,
    initial_bin: int = 0,
    k: int = 20,
    params: Optional[LightParams] = None,
    node_type_filter: Optional[str] = None,
    builder=None,
) -> pd.DataFrame:
    """Find the top-k nodes most influenced by source via light propagation.

    Args:
        graph: The RGDB graph.
        source: Source node ID.
        initial_bin: Starting angle bin for propagation.
        k: Number of results to return.
        params: Propagation parameters.
        node_type_filter: If set, only return nodes of this type.
        builder: GraphBuilder instance (for labels and type info).

    Returns:
        DataFrame with columns: node_id, label, node_type, intensity, distance
    """
    intensities = propagate_light(graph, source, initial_bin, params)

    # Build results
    results = []
    for node_id in range(len(intensities)):
        if node_id == source:
            continue
        intensity = intensities[node_id]
        if intensity <= 0:
            continue

        # Get label: try builder first, then graph._node_labels (native), then graph method
        if builder:
            label = builder.get_label(node_id) or str(node_id)
        elif hasattr(graph, '_node_labels'):
            label = graph._node_labels.get(node_id, str(node_id))
        elif hasattr(graph, 'get_node_label'):
            label = graph.get_node_label(node_id) or str(node_id)
        else:
            label = str(node_id)
        node_type = builder.get_type(node_id) if builder else None

        if node_type_filter and node_type != node_type_filter:
            continue

        distance = -np.log(intensity + 1e-10)
        results.append({
            "node_id": node_id,
            "label": label,
            "node_type": node_type,
            "intensity": float(intensity),
            "distance": float(distance),
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    return df.nlargest(k, "intensity").reset_index(drop=True)


def query_distance(
    graph: Graph,
    source: int,
    target: int,
    initial_bin: int = 0,
    params: Optional[LightParams] = None,
) -> float:
    """Compute the propagation distance between source and target.

    Returns -log(intensity) or inf if no path exists.
    """
    intensities = propagate_light(graph, source, initial_bin, params)
    intensity = intensities[target]
    if intensity <= 0:
        return float("inf")
    return -np.log(intensity + 1e-10)


def query_influence_comparison(
    graph: Graph,
    sources: list[int],
    initial_bin: int = 0,
    k: int = 20,
    params: Optional[LightParams] = None,
    builder=None,
) -> pd.DataFrame:
    """Compare influence from multiple source nodes.

    Returns a DataFrame with intensity from each source for the top-k most
    influenced nodes across all sources combined.
    """
    all_intensities = {}
    for src in sources:
        intensities = propagate_light(graph, src, initial_bin, params)
        src_label = (builder.get_label(src) if builder else
                     graph._node_labels.get(src, str(src)) if hasattr(graph, '_node_labels') else
                     graph.get_node_label(src) if hasattr(graph, 'get_node_label') else str(src))
        all_intensities[src_label] = intensities

    # Find top-k by max intensity across any source
    max_intensity = np.zeros(graph.num_nodes, dtype=np.float32)
    for intensities in all_intensities.values():
        max_intensity = np.maximum(max_intensity, intensities)

    # Exclude source nodes
    for src in sources:
        max_intensity[src] = 0

    top_k_ids = np.argsort(max_intensity)[-k:][::-1]
    top_k_ids = top_k_ids[max_intensity[top_k_ids] > 0]

    rows = []
    for node_id in top_k_ids:
        row = {
            "node_id": int(node_id),
            "label": (builder.get_label(int(node_id)) if builder else
                      graph._node_labels.get(int(node_id), str(node_id)) if hasattr(graph, '_node_labels') else
                      graph.get_node_label(int(node_id)) if hasattr(graph, 'get_node_label') else str(node_id)),
            "node_type": builder.get_type(int(node_id)) if builder else None,
        }
        for src_label, intensities in all_intensities.items():
            row[f"intensity_{src_label}"] = float(intensities[node_id])
        rows.append(row)

    return pd.DataFrame(rows)
