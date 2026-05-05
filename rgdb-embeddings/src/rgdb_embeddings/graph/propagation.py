"""Light propagation engine -- Python port of src/propagation.rs.

Implements the core BFS-like propagation with refraction penalty:
    intensity_out = intensity_in * reflection * (1 - attenuation) * refraction_factor
    refraction_factor = exp(-k * n * (delta / B)^2)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .core import Graph, N_ANGLE_BINS


@dataclass
class LightParams:
    """Parameters for light propagation (mirrors Rust LightParams)."""
    k: float = 5.0
    min_intensity: float = 1e-3
    max_depth: int = 4
    num_angle_bins: int = N_ANGLE_BINS


def angular_distance(b1: int, b2: int, num_bins: int) -> int:
    """Circular angular distance between two bins."""
    diff = abs(b1 - b2)
    return min(diff, num_bins - diff)


def refraction_factor(bin_in: int, bin_out: int, n: float, params: LightParams) -> float:
    """Compute refraction penalty: rho = exp(-k * n * (delta/B)^2)."""
    if params.num_angle_bins == 0:
        return 0.0
    delta = angular_distance(bin_in, bin_out, params.num_angle_bins)
    b = params.num_angle_bins
    x = (delta / b) ** 2
    return np.exp(-params.k * n * x)


def propagate_light(
    graph,
    source: int,
    initial_bin: int = 0,
    params: Optional[LightParams] = None,
) -> np.ndarray:
    """Perform refractive light propagation from a source node.

    Returns total intensity per node as a numpy array of shape (num_nodes,).
    Uses native Rust backend automatically if the graph was built with it.
    """
    # Dispatch to native backend if available
    if getattr(graph, '_is_native', False):
        from _rgdb_core import propagate_light as _native_propagate, LightParams as NativeLightParams
        if params is None:
            params = LightParams()
        native_params = NativeLightParams(k=params.k, min_intensity=params.min_intensity, max_depth=params.max_depth)
        return _native_propagate(graph, source, initial_bin, native_params)

    if params is None:
        params = LightParams()

    if not graph._frozen:
        graph.freeze()

    n = graph.num_nodes
    b = params.num_angle_bins

    if n == 0 or b == 0:
        return np.zeros(0, dtype=np.float32)

    # Per-(node, angle_bin) intensities
    intensities = np.zeros(n * b, dtype=np.float32)
    total_intensity = np.zeros(n, dtype=np.float32)

    src_props = graph.node_props(source)

    # Initialize frontier
    frontier = []
    initial_intensity = max(src_props.luminance, 1.0)
    bin_idx = initial_bin % b
    intensities[source * b + bin_idx] = initial_intensity
    total_intensity[source] += initial_intensity
    frontier.append((source, bin_idx, initial_intensity))

    # BFS propagation
    for _depth in range(params.max_depth):
        if not frontier:
            break

        next_frontier = []

        for u, bin_in, intensity_in in frontier:
            if intensity_in < params.min_intensity:
                continue

            u_props = graph.node_props(u)
            reflected = intensity_in * u_props.reflection

            for v, eprops in graph.neighbors(u):
                bin_out = eprops.angle_bin
                n_u = u_props.refraction_index
                rho = refraction_factor(bin_in, bin_out, n_u, params)
                attenuation_factor = 1.0 - eprops.attenuation
                transmitted = reflected * attenuation_factor * rho

                if transmitted < params.min_intensity:
                    continue

                idx = v * b + bin_out
                if transmitted > intensities[idx]:
                    intensities[idx] = transmitted
                    total_intensity[v] = max(total_intensity[v], total_intensity[v] + transmitted - intensities[idx])
                    # Simpler: accumulate
                    total_intensity[v] += transmitted - intensities[idx] if intensities[idx] > 0 else transmitted
                    next_frontier.append((v, bin_out, transmitted))

        frontier = next_frontier

    # Recompute total from per-bin intensities for accuracy
    total_intensity = np.zeros(n, dtype=np.float32)
    for node_id in range(n):
        start = node_id * b
        total_intensity[node_id] = intensities[start:start + b].sum()

    return total_intensity
