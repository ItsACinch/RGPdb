"""RGDB Graph module -- uses native Rust backend when available, falls back to pure Python."""

from .core import Graph, NodeProps, EdgeProps, N_ANGLE_BINS
from .propagation import LightParams, propagate_light
from .builder import GraphBuilder
from .query import QueryResult, query_top_k, query_distance

# Check for native Rust backend
try:
    from _rgdb_core import (
        build_graph as _native_build_graph,
        propagate_light as _native_propagate_light,
        query_top_k as _native_query_top_k,
        LightParams as NativeLightParams,
        N_ANGLE_BINS as _native_N_ANGLE_BINS,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

__all__ = [
    "Graph", "NodeProps", "EdgeProps", "N_ANGLE_BINS",
    "LightParams", "propagate_light",
    "GraphBuilder",
    "QueryResult", "query_top_k", "query_distance",
    "HAS_NATIVE",
]
