# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RGDB (Refractive Graph Database) is a Rust-based graph database that uses physics-inspired "light propagation" for computing contextual, directional, multi-hop similarity between nodes. Nodes behave like optical materials with luminance, reflection, and refraction properties. Edges act as directional optical paths with attenuation and angle bins.

## Build Commands

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run a single test
cargo test test_name

# Run the example binary
cargo run

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## Architecture

### Core Data Flow

1. **Graph Creation** (`graph.rs`): CSR (Compressed Sparse Row) format with `NodeProps` (luminance, reflection, refraction_index, default_angle_bin) and `EdgeProps` (attenuation, angle_bin, is_portal)
2. **Partitioning** (`partitioning.rs`): Divide graph into rooms using Connected Components or BFS partitioning
3. **PVS Computation** (`pvs.rs`): Precompute Potentially Visible Sets per (room, angle_bin) for query optimization
4. **Light Propagation** (`propagation.rs`): Multi-hop influence propagation with refraction model and PVS-based pruning
5. **Queries** (`queries.rs`): Top-K influence, distance queries, hybrid (graph + vector similarity) queries

### Module Responsibilities

- `graph.rs` - Core CSR graph structure, NodeProps, EdgeProps, type aliases (NodeId=u32, AngleBin=u8, RoomId=u32)
- `propagation.rs` - `propagate_light()` and `propagate_light_with_pvs()` functions, `LightParams` configuration
- `partitioning.rs` - `partition_graph()` with `PartitioningAlgorithm` enum (ConnectedComponents, BFSPartitioning)
- `rooms.rs` - `RoomCollection`, portal detection via `mark_portals()`
- `pvs.rs` - PVS computation and lookup for room-based query pruning
- `level_file.rs` - Binary serialization with mmap support (write_level_file, read_level_file_mmap)
- `queries.rs` - `query_top_k_influence()`, `query_distance()`, `query_hybrid()`
- `embeddings.rs` - Multi-hop contextual embedding generation
- `property_map.rs` - RelationshipProperty enum and PropertyAngleMap for directional luminance
- `cuda/` - GPU acceleration (requires CUDA toolkit at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6)
- `rag/` - RAG (Retrieval-Augmented Generation) module for LLM integration

### RAG Module (`src/rag/`)

The RAG module provides hybrid retrieval combining graph reasoning with vector similarity:

- `embedding_store.rs` - Vector storage with cosine similarity and top-K search
- `intent.rs` - Query intent classification (Definition, Causation, Requirements, etc.) → angle bin mapping
- `personalization.rs` - UserContext for personalized ranking (topic affinities, interaction history, access control)
- `query_engine.rs` - RAGQueryEngine combining multi-source propagation + vector similarity + personalization

**Usage:**
```rust
use rgdb::{RAGQueryEngine, EmbeddingStore, QueryConfig, UserContext};

let engine = RAGQueryEngine::new(graph, embedding_store, pvs);
let results = engine.query("What causes X?", &query_embedding, Some(&user_ctx), &config);
```

See `docs/LLM_INTEGRATION_GUIDE.md` for LLM tool definitions and example prompts.

### Key Constants

- `N_ANGLE_BINS`: 16 (discretized 2D semantic directions)
- Default `LightParams`: k=5.0, min_intensity=1e-3, max_depth=4

### Propagation Formula

```
intensity_out = intensity_in × reflection × (1 - attenuation) × refraction_factor
refraction_factor = exp(-k × refraction_index × (angular_distance/N_ANGLE_BINS)²)
```

## Dependencies

Key crates: `petgraph` (partitioning), `hashbrown` (fast HashMaps), `memmap2` (mmap I/O), `byteorder` (binary I/O), `ndarray` (vectors), `cudarc` (CUDA bindings)

## Current Development

The `directional_luminance` branch is implementing per-angle-bin luminance emission instead of uniform luminance. See `DIRECTIONAL_LUMINANCE_PLAN.md` for design details and `property_map.rs` for the RelationshipProperty enum.

## Code Patterns

- Use `Result<T, E>` for fallible operations, not `assert!()`
- CSR graph structure: `row_ptr[node]..row_ptr[node+1]` indexes into `col_idx` and `edge_props`
- Iterating neighbors: `for (neighbor_id, edge_props) in graph.neighbors(node_id)`
- Room lookup: `graph.get_room(node_id)` returns RoomId
