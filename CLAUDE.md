# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RGDB (Refractive Graph Database) is a Rust-based graph database that uses physics-inspired "light propagation" for computing contextual, multi-hop similarity between nodes. Nodes behave like optical materials with reflection and refraction properties. Edges are typed (relation id) and carry attenuation. Propagation is a sparse, typed personalized-PageRank (PPR) diffusion where a relation-similarity matrix ("refraction") penalizes hops that change relation type.

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

1. **Graph Creation** (`graph.rs`): CSR (Compressed Sparse Row) format with `NodeProps` (reflection, refraction_index) and `EdgeProps` (attenuation, relation, is_portal)
2. **Partitioning** (`partitioning.rs`): Divide graph into rooms using Connected Components or BFS partitioning (unchanged by the rewrite)
3. **Relation Vocabulary** (`relation.rs`): Data-driven `RelationVocab` mapping relation names <-> `RelationId`, plus an n×n similarity matrix used for refraction
4. **Light Propagation** (`propagation.rs`): Sparse, typed-PPR multi-hop diffusion from seed nodes, with relation-similarity refraction penalizing type-changing hops
5. **Queries** (`queries.rs`): Top-K influence, distance queries, hybrid (graph + vector similarity) queries

### Module Responsibilities

- `graph.rs` - Core CSR graph structure, NodeProps, EdgeProps, type aliases (`NodeId=u32`, `RelationId=u16`, `RoomId=u32`)
- `relation.rs` - `RelationVocab`: data-driven relation names + n×n similarity matrix (`uniform()` for pure PPR / no refraction, `new()`, `id_of()`, `similarity()`)
- `propagation.rs` - `propagate()` (multi-seed) and `propagate_single()` functions, `PropagationParams` configuration
- `partitioning.rs` - `partition_graph()` with `PartitioningAlgorithm` enum (ConnectedComponents, BFSPartitioning) - kept as-is
- `rooms.rs` - `RoomCollection`, portal detection via `mark_portals()` - kept as-is
- `level_file.rs` - Binary serialization v2 with mmap support (write_level_file, read_level_file_mmap); header + nodes + CSR edges + room map + rooms + relation vocab section
- `queries.rs` - `query_top_k_influence()`, `query_distance()`, `query_hybrid()` - built on `propagate()`
- `embeddings.rs` - Multi-hop contextual embedding generation via `propagate_single()`
- `cuda/` - GPU acceleration; **stale**, gated off (see "Stale/deferred backends" below)
- `llm_integration.rs` - llm-chain query parsing; **stale**, gated off (see below)
- `rag/` - RAG (Retrieval-Augmented Generation) module for LLM integration

Removed in the rewrite: `pvs.rs` (Potentially Visible Set precomputation/query pruning) and `property_map.rs` (`RelationshipProperty`/`PropertyAngleMap` for directional luminance) no longer exist. There are no angle bins and no PVS pruning; rooms/partitioning/portals remain but are no longer used for PVS.

### RAG Module (`src/rag/`)

The RAG module provides hybrid retrieval combining graph reasoning with vector similarity:

- `embedding_store.rs` - Vector storage with cosine similarity and top-K search
- `intent.rs` - Query intent classification (Definition, Causation, Requirements, etc.) → canonical relation NAME (e.g. `"Causes"`, `"IsA"`), looked up in the graph's `RelationVocab` via `id_of()` (untyped query if the vocab lacks it)
- `personalization.rs` - `UserContext` for personalized ranking (interaction history, session recency, room-based access control). Angle-bin/topic-affinity boosts are gone; access control and interaction/session boosts remain.
- `query_engine.rs` - `RAGQueryEngine` combining typed-PPR propagation + vector similarity + personalization via calibrated fusion: seed weights are normalized to sum to 1 (not max-normalized), and `score = (alpha*graph + beta*vector) * (1 + gamma*(boost-1))`

**Usage:**
```rust
use rgdb::{RAGQueryEngine, EmbeddingStore, QueryConfig, UserContext, RelationVocab};

let engine = RAGQueryEngine::new(graph, vocab, embedding_store);
let results = engine.query("What causes X?", &query_embedding, Some(&user_ctx), &config);
```

See `docs/LLM_INTEGRATION_GUIDE.md` for LLM tool definitions and example prompts.

### Key Constants

There are no angle bins in the rewrite (`N_ANGLE_BINS` is gone). Defaults:

- `PropagationParams`: `max_depth = 4`, `min_intensity = 1e-3`
- `NodeProps`: `reflection = 0.85` (per-node continuation probability, in [0,1]), `refraction_index = 1.0` (exponent applied to relation similarity)
- `EdgeProps`: `attenuation = 0.0`, `relation = 0`, `is_portal = false`

### Propagation Formula

Sparse typed-PPR diffusion from seed(s), with path-internal refraction. For a hop `u -> v`:

```
contribution(v) += m · reflection(u) · p(u→v) · sim(r_in, r_out)^refraction_index(u)

p(u→v)   = (1 - attenuation(u→v)) / Σ_w (1 - attenuation(u→w))   // row-stochastic over u's out-edges
r_out    = relation(u→v)
sim(a,b) = RelationVocab::similarity(a, b) in [0,1]; sim(None, ·) = 1.0 (untyped/first hop)
```

`m` is the mass arriving at `u` via incoming relation `r_in` (the query relation on the first hop, then whatever relation was last traversed). Total intensity at each reached node is accumulated as a **sum over all paths** reaching it (not a max). `RelationVocab::uniform(n)` sets all similarities to 1.0, which disables refraction entirely (pure typed PPR). Multi-seed propagation (`propagate()`) is exactly the sum of single-seed propagations (`propagate_single()`) — the model is linear in seed mass.

## Dependencies

Key crates: `petgraph` (partitioning), `hashbrown` (fast HashMaps), `memmap2` (mmap I/O), `byteorder` (binary I/O), `ndarray` (vectors), `cudarc` (CUDA bindings), `rayon` (parallel multi-seed propagation)

### Stale/deferred backends

Two optional backends target the pre-rewrite angle-bin model and are intentionally disabled with a `compile_error!` at the top of the module, so `--features <name>` fails fast with a clear message instead of a confusing cascade:

- `cuda/` (`--features cuda`) - dense angle-bin GPU kernels, not ported to the sparse typed-PPR model
- `llm_integration.rs` (`--features llm`) - llm-chain query parsing against the old `AngleBin`/`LightParams`/`propagate_light` API, not ported

Both are off by default; do not build with these features until they are ported. Separately, the `rgdb-embeddings` pandas/Jupyter `GraphBuilder` also still targets the old model and currently falls back to a pure-Python path (`HAS_NATIVE = False` in `rgdb_embeddings/graph/__init__.py`) rather than the native bindings.

### Evaluation harness (`rgdb-eval/`)

A Python retrieval-quality evaluation harness, separate from the Rust crate's own tests. Loads MetaQA and synthetic typed-graph datasets (`dataset.py`, `metaqa.py`, `synthetic.py`), builds several contenders - vector and vector-two-hop baselines, a PPR baseline, and the new-RGDB contender (`rankers/rgdb_new.py`, built on the native `_rgdb_core` bindings) run in both `refraction` and `uniform` (ablation, refraction disabled) modes - and scores them with Hits@k / Recall@k / MRR (`metrics.py`). Run via `python -m rgdb_eval.run` (see `report.py` for markdown output). Tests live in `rgdb-eval/tests` and only need `numpy`, `scipy`, `pandas`, and the built native bindings - not `sentence-transformers`/`torch` (vector tests inject fixed vectors; `NewRgdbRanker`'s test uses `vocab_mode="uniform"`).

## Current Development

The typed-PPR relation model (v0.2.0) described above is the current state of the codebase - it supersedes the old angle-bin/luminance model. The former `directional_luminance` branch (per-angle-bin luminance emission, see the now-obsolete `DIRECTIONAL_LUMINANCE_PLAN.md`) has been superseded by this rewrite; `property_map.rs` and the `RelationshipProperty` enum it introduced no longer exist.

## Code Patterns

- Use `Result<T, E>` for fallible operations, not `assert!()`
- CSR graph structure: `row_ptr[node]..row_ptr[node+1]` indexes into `col_idx` and `edge_props`
- Iterating neighbors: `for (neighbor_id, edge_props) in graph.neighbors(node_id)`
- Room lookup: `graph.get_room(node_id)` returns RoomId
