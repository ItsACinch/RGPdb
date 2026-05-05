# Refractive Graph Database (RGDB)

> ## ⚠️ Early Beta — Expect Breaking Changes
>
> **Status:** Early beta · **Version:** 0.7 · **Stability:** Experimental
>
> RGDB is under active development. APIs, file formats, and on-disk
> representations are **not yet stable** and may change without notice between
> commits. The light-propagation model itself is still being tuned.
>
> **Not recommended for production use.** Suitable for research,
> experimentation, prototyping, and feedback. Pin to a specific git commit if
> you depend on current behavior, and expect to migrate. For production use cases, please reach out to discuss roadmap and timelines for stability. At this time, I'd recommend limiting usage to proofs of concepts, experiments or for jupyter notebook exploration/stages.
>
> Issues, ideas, and PRs are very welcome — this is the right time to
> influence the design.
>
> **Notes** This project has been under development on and off for several years. The current state is the result of multiple iterations and refactors. The core light propagation model has been stable for a while, but the API and file format have evolved significantly. The roadmap reflects the current plan, but may be adjusted as we learn from implementation and user feedback. We do have users reliably using RGDB in production for specific use cases, but the project as a whole is still in early beta due to limited user base.
>
> **Use at your own risk.** Please reach out if you have questions about stability or suitability for your use case. This project is provided "as is" without warranty of any kind. The author is not liable for any damages arising from the use of or inability to use this software.

A new type of graph-based reasoning and similarity system using physically-inspired “light propagation” across a directional, partitioned graph.

This project brief provides full background for developers, collaborators, and LLM-based engineering tools.

---

# Table of Contents
- [Refractive Graph Database (RGDB)](#refractive-graph-database-rgdb)
- [Table of Contents](#table-of-contents)
- [Concept Overview](#concept-overview)
- [Motivation](#motivation)
- [Core Ideas](#core-ideas)
    - [1. Nodes are “materials”](#1-nodes-are-materials)
    - [2. Edges are “optical paths”](#2-edges-are-optical-paths)
    - [3. Influence propagation](#3-influence-propagation)
    - [4. Room/portal structure](#4-roomportal-structure)
    - [5. Level file storage](#5-level-file-storage)
- [Mathematical Framework](#mathematical-framework)
  - [Node properties](#node-properties)
  - [Edge properties](#edge-properties)
  - [Angular distance](#angular-distance)
  - [Refraction factor](#refraction-factor)
  - [Influence update](#influence-update)
  - [Total influence](#total-influence)
  - [Light-distance metric](#light-distance-metric)
- [Node and Edge Properties](#node-and-edge-properties)
  - [Node](#node)
  - [Angle Bins and Refraction](#angle-bins-and-refraction)
  - [Propagation Algorithm](#propagation-algorithm)
  - [Rooms, Portals, and PVS](#rooms-portals-and-pvs)
    - [Rooms](#rooms)
    - [Portals](#portals)
    - [PVS (Potentially Visible Set)](#pvs-potentially-visible-set)
  - [Level File Format](#level-file-format)
  - [Comparison to Vector Databases](#comparison-to-vector-databases)
  - [Proof of Concept (Rust Implementation)](#proof-of-concept-rust-implementation)
  - [Features and Capabilities](#features-and-capabilities)
    - [Phase 1 — Core Engine ✅ Done](#phase-1--core-engine--done)
    - [Phase 2 — Rooms \& PVS ✅ Done](#phase-2--rooms--pvs--done)
    - [Phase 3 — Level File Format ✅ Done](#phase-3--level-file-format--done)
    - [Phase 4 — GPU Acceleration ✅ Done](#phase-4--gpu-acceleration--done)
    - [Phase 5 — Query Engine ✅ Done (API server pending)](#phase-5--query-engine--done-api-server-pending)
    - [Phase 6 — ML \& LLM Integration ✅ Done](#phase-6--ml--llm-integration--done)
    - [Phase 7 — Python \& Jupyter Notebook Support ✅ Done](#phase-7--python--jupyter-notebook-support--done)
    - [Phase 8 — Directional Luminance (in progress)](#phase-8--directional-luminance-in-progress)
    - [Future Work](#future-work)
  - [License](#license)

---

# Concept Overview
The Refractive Graph Database (RGDB) is a graph database and reasoning engine that computes **contextual, directional, and multi-hop similarity** using a physics-inspired model of light propagation.

Nodes behave like optical materials with:
- luminance,
- reflection, and
- refraction properties.

Edges behave like directional optical paths with:
- attenuation and
- angle bins (discretized direction vectors).

Queries propagate “light” from a source node across the graph. Influence bends and attenuates based on edge orientation and node refraction. Multi-hop paths with compatible directionality deliver more intensity.

---

# Motivation
Traditional vector databases:
- use global embeddings
- have no notion of multi-hop semantics
- cannot incorporate direction or path structure
- return the same neighbors regardless of context

Graph databases:
- perform literal traversal
- do not propagate influence or compute similarity
- have no directional compatibility or refraction models

RGDB introduces:
- a structured directional space,
- multi-hop influence propagation,
- dynamic similarity depending on path compatibility,
- room-based partitioning for fast pruning,
- GPU acceleration.

This allows highly contextual reasoning not possible in traditional systems.

---

# Core Ideas

### 1. Nodes are “materials”
- Luminance = emission
- Reflection = how much influence leaves the node
- Refraction = how much directional mismatch penalizes propagation

### 2. Edges are “optical paths”
- Attenuation = loss
- Angle bin = direction in discretized 2D manifold

### 3. Influence propagation
- Simulates directional light moving through materials
- Output = multi-hop influence score + light-distance metric

### 4. Room/portal structure
- Inspired by 2.5D FPS engines (DOOM, Wolf3D)
- Rooms partition large graphs
- Portals connect rooms
- PVS (Potentially Visible Sets) prune irrelevant regions

### 5. Level file storage
- Memory-aligned sections
- CSR topology
- Node/edge arrays
- PVS structures
- Direct mmap() → GPU-ready

---

# Mathematical Framework

Given graph \(G = (V,E)\):

## Node properties
- Luminance \( L(u) \)
- Reflection \( R(u) \in [0,1] \)
- Refraction index \( n(u) \ge 0 \)

## Edge properties
- Attenuation \( a(u,v) \in [0,1] \)
- Angle bin \( \theta(u,v) \in \{0,\ldots,B-1\} \)

## Angular distance
\[
\Delta = \min(|b_1 - b_2|,\; B - |b_1 - b_2|)
\]

## Refraction factor
\[
\rho = \exp\left(-k\cdot n(u)\cdot(\Delta/B)^2\right)
\]

## Influence update
\[
I_{t+1}(v,b_{uv}) =
\max_{u:(u,v)\in E}
\left[
I_t(u,b_{in})\cdot R(u)\cdot (1-a(u,v))\cdot \rho
\right]
\]

## Total influence
\[
I(v) = \sum_{b=0}^{B-1} I_T(v,b)
\]

## Light-distance metric
\[
d(s,v) = -\log(I(v)+\epsilon)
\]

This is NOT a metric:
- not symmetric  
- triangle inequality fails  
- but useful as contextual distance

---

# Node and Edge Properties

## Node
```text
struct NodeProps {
    luminance: f32,
    reflection: f32,
    refraction_index: f32
}
```
## Angle Bins and Refraction

Angle bins represent a 2D semantic plane of directions and map relationship semantics into discrete orientations.

Example:

- 16 bins (0–15)
- edge type → embedding → 2D PCA → atan2 → bin assignment

Directional compatibility is cheap:
- difference in bin indices
- wrap-around on a circle
- table-lookup refraction

This gives DOOM/Wolf3D-like 2.5D behavior (only angles in a plane).

## Propagation Algorithm

A frontier-based multi-hop expansion:

```cpp
Initialize I(source, initial_bin) = L(source)

For t in 1..max_depth:
    For each (u, b_in, intensity) in frontier:
        reflected = intensity * R(u)
        For each neighbor (v):
            bin_out = θ(u,v)
            ρ = refraction(b_in, bin_out, n(u))
            transmitted = reflected * (1 - a(u,v)) * ρ
            if transmitted >= min_intensity:
                update intensity[v][bin_out]
                push (v, bin_out, transmitted) to next frontier
```

Total intensity per node:
- sum across bins
Distance:
- negative log of intensity

## Rooms, Portals, and PVS
### Rooms
- Graph partitioned into communities / subgraphs
- Nodes in each room are contiguous in memory

### Portals
- Edges crossing room boundaries
- Marked with portal flag

### PVS (Potentially Visible Set)
For each (room R, angle bin b):
- a list of rooms that could receive influence
- prunes propagation during queries
Inspired by rendering optimizations in early FPS engines.

## Level File Format

Binary structure with aligned blocks:
```
[Header]
[NodeSection]
[EdgeSection]
[RoomsSection]
[PortalsSection]
[AngleTableSection]
[PVSSection]
[LocalLightSection?]
[UserMetadata]
```

All arrays stored in struct-of-arrays form, suitable for:
- direct mmap() into memory
- direct upload to GPU buffers

CSR is the core topology:
- row_ptr: Vec<usize> (size = num_nodes+1)
- col_idx: Vec<NodeId> (size = num_edges)
- edge_props aligned with col_idx

## Comparison to Vector Databases
| Capability                   | Vector DB | RGDB          |
| ---------------------------- | --------- | ------------- |
| Multi-hop reasoning          | ❌         | ✔             |
| Direction-aware similarity   | ❌         | ✔             |
| Path semantics               | ❌         | ✔             |
| Context-dependent similarity | ❌         | ✔             |
| Room-based pruning           | ❌         | ✔             |
| GPU propagation              | partial   | ✔             |
| Simplicity                   | ✔         | ❌             |
| Update cost                  | low       | moderate-high |

RGDB does not replace vector DBs—it supersets them for multi-hop reasoning.

## Proof of Concept (Rust Implementation)

A working PoC is implemented using:
- CSR graph
- NodeProps / EdgeProps
- frontier-based propagation
- angular refraction model

Key code concepts:
```rust
struct Graph {
    row_ptr: Vec<usize>,
    col_idx: Vec<NodeId>,
    node_props: Vec<NodeProps>,
    edge_props: Vec<EdgeProps>
}
```
Propagation engine:

```rust
fn propagate_light(graph: &Graph, source: NodeId, initial_bin: AngleBin, params: LightParams)
    -> Vec<f32>
```

This produces:
- intensity per node
- light-distance per node
Changing angle bins demonstrates refraction effects.

## Features and Capabilities

### Phase 1 — Core Engine ✅ Done
- CSR graph
- node/edge properties
- influence propagation
- refraction model
- Rust PoC

### Phase 2 — Rooms & PVS ✅ Done
- graph partitioning (Connected Components, BFS)
- portal detection
- PVS computation
- propagation pruning

### Phase 3 — Level File Format ✅ Done
- binary writer/reader
- aligned sections
- mmap integration
- versioning

### Phase 4 — GPU Acceleration ✅ Done
- CUDA compute kernels (via `cudarc`)
- frontier streaming
- atomic intensity accumulation
- memory optimization passes
- See `docs/CUDA_COMPLETE.md` and `docs/CUDA_WEEK1_OPTIMIZATION_RESULTS.md`

### Phase 5 — Query Engine ✅ Done (API server pending)
- top-k influence queries
- distance queries
- hybrid semantic queries (graph + vector similarity)
- embedding export
- ⏳ REST API server (Axum) — planned

### Phase 6 — ML & LLM Integration ✅ Done
- multi-hop contextual embeddings
- LLM-driven queries with intent classification
- RAG module (`src/rag/`) combining propagation, vector search, and personalization
- See `docs/LLM_INTEGRATION_GUIDE.md`

### Phase 7 — Python & Jupyter Notebook Support ✅ Done
- Pip-installable package: `rgdb-embeddings/`
- Python bindings for graph construction and querying
- `GraphBuilder` API for building graphs from pandas DataFrames in Jupyter
- Query helpers returning pandas DataFrames for notebook workflows
- Embedding training pipeline (pretrained, RotatE, GNN) with CLI:
  `rgdb-embed process | train | evaluate | info`
- See `rgdb-embeddings/README.md`

### Phase 8 — Directional Luminance (in progress)
- Per-angle-bin luminance emission (replacing uniform luminance)
- `RelationshipProperty` enum and `PropertyAngleMap`
- See `docs/DIRECTIONAL_LUMINANCE_PLAN.md`

### Future Work
- Ingestion pipeline (entity/relationship extraction)
- Reasoning path tracking
- Incremental graph updates
- Monitoring and metrics

## License

**This is a temporary license.** RGDB is currently distributed under the
[PolyForm Noncommercial License 1.0.0](./LICENSE). Free to use for personal,
research, educational, and noncommercial open source projects, with attribution
required. Commercial and for-profit use is not permitted at this time.

The copyright holder intends to permit commercial use in the future under terms
to be announced. For commercial licensing inquiries in the meantime, contact
maarten@acinch.com.

> ⚠️ **No warranty. Use at your own risk.** RGDB is provided **as-is, without
> warranty of any kind**, express or implied. The authors and copyright holders
> accept no liability for any damages arising from its use. See the
> [LICENSE](./LICENSE) file for the full disclaimer.