# RGDB Sub-project B — Core Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the order-dependent, dense, angle-bin propagation with a principled, sparse, calibrated typed-PPR kernel featuring path-internal refraction; drop PVS; bump the file format; feature-gate CUDA off; and add the new-RGDB contender (plus its no-refraction ablation) to the eval harness.

**Architecture:** A new `RelationVocab` (data-driven relation vocabulary + similarity matrix) replaces the hardcoded `RelationshipProperty` enum. `NodeProps` becomes `{reflection, refraction_index}` and `EdgeProps` carries a `relation: RelationId`. The kernel is a sparse frontier walk keyed on `(node, incoming_relation)` that sums path products with per-node damping and per-hop `sim(prev_relation, next_relation)` refraction; multi-seed diffusion is parallel-then-merge (linear). PVS is deleted; rooms/partitioning/portals stay. The level file gains a relation-vocab section and drops PVS/angle/directional sections.

**Tech Stack:** Rust (workspace crate `rgdb`), rayon (parallel seeds), hashbrown (sparse maps), pyo3/maturin (bindings), Python eval harness from sub-project C.

## Global Constraints

- **Free rein on compatibility:** break the Rust API and file format. Bump crate version `0.1.0 → 0.2.0` (both `rgdb` and `rgdb-python`); bump level-file `VERSION 1 → 2`.
- **Kernel definition (authoritative):** for state `(u, r_in)` with mass `m`, along edge `u→v` of relation `r_out`:
  `contribution(v) += m · reflection(u) · p(u→v) · sim(r_in, r_out)^refraction_index(u)`
  where `p(u→v) = (1−attenuation(u→v)) / Σ_w (1−attenuation(u→w))` (row-stochastic), `reflection(u) ∈ [0,1]` (per-node damping, default **0.85**), `refraction_index(u)` default **1.0**, and `sim(None, ·) = 1.0` (first hop of an untyped query, and any untyped seed).
- **Seeding:** each seed injects initial mass; the seed's own mass is added to its total (PPR restart mass). Multi-seed = parallel single-seed then summed (mathematically exact by linearity).
- **Type names (fixed across tasks):** `RelationId = u16`; `NodeProps { reflection: f32, refraction_index: f32 }`; `EdgeProps { attenuation: f32, relation: RelationId, is_portal: bool }`; `PropagationParams { max_depth: usize, min_intensity: f32 }`; kernel `propagate(graph, vocab, seeds: &[(NodeId, f32)], query_relation: Option<RelationId>, params) -> HashMap<NodeId, f32>`.
- **Rooms kept, PVS removed.** `rooms.rs`, `partitioning.rs`, portals stay. `pvs.rs` is deleted.
- **CUDA:** off by default (already `#[cfg(feature = "cuda")]`); do not port. Make `--features cuda` fail fast with a clear staleness message.
- Commit after each task. `cargo test -p rgdb` must be green at the end of every task (a module may be temporarily disabled within a task, but never left broken at a task boundary).

---

### Task 1: `RelationVocab` (new `relation.rs`)

**Files:**
- Create: `rgdb/src/relation.rs`
- Modify: `rgdb/src/lib.rs` (add `pub mod relation;`)

**Interfaces:**
- Produces:
  - `RelationId = u16` (defined in `graph.rs` in Task 2; here `relation.rs` defines its own until then — see step 1 note).
  - `RelationVocab::new(names: Vec<String>, similarity: Vec<f32>) -> Result<Self, RelationVocabError>`
  - `RelationVocab::uniform(n: usize) -> Self` (all similarities 1.0 — the no-refraction ablation)
  - `RelationVocab::with_names_uniform(names: Vec<String>) -> Self`
  - `len`, `is_empty`, `name(id) -> Option<&str>`, `id_of(name) -> Option<u16>`, `similarity(a, b) -> f32`

**Note:** This task compiles alongside the *old* types. It defines `RelationId` locally as `u16` here and Task 2 moves the canonical `pub type RelationId = u16;` to `graph.rs`; in Task 2, `relation.rs` switches to `use crate::graph::RelationId;`. For now, define it here.

- [ ] **Step 1: Write the failing test**

Create `rgdb/src/relation.rs` with only the test module first (so it fails to compile / fail):

```rust
//! Data-driven relation vocabulary and similarity matrix.

pub type RelationId = u16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_has_all_ones() {
        let v = RelationVocab::uniform(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v.similarity(0, 2), 1.0);
        assert_eq!(v.similarity(1, 1), 1.0);
    }

    #[test]
    fn new_validates_matrix_shape() {
        assert!(RelationVocab::new(vec!["a".into(), "b".into()], vec![1.0; 4]).is_ok());
        assert!(RelationVocab::new(vec!["a".into(), "b".into()], vec![1.0; 3]).is_err());
    }

    #[test]
    fn lookup_by_name_and_similarity() {
        let names = vec!["isa".into(), "causes".into()];
        let sim = vec![1.0, 0.2, 0.2, 1.0]; // row-major 2x2
        let v = RelationVocab::new(names, sim).unwrap();
        assert_eq!(v.id_of("causes"), Some(1));
        assert_eq!(v.id_of("missing"), None);
        assert_eq!(v.similarity(0, 1), 0.2);
        assert_eq!(v.name(1), Some("causes"));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rgdb relation:: 2>&1 | tail -5`
Expected: compile error (`cannot find ... RelationVocab`).

- [ ] **Step 3: Implement `RelationVocab`**

Prepend to `rgdb/src/relation.rs` (above the test module):

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RelationVocabError {
    #[error("similarity matrix must be {expected} entries ({n}x{n}), got {got}")]
    BadShape { n: usize, expected: usize, got: usize },
}

/// A relation vocabulary with a symmetric similarity matrix (row-major, n x n).
#[derive(Debug, Clone)]
pub struct RelationVocab {
    names: Vec<String>,
    similarity: Vec<f32>,
}

impl RelationVocab {
    pub fn new(names: Vec<String>, similarity: Vec<f32>) -> Result<Self, RelationVocabError> {
        let n = names.len();
        let expected = n * n;
        if similarity.len() != expected {
            return Err(RelationVocabError::BadShape { n, expected, got: similarity.len() });
        }
        Ok(Self { names, similarity })
    }

    /// All-ones similarity: disables refraction (pure typed PPR). Names are `r0..r{n-1}`.
    pub fn uniform(n: usize) -> Self {
        Self { names: (0..n).map(|i| format!("r{i}")).collect(), similarity: vec![1.0; n * n] }
    }

    /// All-ones similarity with caller-supplied names.
    pub fn with_names_uniform(names: Vec<String>) -> Self {
        let n = names.len();
        Self { names, similarity: vec![1.0; n * n] }
    }

    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }

    pub fn name(&self, id: RelationId) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    pub fn id_of(&self, name: &str) -> Option<RelationId> {
        self.names.iter().position(|n| n == name).map(|i| i as RelationId)
    }

    /// Similarity in [0,1]; returns 0.0 for out-of-range ids.
    pub fn similarity(&self, a: RelationId, b: RelationId) -> f32 {
        let n = self.names.len();
        let (ai, bi) = (a as usize, b as usize);
        if ai >= n || bi >= n { return 0.0; }
        self.similarity[ai * n + bi]
    }
}
```

- [ ] **Step 4: Register the module**

In `rgdb/src/lib.rs`, add after `pub mod property_map;`:

```rust
pub mod relation;
```

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p rgdb relation:: 2>&1 | tail -8`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/relation.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): add data-driven RelationVocab + similarity matrix"
```

---

### Task 2: Core type + kernel migration

**Files:**
- Modify: `rgdb/src/graph.rs` (new `NodeProps`/`EdgeProps`, `RelationId`, drop angle/luminance, constructor tweaks)
- Rewrite: `rgdb/src/propagation.rs` (sparse typed-PPR kernel + hand-computed tests)
- Modify: `rgdb/src/relation.rs` (use `crate::graph::RelationId`)
- Modify: `rgdb/src/queries.rs`, `rgdb/src/embeddings.rs` (new signatures)
- Modify: `rgdb/src/error.rs` (drop `PVSError`, fix `PropagationError`)
- Modify: `rgdb/src/rooms.rs`, `rgdb/src/partitioning.rs` (only if they reference removed items — they use `EdgeProps::default()`, which still exists)
- Modify: `rgdb/src/main.rs` (demo uses new API)
- Rewrite: `rgdb/benches/directional_luminance.rs` (new kernel bench)
- Delete: `rgdb/src/pvs.rs`, `rgdb/src/property_map.rs`
- Modify: `rgdb/src/lib.rs` (drop `pvs`, `property_map`; **temporarily** disable `rag`, `llm_integration`, `level_file`; add `rayon`)
- Modify: `rgdb/Cargo.toml` (add `rayon`)

**Interfaces:**
- Produces (consumed by Tasks 3–8):
  - `pub type RelationId = u16;` (in `graph.rs`)
  - `NodeProps { reflection: f32, refraction_index: f32 }`, `Default` = `{0.85, 1.0}`
  - `EdgeProps { attenuation: f32, relation: RelationId, is_portal: bool }`, `Default` = `{0.0, 0, false}`
  - `PropagationParams { max_depth: usize, min_intensity: f32 }`, `Default` = `{4, 1e-3}`
  - `propagate(graph, vocab, seeds, query_relation, params) -> HashMap<NodeId, f32>`
  - `propagate_single(graph, vocab, seed, initial_mass, query_relation, params) -> HashMap<NodeId, f32>`
  - `intensity_to_distance(&[f32], eps) -> Vec<f32>` (unchanged)

- [ ] **Step 1: Add `rayon` dependency**

In `rgdb/Cargo.toml`, under `[dependencies]`, add:

```toml
rayon = "1.10"
```

- [ ] **Step 2: Rewrite `NodeProps`/`EdgeProps` and add `RelationId` in `graph.rs`**

In `rgdb/src/graph.rs`, replace the type aliases block (lines ~6–11) with:

```rust
pub type NodeId = u32;
pub type RelationId = u16;
pub type RoomId = u32;
```

Replace the entire `NodeProps` definition + `impl Default` + `impl NodeProps` block (lines ~18–89) with:

```rust
/// Per-node material properties.
#[derive(Debug, Clone, Copy)]
pub struct NodeProps {
    /// Continuation probability (per-node damping), in [0,1].
    pub reflection: f32,
    /// Exponent applied to relation-similarity (refraction sharpness).
    pub refraction_index: f32,
}

impl Default for NodeProps {
    fn default() -> Self {
        Self { reflection: 0.85, refraction_index: 1.0 }
    }
}
```

Replace the `EdgeProps` definition + `impl Default` (lines ~91–110) with:

```rust
/// Per-edge properties.
#[derive(Debug, Clone, Copy)]
pub struct EdgeProps {
    /// Fraction of intensity lost along this edge, in [0,1].
    pub attenuation: f32,
    /// Relation type id (indexes into the RelationVocab).
    pub relation: RelationId,
    /// Whether this edge crosses a room boundary.
    pub is_portal: bool,
}

impl Default for EdgeProps {
    fn default() -> Self {
        Self { attenuation: 0.0, relation: 0, is_portal: false }
    }
}
```

Remove the now-unused import `use crate::property_map::RelationshipProperty;` at the top of `graph.rs`. Leave `from_csr`, `from_adjacency`, `validate`, `neighbors`, etc. unchanged (they are generic over the prop types).

- [ ] **Step 3: Point `relation.rs` at the canonical `RelationId`**

In `rgdb/src/relation.rs`, replace `pub type RelationId = u16;` with:

```rust
use crate::graph::RelationId;
```

- [ ] **Step 4: Write the new kernel with hand-computed tests**

Replace the entire contents of `rgdb/src/propagation.rs` with:

```rust
//! Sparse typed-PPR light propagation with path-internal refraction.

use crate::graph::{Graph, NodeId, RelationId};
use crate::relation::RelationVocab;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Propagation parameters.
#[derive(Debug, Clone, Copy)]
pub struct PropagationParams {
    /// Maximum number of hops.
    pub max_depth: usize,
    /// Minimum mass to keep propagating (also prunes tiny contributions).
    pub min_intensity: f32,
}

impl Default for PropagationParams {
    fn default() -> Self {
        Self { max_depth: 4, min_intensity: 1e-3 }
    }
}

/// Frontier key: a node reached via an incoming relation (None on the first
/// hop of an untyped query).
type FrontierKey = (NodeId, Option<RelationId>);

/// Multi-seed diffusion. Equivalent to summing single-seed diffusions (linear).
pub fn propagate(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> HashMap<NodeId, f32> {
    seeds
        .par_iter()
        .map(|&(seed, mass)| propagate_single(graph, vocab, seed, mass, query_relation, params))
        .reduce(HashMap::new, |mut acc, m| {
            for (k, v) in m {
                *acc.entry(k).or_insert(0.0) += v;
            }
            acc
        })
}

/// Single-seed sparse diffusion. Returns total intensity per reached node
/// (including the seed itself).
pub fn propagate_single(
    graph: &Graph,
    vocab: &RelationVocab,
    seed: NodeId,
    initial_mass: f32,
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> HashMap<NodeId, f32> {
    let mut totals: HashMap<NodeId, f32> = HashMap::new();
    let n = graph.num_nodes();
    if (seed as usize) >= n || initial_mass < params.min_intensity {
        return totals;
    }

    *totals.entry(seed).or_insert(0.0) += initial_mass;

    let mut frontier: HashMap<FrontierKey, f32> = HashMap::new();
    frontier.insert((seed, query_relation), initial_mass);

    // Lazily cache each node's out-weight sum (keeps cost proportional to the
    // reached ball rather than the whole graph).
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();

    for _depth in 0..params.max_depth {
        if frontier.is_empty() {
            break;
        }
        let mut next: HashMap<FrontierKey, f32> = HashMap::new();

        for (&(u, r_in), &mass) in frontier.iter() {
            if mass < params.min_intensity {
                continue;
            }
            let u_idx = u as usize;
            let props = graph.node_props()[u_idx];
            let refl = props.reflection;
            if refl <= 0.0 {
                continue;
            }
            let rix = props.refraction_index;

            let denom = *denom_cache.entry(u).or_insert_with(|| {
                let mut s = 0.0f32;
                for (_v, ep) in graph.neighbors(u) {
                    s += (1.0 - ep.attenuation).max(0.0);
                }
                s
            });
            if denom <= 0.0 {
                continue;
            }

            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                if base <= 0.0 {
                    continue;
                }
                let p = base / denom;
                let sim = match r_in {
                    Some(a) => vocab.similarity(a, ep.relation),
                    None => 1.0,
                };
                let sim_term = if rix == 1.0 { sim } else { sim.powf(rix) };
                let transmitted = mass * refl * p * sim_term;
                if transmitted < params.min_intensity {
                    continue;
                }
                *totals.entry(v).or_insert(0.0) += transmitted;
                *next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;
            }
        }
        frontier = next;
    }

    totals
}

/// Convert intensity to a "light distance": d = -log(I + eps).
pub fn intensity_to_distance(intensities: &[f32], eps: f32) -> Vec<f32> {
    intensities
        .iter()
        .map(|&i| {
            let value = i + eps;
            if value > 0.0 { -value.ln() } else { f32::INFINITY }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::relation::RelationVocab;

    fn chain() -> Graph {
        // 0 -> 1 -> 2 -> 3, all relation 0, no attenuation.
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1)], vec![e(2)], vec![e(3)], vec![]];
        Graph::from_adjacency(4, adj, NodeProps::default()).unwrap()
    }

    #[test]
    fn chain_decays_by_reflection() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6 };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // reflection 0.85, p=1, sim=1 => geometric decay.
        assert!((t[&0] - 1.0).abs() < 1e-6);
        assert!((t[&1] - 0.85).abs() < 1e-5);
        assert!((t[&2] - 0.7225).abs() < 1e-5);
        assert!((t[&3] - 0.614125).abs() < 1e-5);
    }

    #[test]
    fn refraction_penalizes_relation_turn() {
        // 0 -(relA)-> 1 -(relB)-> 2 ; query relation = A.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let adj = vec![vec![ea], vec![eb], vec![]];
        let g = Graph::from_adjacency(3, adj, NodeProps::default()).unwrap();
        // sim(A,B) = 0.5
        let vocab = RelationVocab::new(
            vec!["A".into(), "B".into()],
            vec![1.0, 0.5, 0.5, 1.0],
        ).unwrap();
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6 };
        let refr = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // 1: 1*0.85*1*sim(A,A)=0.85 ; 2: 0.85*0.85*1*sim(A,B)=0.36125
        assert!((refr[&1] - 0.85).abs() < 1e-5);
        assert!((refr[&2] - 0.36125).abs() < 1e-5);

        // With uniform vocab (no refraction), node 2 gets the full 0.7225.
        let uni = RelationVocab::uniform(2);
        let plain = propagate_single(&g, &uni, 0, 1.0, Some(0), &params);
        assert!((plain[&2] - 0.7225).abs() < 1e-5);
        assert!(refr[&2] < plain[&2]);
    }

    #[test]
    fn sums_over_multiple_paths() {
        // 0 -> 1 -> 2 and 0 -> 2 (two paths to node 2).
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1), e(2)], vec![e(2)], vec![]];
        let g = Graph::from_adjacency(3, adj, NodeProps::default()).unwrap();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6 };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // node 0 has 2 out-edges => p=0.5 each.
        // direct 0->2: 1*0.85*0.5 = 0.425
        // via 1:   (0.425) then 1->2: 0.425*0.85*1 = 0.36125
        // total node 2 = 0.425 + 0.36125 = 0.78625
        assert!((t[&2] - 0.78625).abs() < 1e-5);
    }

    #[test]
    fn multi_seed_equals_sum_of_singles() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6 };
        let combined = propagate(&g, &vocab, &[(0, 1.0), (1, 1.0)], Some(0), &params);
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        let b = propagate_single(&g, &vocab, 1, 1.0, Some(0), &params);
        for node in 0..4u32 {
            let expected = a.get(&node).copied().unwrap_or(0.0)
                + b.get(&node).copied().unwrap_or(0.0);
            let got = combined.get(&node).copied().unwrap_or(0.0);
            assert!((got - expected).abs() < 1e-5, "node {node}");
        }
    }
}
```

- [ ] **Step 5: Rewrite `queries.rs` to the new API**

Replace the entire contents of `rgdb/src/queries.rs` with:

```rust
//! Query helpers over the sparse propagation kernel.

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate, intensity_to_distance, PropagationParams};
use crate::relation::RelationVocab;
use std::cmp::Ordering;

const INTENSITY_EPSILON: f32 = 1e-6;
const MIN_NORM: f32 = 1e-10;

#[derive(Debug, Clone)]
pub struct InfluenceResult {
    pub node: NodeId,
    pub intensity: f32,
    pub distance: f32,
}

/// Top-K nodes by influence, excluding the seed nodes themselves.
pub fn query_top_k_influence(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    k: usize,
    params: &PropagationParams,
) -> Vec<InfluenceResult> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let seed_set: std::collections::HashSet<NodeId> = seeds.iter().map(|&(n, _)| n).collect();

    let mut results: Vec<InfluenceResult> = totals
        .into_iter()
        .filter(|(node, _)| !seed_set.contains(node))
        .map(|(node, intensity)| InfluenceResult {
            node,
            intensity,
            distance: intensity_to_distance(&[intensity], INTENSITY_EPSILON)[0],
        })
        .collect();

    results.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(Ordering::Equal));
    results.truncate(k);
    results
}

/// Contextual distance from the seeds to a target (None if unreached).
pub fn query_distance(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    target: NodeId,
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> Option<f32> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let intensity = totals.get(&target).copied()?;
    if intensity <= params.min_intensity {
        return None;
    }
    Some(intensity_to_distance(&[intensity], INTENSITY_EPSILON)[0])
}

/// Hybrid query: fuse graph influence with cosine similarity over dense embeddings.
pub fn query_hybrid(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    embeddings: &[ndarray::Array1<f32>],
    query_embedding: &ndarray::Array1<f32>,
    alpha: f32,
    query_relation: Option<RelationId>,
    k: usize,
    params: &PropagationParams,
) -> Vec<InfluenceResult> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let norm_q = query_embedding.dot(query_embedding).sqrt();

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        let g = totals.get(&(i as NodeId)).copied().unwrap_or(0.0);
        let norm_e = emb.dot(emb).sqrt();
        let sim = if norm_q > MIN_NORM && norm_e > MIN_NORM {
            query_embedding.dot(emb) / (norm_q * norm_e)
        } else {
            0.0
        };
        scored.push((i, alpha * g + (1.0 - alpha) * sim));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(node, score)| InfluenceResult {
            node: node as NodeId,
            intensity: score,
            distance: intensity_to_distance(&[score], INTENSITY_EPSILON)[0],
        })
        .collect()
}
```

- [ ] **Step 6: Rewrite `embeddings.rs` to the new API**

Replace the entire contents of `rgdb/src/embeddings.rs` with:

```rust
//! Multi-source contextual embedding generation via propagation.

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate_single, PropagationParams};
use crate::relation::RelationVocab;
use ndarray::Array2;

/// One embedding column per (source, query_relation) pair, normalized to [0,1].
pub fn generate_embeddings(
    graph: &Graph,
    vocab: &RelationVocab,
    sources: &[NodeId],
    query_relations: &[Option<RelationId>],
    params: &PropagationParams,
) -> Array2<f32> {
    let num_nodes = graph.num_nodes();
    let num_sources = sources.len();
    let mut m = Array2::zeros((num_nodes, num_sources));

    for (i, (&src, &rel)) in sources.iter().zip(query_relations.iter()).enumerate() {
        let totals = propagate_single(graph, vocab, src, 1.0, rel, params);
        let max = totals.values().copied().fold(0.0f32, f32::max);
        if max > 0.0 {
            for (node, intensity) in &totals {
                m[[*node as usize, i]] = intensity / max;
            }
        }
    }
    m
}

/// Export a dense embedding matrix as CSV.
pub fn export_embeddings_csv(embeddings: &Array2<f32>, path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(path)?;
    let (num_nodes, dim) = embeddings.dim();
    for i in 0..num_nodes {
        for j in 0..dim {
            write!(file, "{:.6}", embeddings[[i, j]])?;
            if j < dim - 1 {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }
    Ok(())
}
```

- [ ] **Step 7: Fix `error.rs` (drop PVS, fix propagation error)**

In `rgdb/src/error.rs`: change the import line to remove `AngleBin`:

```rust
use crate::graph::{NodeId, RoomId};
```

Delete the entire `PVSError` enum (the `#[derive(Debug, Error)] pub enum PVSError { ... }` block). Replace the `PropagationError` enum with:

```rust
#[derive(Debug, Error)]
pub enum PropagationError {
    #[error("Invalid source node: {0}")]
    InvalidSourceNode(NodeId),
    #[error("Propagation failed: {0}")]
    PropagationFailed(String),
}
```

- [ ] **Step 8: Delete removed modules and update `lib.rs`**

Run:
```bash
git rm rgdb/src/pvs.rs rgdb/src/property_map.rs
```

Replace the entire contents of `rgdb/src/lib.rs` with:

```rust
pub mod graph;
pub mod propagation;
pub mod partitioning;
pub mod rooms;
pub mod level_file;      // re-enabled in Task 4
pub mod queries;
pub mod embeddings;
pub mod relation;
pub mod error;
pub mod rag;             // re-enabled in Task 3

#[cfg(feature = "llm")]
pub mod llm_integration;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use graph::*;
pub use propagation::*;
pub use partitioning::*;
pub use rooms::*;
pub use relation::*;
```

Then, to keep this task green while `rag` and `level_file` are being migrated in later tasks, temporarily comment out the three lines that will not compile yet and re-add them in their tasks:

```rust
// pub mod level_file;   // Task 4
// pub mod rag;          // Task 3
```

(Leave `queries`, `embeddings`, `relation`, etc. enabled.) Do **not** re-export `rag::*` here yet.

**Note:** `main.rs` and `benches` are separate compile targets; fix them in the next steps so `cargo build`/`cargo test` for the whole crate is green.

- [ ] **Step 9: Rewrite the `main.rs` demo**

Replace the entire contents of `rgdb/src/main.rs` with:

```rust
use rgdb::graph::{EdgeProps, Graph, NodeProps};
use rgdb::propagation::{propagate_single, PropagationParams};
use rgdb::relation::RelationVocab;

fn main() {
    // 0 -(causes)-> 1 -(causes)-> 2, with a relation-similarity matrix.
    let causes = 0u16;
    let edge = |dst, rel| (dst, EdgeProps { attenuation: 0.1, relation: rel, is_portal: false });
    let adj = vec![vec![edge(1, causes)], vec![edge(2, causes)], vec![]];
    let graph = Graph::from_adjacency(3, adj, NodeProps::default()).expect("graph");

    let vocab = RelationVocab::with_names_uniform(vec!["causes".into(), "isa".into()]);
    let params = PropagationParams::default();

    let totals = propagate_single(&graph, &vocab, 0, 1.0, Some(causes), &params);
    println!("Influence from node 0 (relation=causes):");
    for node in 0..graph.num_nodes() as u32 {
        println!("  node {node}: {:.6}", totals.get(&node).copied().unwrap_or(0.0));
    }
}
```

- [ ] **Step 10: Rewrite the benchmark**

Replace the entire contents of `rgdb/benches/directional_luminance.rs` with:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rgdb::graph::{EdgeProps, Graph, NodeProps, NodeId};
use rgdb::propagation::{propagate_single, PropagationParams};
use rgdb::relation::RelationVocab;

fn build(num_nodes: usize) -> Graph {
    let mut adj = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let mut ns = Vec::new();
        for j in 1..=5 {
            let dst = ((i + j) % num_nodes) as NodeId;
            if dst as usize != i {
                ns.push((dst, EdgeProps {
                    attenuation: 0.1,
                    relation: ((i + j) % 4) as u16,
                    is_portal: false,
                }));
            }
        }
        adj.push(ns);
    }
    Graph::from_adjacency(num_nodes, adj, NodeProps::default()).unwrap()
}

fn bench_propagation(c: &mut Criterion) {
    let graph = build(1000);
    let vocab = RelationVocab::uniform(4);
    let params = PropagationParams::default();
    c.bench_function("propagate_single_1000", |b| {
        b.iter(|| {
            let t = propagate_single(black_box(&graph), &vocab, 0, 1.0, Some(0), &params);
            black_box(t.len());
        });
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
```

- [ ] **Step 11: Build and run the core test suite (rag/level_file disabled)**

Run: `cargo test -p rgdb 2>&1 | tail -20`
Expected: compiles; `test result: ok` for all currently-enabled modules, including the 4 new `propagation::tests` and the 3 `relation::tests`. (`rag` and `level_file` tests are absent because those modules are temporarily disabled.)

- [ ] **Step 12: Confirm the bench and main build**

Run:
```bash
cargo build -p rgdb --benches 2>&1 | tail -3
cargo run -p rgdb 2>&1 | tail -6
```
Expected: `Finished`; the demo prints influence for nodes 0/1/2 with node 0 = 1.0 and decreasing values.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat(rgdb)!: sparse typed-PPR kernel, RelationId props, drop PVS/property_map

BREAKING CHANGE: NodeProps/EdgeProps reshaped; propagation API replaced;
rag and level_file temporarily disabled pending their migration tasks."
```

---

### Task 3: Migrate the RAG layer

**Files:**
- Rewrite: `rgdb/src/rag/query_engine.rs` (new propagation calls, fusion cleanup, vocab field)
- Rewrite: `rgdb/src/rag/intent.rs` (resolve to relation names, drop angle-bin methods)
- Modify: `rgdb/src/rag/personalization.rs` (drop `N_ANGLE_BINS` affinity machinery; keep access/interaction/session boosts)
- Modify: `rgdb/src/rag/mod.rs` (exports)
- Modify: `rgdb/src/lib.rs` (re-enable `pub mod rag;` + re-export)

**Interfaces:**
- Consumes: `propagate`, `RelationVocab`, `PropagationParams`, `EmbeddingStore` (unchanged).
- Produces: `RAGQueryEngine::new(graph: Graph, vocab: RelationVocab, embedding_store: EmbeddingStore)`; `query(query_text, query_embedding, user_context, config) -> Vec<QueryResult>`.

**Background:** `EmbeddingStore` (`rag/embedding_store.rs`) has no angle-bin coupling and is unchanged. The fusion drops max-normalization because graph scores are calibrated: seed masses are normalized to sum to 1, so graph intensity is a proper distribution.

- [ ] **Step 1: Simplify `personalization.rs`**

In `rgdb/src/rag/personalization.rs`:
- Change the import to `use crate::graph::{NodeId, RoomId};` (drop `N_ANGLE_BINS`).
- Remove the `topic_affinities: [f32; N_ANGLE_BINS]` field from `UserContext` and its initializer in `new` (remove that line).
- Delete the methods `set_topic_affinity`, `modulate_luminance`, `preferred_bins` (all angle-bin based).
- Keep `record_interaction`, `add_accessible_room`, `can_access_room`, `compute_boost`, `get_feed_seeds`, and the interaction/session fields.
- If any test in that file references the removed items, delete those test cases.

- [ ] **Step 2: Rewrite `intent.rs` to produce relation names**

In `rgdb/src/rag/intent.rs`:
- Delete `to_angle_bin`, `related_bins`, `to_relationship_property` (they reference `AngleBin`/`RelationshipProperty`, both removed).
- Add:

```rust
impl QueryIntent {
    /// Canonical relation name for this intent (looked up in the graph's vocab).
    pub fn relation_name(&self) -> &'static str {
        match self {
            Self::Definition => "IsA",
            Self::Requirements => "Requires",
            Self::Association => "RelatedTo",
            Self::Causation => "Causes",
            Self::Composition => "Contains",
            Self::Membership => "PartOf",
            Self::Similarity => "SimilarTo",
            Self::Contrast => "OppositeOf",
            Self::Capability => "Enables",
            Self::Conflict => "ConflictsWith",
            Self::MultiHop => "RelatedTo",
        }
    }
}
```
- Keep the `IntentClassifier` keyword logic unchanged. Delete the `test_angle_bin_mapping` and `test_related_bins` tests; keep the classification tests.

- [ ] **Step 3: Rewrite `query_engine.rs`**

Replace the entire contents of `rgdb/src/rag/query_engine.rs` with:

```rust
//! RAG query engine: seeds from vector search, typed-PPR diffusion, calibrated fusion.

use crate::graph::{Graph, NodeId};
use crate::propagation::{propagate, intensity_to_distance, PropagationParams};
use crate::relation::RelationVocab;
use crate::queries::InfluenceResult;

use super::embedding_store::EmbeddingStore;
use super::intent::IntentClassifier;
use super::personalization::UserContext;

use ndarray::Array1;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct QueryConfig {
    pub top_k: usize,
    pub max_hops: usize,
    pub alpha: f32,          // graph weight
    pub beta: f32,           // vector weight
    pub gamma: f32,          // personalization weight
    pub min_relevance: f32,
    pub num_sources: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self { top_k: 10, max_hops: 4, alpha: 0.5, beta: 0.4, gamma: 0.1,
               min_relevance: 1e-3, num_sources: 5 }
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub node_id: NodeId,
    pub score: f32,
    pub graph_intensity: f32,
    pub vector_similarity: f32,
    pub personalization_boost: f32,
    pub distance: f32,
}

pub struct RAGQueryEngine {
    graph: Graph,
    vocab: RelationVocab,
    embedding_store: EmbeddingStore,
    intent_classifier: IntentClassifier,
}

impl RAGQueryEngine {
    pub fn new(graph: Graph, vocab: RelationVocab, embedding_store: EmbeddingStore) -> Self {
        Self { graph, vocab, embedding_store, intent_classifier: IntentClassifier::new() }
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    pub fn embedding_store(&self) -> &EmbeddingStore { &self.embedding_store }

    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &Array1<f32>,
        user_context: Option<&UserContext>,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        // 1. Intent -> relation id (None if the vocab lacks it -> untyped query).
        let intent = self.intent_classifier.classify(query_text);
        let query_relation = self.vocab.id_of(intent.relation_name());

        // 2. Vector seeds, weights normalized to sum to 1 (calibrated graph mass).
        let cands = self.embedding_store.top_k_similar(query_embedding, config.num_sources);
        if cands.is_empty() {
            return Vec::new();
        }
        let sum: f32 = cands.iter().map(|c| c.similarity.max(0.0)).sum();
        let seeds: Vec<(NodeId, f32)> = if sum > 0.0 {
            cands.iter().map(|c| (c.node_id, c.similarity.max(0.0) / sum)).collect()
        } else {
            cands.iter().map(|c| (c.node_id, 1.0 / cands.len() as f32)).collect()
        };

        // 3. Diffuse.
        let params = PropagationParams { max_depth: config.max_hops, min_intensity: config.min_relevance };
        let totals = propagate(&self.graph, &self.vocab, &seeds, query_relation, &params);

        // 4. Vector similarities for all nodes (dense; ANN is future work).
        let sims = self.embedding_store.all_similarities(query_embedding);

        // 5. Fuse (no max-normalization; graph mass is already calibrated).
        let mut results: Vec<QueryResult> = Vec::new();
        for node in 0..self.graph.num_nodes() {
            let g = totals.get(&(node as NodeId)).copied().unwrap_or(0.0);
            let v = sims.get(node).copied().unwrap_or(0.0);
            if g < config.min_relevance && v < config.min_relevance {
                continue;
            }
            let p_boost = match user_context {
                Some(ctx) => ctx.compute_boost(node as NodeId, self.graph.room_map().get(node).copied()),
                None => 1.0,
            };
            if p_boost <= 0.0 {
                continue;
            }
            let base = config.alpha * g + config.beta * v;
            let score = base * (1.0 + config.gamma * (p_boost - 1.0));
            results.push(QueryResult {
                node_id: node as NodeId,
                score,
                graph_intensity: g,
                vector_similarity: v,
                personalization_boost: p_boost,
                distance: intensity_to_distance(&[score], 1e-6)[0],
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(config.top_k);
        results
    }

    pub fn to_influence_results(results: Vec<QueryResult>) -> Vec<InfluenceResult> {
        results.into_iter().map(|r| InfluenceResult {
            node: r.node_id, intensity: r.score, distance: r.distance,
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::RelationVocab;
    use ndarray::Array2;

    fn engine() -> RAGQueryEngine {
        let graph = Graph::new(5).unwrap();
        let emb = Array2::from_shape_vec(
            (5, 4),
            vec![1.0,0.0,0.0,0.0, 0.8,0.2,0.0,0.0, 0.0,1.0,0.0,0.0,
                 0.0,0.0,1.0,0.0, 0.5,0.5,0.0,0.0],
        ).unwrap();
        RAGQueryEngine::new(graph, RelationVocab::uniform(1), EmbeddingStore::new(emb))
    }

    #[test]
    fn engine_builds_and_queries() {
        let e = engine();
        assert_eq!(e.graph().num_nodes(), 5);
        let q = ndarray::arr1(&[1.0, 0.0, 0.0, 0.0]);
        let res = e.query("what is x?", &q, None, &QueryConfig::default());
        assert!(res.len() <= QueryConfig::default().top_k);
    }
}
```

- [ ] **Step 4: Update `rag/mod.rs` exports**

Open `rgdb/src/rag/mod.rs`. Ensure it re-exports the surviving items and nothing removed. It should read:

```rust
pub mod embedding_store;
pub mod intent;
pub mod personalization;
pub mod query_engine;

pub use embedding_store::EmbeddingStore;
pub use intent::{IntentClassifier, QueryIntent};
pub use personalization::UserContext;
pub use query_engine::{QueryConfig, QueryResult, RAGQueryEngine};
```

- [ ] **Step 5: Re-enable `rag` in `lib.rs`**

In `rgdb/src/lib.rs`, uncomment / restore:

```rust
pub mod rag;
```
and add the re-export line back:

```rust
pub use rag::{RAGQueryEngine, QueryConfig, QueryResult, EmbeddingStore, IntentClassifier, QueryIntent, UserContext};
```

- [ ] **Step 6: Build and test**

Run: `cargo test -p rgdb 2>&1 | tail -20`
Expected: green, including `rag::query_engine::tests::engine_builds_and_queries` and the retained intent/personalization/embedding_store tests.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(rgdb)!: migrate RAG engine to typed-PPR + calibrated fusion"
```

---

### Task 4: Level file format v2

**Files:**
- Rewrite: `rgdb/src/level_file.rs`
- Modify: `rgdb/src/lib.rs` (re-enable `pub mod level_file;`)

**Interfaces:**
- Produces:
  - `write_level_file(graph: &Graph, rooms: &RoomCollection, vocab: &RelationVocab, path: &str) -> Result<(), LevelFileError>`
  - `read_level_file_mmap(path: &str) -> Result<(Graph, RoomCollection, RelationVocab), LevelFileError>`
  - `VERSION = 2`.

**Background:** Drops the PVS, angle-table, and directional-luminance sections. Node section is 2×f32 (`reflection`, `refraction_index`). Edge props are `attenuation:f32` + `relation:u16` + `is_portal:u8` + 1 pad byte (8-byte aligned). A new relation-vocab section stores names + the flattened similarity matrix.

- [ ] **Step 1: Rewrite `level_file.rs`**

Replace the entire contents of `rgdb/src/level_file.rs` with:

```rust
//! Level file format v2: header, nodes, CSR edges, room map, rooms, relation vocab.

use crate::graph::{Graph, NodeProps, EdgeProps, RelationId};
use crate::rooms::RoomCollection;
use crate::relation::RelationVocab;
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"RGDB";
const VERSION: u32 = 2;
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum LevelFileError {
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Structure mismatch: {0}")]
    StructureMismatch(String),
}

fn write_node<W: Write>(w: &mut W, p: &NodeProps) -> std::io::Result<()> {
    w.write_all(&p.reflection.to_le_bytes())?;
    w.write_all(&p.refraction_index.to_le_bytes())?;
    Ok(())
}

fn read_node<R: Read>(r: &mut R) -> std::io::Result<NodeProps> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    let reflection = f32::from_le_bytes(b);
    r.read_exact(&mut b)?;
    let refraction_index = f32::from_le_bytes(b);
    Ok(NodeProps { reflection, refraction_index })
}

fn write_edge<W: Write>(w: &mut W, e: &EdgeProps) -> std::io::Result<()> {
    w.write_all(&e.attenuation.to_le_bytes())?;
    w.write_u16::<LittleEndian>(e.relation)?;
    w.write_u8(if e.is_portal { 1 } else { 0 })?;
    w.write_u8(0)?; // pad to 8 bytes
    Ok(())
}

fn read_edge<R: Read>(r: &mut R) -> std::io::Result<EdgeProps> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    let attenuation = f32::from_le_bytes(b);
    let relation = r.read_u16::<LittleEndian>()?;
    let is_portal = r.read_u8()? != 0;
    let _pad = r.read_u8()?;
    Ok(EdgeProps { attenuation, relation, is_portal })
}

/// Write graph + rooms + relation vocab to a v2 level file.
pub fn write_level_file(
    graph: &Graph,
    rooms: &RoomCollection,
    vocab: &RelationVocab,
    path: &str,
) -> Result<(), LevelFileError> {
    let mut file = File::create(path)?;

    // Header: magic, version, num_nodes, num_edges (all u32) = 16 bytes.
    file.write_all(MAGIC)?;
    file.write_u32::<LittleEndian>(VERSION)?;
    file.write_u32::<LittleEndian>(graph.num_nodes() as u32)?;
    file.write_u32::<LittleEndian>(graph.num_edges() as u32)?;

    // Nodes.
    for p in graph.node_props() {
        write_node(&mut file, p)?;
    }
    // CSR: row_ptr (u32), col_idx (u32), edge_props (8 bytes).
    for &ptr in graph.row_ptr() {
        file.write_u32::<LittleEndian>(ptr as u32)?;
    }
    for &c in graph.col_idx() {
        file.write_u32::<LittleEndian>(c)?;
    }
    for e in graph.edge_props() {
        write_edge(&mut file, e)?;
    }
    // Room map.
    file.write_u32::<LittleEndian>(graph.room_map().len() as u32)?;
    for &r in graph.room_map() {
        file.write_u32::<LittleEndian>(r)?;
    }
    // Rooms.
    file.write_u32::<LittleEndian>(rooms.num_rooms() as u32)?;
    for room in &rooms.rooms {
        file.write_u32::<LittleEndian>(room.id)?;
        file.write_u32::<LittleEndian>(room.node_start as u32)?;
        file.write_u32::<LittleEndian>(room.node_count as u32)?;
    }
    // Relation vocab: count, then each name (len-prefixed utf8), then n*n f32 matrix.
    let n = vocab.len();
    file.write_u32::<LittleEndian>(n as u32)?;
    for i in 0..n {
        let name = vocab.name(i as RelationId).unwrap_or("");
        let bytes = name.as_bytes();
        file.write_u32::<LittleEndian>(bytes.len() as u32)?;
        file.write_all(bytes)?;
    }
    for a in 0..n {
        for b in 0..n {
            file.write_all(&vocab.similarity(a as RelationId, b as RelationId).to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read a v2 level file (streamed over an mmap).
pub fn read_level_file_mmap(
    path: &str,
) -> Result<(Graph, RoomCollection, RelationVocab), LevelFileError> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size < 16 {
        return Err(LevelFileError::StructureMismatch("file too small".into()));
    }
    if size > MAX_FILE_SIZE {
        return Err(LevelFileError::StructureMismatch("file too large".into()));
    }
    let mmap = unsafe { MmapOptions::new().len(size as usize).map(&file)? };
    let mut cur = &mmap[..];

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LevelFileError::InvalidMagic);
    }
    let version = cur.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(LevelFileError::UnsupportedVersion(version));
    }
    let num_nodes = cur.read_u32::<LittleEndian>()? as usize;
    let num_edges = cur.read_u32::<LittleEndian>()? as usize;

    let mut node_props = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        node_props.push(read_node(&mut cur)?);
    }
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    for _ in 0..=num_nodes {
        row_ptr.push(cur.read_u32::<LittleEndian>()? as usize);
    }
    let mut col_idx = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        col_idx.push(cur.read_u32::<LittleEndian>()?);
    }
    let mut edge_props = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        edge_props.push(read_edge(&mut cur)?);
    }
    let room_map_len = cur.read_u32::<LittleEndian>()? as usize;
    let mut room_map = Vec::with_capacity(room_map_len);
    for _ in 0..room_map_len {
        room_map.push(cur.read_u32::<LittleEndian>()?);
    }
    let graph = Graph::from_csr(num_nodes, row_ptr, col_idx, node_props, edge_props, room_map)
        .map_err(|e| LevelFileError::StructureMismatch(format!("{e}")))?;

    let num_rooms = cur.read_u32::<LittleEndian>()? as usize;
    let mut rooms_vec = Vec::with_capacity(num_rooms);
    for _ in 0..num_rooms {
        let id = cur.read_u32::<LittleEndian>()?;
        let node_start = cur.read_u32::<LittleEndian>()? as usize;
        let node_count = cur.read_u32::<LittleEndian>()? as usize;
        rooms_vec.push(crate::rooms::Room { id, node_start, node_count });
    }
    let max_room_id = rooms_vec.iter().map(|r| r.id).max().unwrap_or(0) as usize + 1;
    let mut room_index = vec![usize::MAX; max_room_id];
    for (idx, room) in rooms_vec.iter().enumerate() {
        room_index[room.id as usize] = idx;
    }
    let rooms = RoomCollection { rooms: rooms_vec, room_index };

    let n_rel = cur.read_u32::<LittleEndian>()? as usize;
    let mut names = Vec::with_capacity(n_rel);
    for _ in 0..n_rel {
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        cur.read_exact(&mut buf)?;
        names.push(String::from_utf8_lossy(&buf).into_owned());
    }
    let mut sim = Vec::with_capacity(n_rel * n_rel);
    for _ in 0..(n_rel * n_rel) {
        sim.push(cur.read_f32::<LittleEndian>()?);
    }
    let vocab = RelationVocab::new(names, sim)
        .map_err(|e| LevelFileError::StructureMismatch(format!("{e}")))?;

    Ok((graph, rooms, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::rooms::RoomCollection;
    use crate::relation::RelationVocab;
    use std::fs;

    #[test]
    fn roundtrip_v2() {
        let e = (1u32, EdgeProps { attenuation: 0.25, relation: 1, is_portal: false });
        let adj = vec![vec![e], vec![]];
        let mut graph = Graph::from_adjacency(2, adj, NodeProps::default()).unwrap();
        graph.set_room(0, 0).unwrap();
        graph.set_room(1, 1).unwrap();
        graph.set_node_props(0, NodeProps { reflection: 0.5, refraction_index: 2.0 }).unwrap();

        let rooms = RoomCollection::from_room_map(graph.room_map());
        let vocab = RelationVocab::new(
            vec!["isa".into(), "causes".into()],
            vec![1.0, 0.3, 0.3, 1.0],
        ).unwrap();

        let path = "test_level_v2.rgdb";
        write_level_file(&graph, &rooms, &vocab, path).unwrap();
        let (g2, r2, v2) = read_level_file_mmap(path).unwrap();

        assert_eq!(g2.num_nodes(), 2);
        assert_eq!(g2.num_edges(), 1);
        assert!((g2.node_props()[0].reflection - 0.5).abs() < 1e-6);
        assert_eq!(g2.edge_props()[0].relation, 1);
        assert_eq!(r2.num_rooms(), rooms.num_rooms());
        assert_eq!(v2.id_of("causes"), Some(1));
        assert!((v2.similarity(0, 1) - 0.3).abs() < 1e-6);

        let _ = fs::remove_file(path);
    }
}
```

- [ ] **Step 2: Re-enable `level_file` in `lib.rs`**

In `rgdb/src/lib.rs`, restore:

```rust
pub mod level_file;
```

- [ ] **Step 3: Build and test**

Run: `cargo test -p rgdb level_file 2>&1 | tail -8`
Expected: `roundtrip_v2` passes.

- [ ] **Step 4: Full suite**

Run: `cargo test -p rgdb 2>&1 | tail -10`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(rgdb)!: level file v2 (relation vocab section, drop PVS/angle/directional)"
```

---

### Task 5: Feature-gate CUDA off (mark stale)

**Files:**
- Modify: `rgdb/src/cuda/mod.rs`

**Interfaces:**
- Produces: a clear compile error when `--features cuda` is used (the kernels target the removed dense angle-bin model), without affecting the default build.

**Background:** CUDA is already `#[cfg(feature = "cuda")]` and off by default, so the default build and CI are unaffected. This task makes the staleness explicit instead of emitting confusing type errors.

- [ ] **Step 1: Add a fail-fast staleness guard**

At the very top of `rgdb/src/cuda/mod.rs` (before any other item), add:

```rust
compile_error!(
    "The CUDA backend targets the pre-rewrite dense angle-bin kernel and has not \
     been ported to the sparse typed-PPR model. It is intentionally disabled. \
     Track the GPU port in a separate effort; do not build with --features cuda."
);
```

- [ ] **Step 2: Confirm the default build is unaffected**

Run: `cargo build -p rgdb 2>&1 | tail -3`
Expected: `Finished` (no `cuda` feature → guard not compiled).

- [ ] **Step 3: Confirm the cuda feature fails fast with the message**

Run: `cargo build -p rgdb --features cuda 2>&1 | grep -A1 "CUDA backend" | head -3`
Expected: the `compile_error!` message text appears (the build fails, as intended).

- [ ] **Step 4: Commit**

```bash
git add rgdb/src/cuda/mod.rs
git commit -m "chore(rgdb): mark CUDA backend stale (fail fast under --features cuda)"
```

---

### Task 6: Update the Python bindings

**Files:**
- Rewrite: `rgdb-python/src/lib.rs`
- Modify: `rgdb-eval/scripts/smoke_bindings.py` (new call shape)

**Interfaces:**
- Produces (Python):
  - `build_graph(num_nodes, adjacency=[(dst, attenuation, relation)], node_reflections=None, node_refractions=None) -> RgdbGraph`
  - `uniform_vocab(n) -> RelationVocab`; `vocab_from_matrix(names, flat_similarity) -> RelationVocab`
  - `propagate(graph, vocab, seeds=[(node, mass)], query_relation=None, max_depth=4, min_intensity=1e-3) -> list[(node, intensity)]`

**Background:** The eval harness (sub-project C, Task 6/10) consumes these. The new `propagate` returns sparse `(node, intensity)` pairs.

- [ ] **Step 1: Rewrite the bindings**

Replace the entire contents of `rgdb-python/src/lib.rs` with:

```rust
//! PyO3 bindings for the sparse typed-PPR RGDB core.

use pyo3::prelude::*;
use rgdb::graph::{Graph, NodeProps, EdgeProps, NodeId, RelationId};
use rgdb::relation::RelationVocab;
use rgdb::propagation::{propagate as rust_propagate, PropagationParams};

#[pyclass(name = "RgdbGraph")]
struct PyGraph { inner: Graph }

#[pymethods]
impl PyGraph {
    #[getter]
    fn num_nodes(&self) -> usize { self.inner.num_nodes() }
    #[getter]
    fn num_edges(&self) -> usize { self.inner.num_edges() }
}

#[pyclass(name = "RelationVocab")]
#[derive(Clone)]
struct PyVocab { inner: RelationVocab }

#[pyfunction]
fn uniform_vocab(n: usize) -> PyVocab {
    PyVocab { inner: RelationVocab::uniform(n) }
}

#[pyfunction]
fn vocab_from_matrix(names: Vec<String>, flat_similarity: Vec<f32>) -> PyResult<PyVocab> {
    RelationVocab::new(names, flat_similarity)
        .map(|inner| PyVocab { inner })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
#[pyo3(signature = (num_nodes, adjacency, node_reflections=None, node_refractions=None))]
fn build_graph(
    num_nodes: usize,
    adjacency: Vec<Vec<(u32, f32, u16)>>,
    node_reflections: Option<Vec<f32>>,
    node_refractions: Option<Vec<f32>>,
) -> PyResult<PyGraph> {
    let adj: Vec<Vec<(NodeId, EdgeProps)>> = adjacency
        .into_iter()
        .map(|ns| ns.into_iter().map(|(dst, attenuation, relation)| {
            (dst, EdgeProps { attenuation, relation: relation as RelationId, is_portal: false })
        }).collect())
        .collect();

    let mut graph = Graph::from_adjacency(num_nodes, adj, NodeProps::default())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;

    for i in 0..num_nodes {
        let mut p = NodeProps::default();
        let mut changed = false;
        if let Some(ref v) = node_reflections {
            if i < v.len() { p.reflection = v[i]; changed = true; }
        }
        if let Some(ref v) = node_refractions {
            if i < v.len() { p.refraction_index = v[i]; changed = true; }
        }
        if changed {
            let _ = graph.set_node_props(i as NodeId, p);
        }
    }
    Ok(PyGraph { inner: graph })
}

#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3))]
fn propagate(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
) -> Vec<(u32, f32)> {
    let params = PropagationParams { max_depth, min_intensity };
    let totals = rust_propagate(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    totals.into_iter().collect()
}

#[pymodule]
fn _rgdb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyVocab>()?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(vocab_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Update the smoke script**

Replace the entire contents of `rgdb-eval/scripts/smoke_bindings.py` with:

```python
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
```

- [ ] **Step 3: Rebuild and run the smoke script**

Run:
```bash
maturin develop -m rgdb-python/Cargo.toml
python rgdb-eval/scripts/smoke_bindings.py
```
Expected: `bindings smoke ok: [(0, 1.0), (1, 0.85...), (2, 0.72...)]`.

- [ ] **Step 4: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/scripts/smoke_bindings.py
git commit -m "feat(bindings)!: expose sparse typed-PPR propagate + RelationVocab"
```

---

### Task 7: Version bumps

**Files:**
- Modify: `rgdb/Cargo.toml`, `rgdb-python/Cargo.toml`, `rgdb-python/pyproject.toml`

**Interfaces:** none (metadata only).

- [ ] **Step 1: Bump versions**

In `rgdb/Cargo.toml` change `version = "0.1.0"` → `version = "0.2.0"`.
In `rgdb-python/Cargo.toml` change `version = "0.1.0"` → `version = "0.2.0"`.
In `rgdb-python/pyproject.toml` change `version = "0.1.0"` → `version = "0.2.0"`.

- [ ] **Step 2: Verify the workspace still builds**

Run: `cargo build --workspace 2>&1 | tail -3`
Expected: `Finished`.

- [ ] **Step 3: Commit**

```bash
git add rgdb/Cargo.toml rgdb-python/Cargo.toml rgdb-python/pyproject.toml
git commit -m "chore: bump rgdb + rgdb-python to 0.2.0 (breaking core rewrite)"
```

---

### Task 8: New-RGDB eval contender + comparison run

**Files:**
- Create: `rgdb-eval/rgdb_eval/rankers/rgdb_new.py`
- Create: `rgdb-eval/tests/test_rgdb_new.py`
- Modify: `rgdb-eval/rgdb_eval/run.py` (add new contenders + ablation)
- Create: `rgdb-eval/results/rewrite-metaqa.md` (generated)

**Interfaces:**
- Consumes: rewritten bindings (`build_graph`, `uniform_vocab`, `vocab_from_matrix`, `propagate`), `TypedGraph`, relation-name embeddings from `embed_texts`.
- Produces:
  - `NewRgdbRanker(graph, vocab_mode="refraction"|"uniform")` behind the shared `Ranker` protocol.

**Background:** This is spec contender #5 and its ablation. `"refraction"` builds the vocab from cosine similarity of embedded relation names; `"uniform"` uses all-ones (typed PPR, no refraction). Comparing the two isolates whether path-internal refraction pays rent.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_rgdb_new.py`:

```python
import pytest
from rgdb_eval import TypedGraph

core = pytest.importorskip("rgdb_embeddings._rgdb_core")
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker


def line_graph():
    return TypedGraph(
        num_nodes=4, entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"], edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_new_rgdb_excludes_seed_and_orders_by_distance():
    r = NewRgdbRanker(line_graph(), vocab_mode="uniform")
    ranked = r.rank(seeds=[0], query_relation="r", k=4)
    assert 0 not in ranked            # seed excluded
    assert ranked.index(1) < ranked.index(3)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_rgdb_new.py -v`
Expected: FAIL (`ModuleNotFoundError: rgdb_eval.rankers.rgdb_new`).

- [ ] **Step 3: Implement the contender**

Create `rgdb-eval/rgdb_eval/rankers/rgdb_new.py`:

```python
"""Post-rewrite RGDB contender: sparse typed-PPR with optional refraction."""
from __future__ import annotations
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from ..dataset import TypedGraph
from ..embeddings import embed_texts


class NewRgdbRanker:
    def __init__(self, graph: TypedGraph, vocab_mode: str = "refraction"):
        if vocab_mode not in ("refraction", "uniform"):
            raise ValueError(vocab_mode)
        self.graph = graph
        self.vocab_mode = vocab_mode
        self.name = f"rgdb-new-{vocab_mode}"

        adj = [[] for _ in range(graph.num_nodes)]
        for (s, d, rel_id) in graph.edges:
            adj[s].append((d, 0.0, rel_id))
        self._g = core.build_graph(graph.num_nodes, adj)

        n = len(graph.relations)
        if vocab_mode == "uniform":
            self._vocab = core.uniform_vocab(n)
        else:
            rel_vecs = embed_texts(list(graph.relations))
            sim = (rel_vecs @ rel_vecs.T).clip(0.0, 1.0).astype(np.float32)
            self._vocab = core.vocab_from_matrix(list(graph.relations), sim.ravel().tolist())

    def rank(self, seeds, query_relation, k):
        if not seeds:
            return []
        rel_id = self.graph.relation_to_id.get(query_relation) if query_relation else None
        seed_pairs = [(int(s), 1.0 / len(seeds)) for s in seeds]
        totals = dict(core.propagate(self._g, self._vocab, seed_pairs, rel_id, 4, 1e-4))
        seed_set = set(seeds)
        items = [(node, sc) for node, sc in totals.items() if node not in seed_set]
        items.sort(key=lambda x: -x[1])
        return [int(n) for n, _ in items[:k]]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_rgdb_new.py -v`
Expected: 1 passed.

- [ ] **Step 5: Add both new contenders to the runner**

In `rgdb-eval/rgdb_eval/run.py`, inside `build_rankers`, replace the `try/except` that appends `CurrentRgdbRanker` with:

```python
    try:
        from .rankers.rgdb_current import CurrentRgdbRanker
        rankers.append(CurrentRgdbRanker(graph))
    except Exception as exc:
        print(f"[warn] rgdb-current contender skipped: {exc}")
    try:
        from .rankers.rgdb_new import NewRgdbRanker
        rankers.append(NewRgdbRanker(graph, vocab_mode="uniform"))
        rankers.append(NewRgdbRanker(graph, vocab_mode="refraction"))
    except Exception as exc:
        print(f"[warn] rgdb-new contenders skipped: {exc}")
```

- [ ] **Step 6: Rebuild bindings and run the full comparison**

Run:
```bash
maturin develop -m rgdb-python/Cargo.toml
cd rgdb-eval && python -m rgdb_eval.run --data data/MetaQA --limit 1000 --out results/rewrite-metaqa.md && cd ..
```
Expected: `wrote results/rewrite-metaqa.md`; no `[warn] rgdb-new contenders skipped`. The table has rows including `rgdb-new-uniform` and `rgdb-new-refraction` per hop.

- [ ] **Step 7: Record the verdict**

Open `rgdb-eval/results/rewrite-metaqa.md` and append a short "Verdict" paragraph stating, per the spec's decision rule, whether `rgdb-new-refraction` beats `untyped-ppr` **and** `rgdb-new-uniform` on 2-hop and 3-hop Hits@10/MRR. Keep it factual (cite the numbers from the table).

- [ ] **Step 8: Commit**

```bash
git add rgdb-eval/rgdb_eval/rankers/rgdb_new.py rgdb-eval/tests/test_rgdb_new.py rgdb-eval/rgdb_eval/run.py rgdb-eval/results/rewrite-metaqa.md
git commit -m "eval: add new-RGDB contender + no-refraction ablation and comparison run"
```

---

## Self-Review

**Spec coverage (sub-project B section):**
- B1 kernel (state `(node, incoming_relation)`, query relation as initial r_in, transition formula, boundedness, sum-over-paths accumulation, termination, seeding) → Task 2 kernel + tests. ✓
- B2 data structures (`NodeProps` 2 fields, `EdgeProps` with `relation`, `RelationVocab`, three population modes incl. all-ones ablation, `intent.rs` → relation name) → Tasks 1, 2, 3, 8. ✓
- B3 sparse frontier + rayon across seeds; deferred within-diffusion parallelism; hand-computed unit tests → Task 2 (`propagate` par_iter, lazy `denom_cache`, tests). ✓
- B4 level file v2 + relation section + drops; rooms kept, `pvs.rs` deleted; bindings updated; CUDA gated off; fusion cleanup (no max-normalization) → Tasks 4, 2 (pvs delete), 6, 5, 3. ✓
- Version bumps (crate 0.2.0, file VERSION 2) → Tasks 7, 4. ✓
- Contender #5 + ablation + comparison table → Task 8. ✓
- Success criteria: sparse (Task 2 maps + lazy denom), deterministic-up-to-fp / order-independent (kernel sums, tested), data-supported refraction verdict (Task 8 step 7). ✓
- Non-goals respected: no GPU port (Task 5 gates off), no REST API, no incremental updates, no within-diffusion parallelism. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" in code. Every rewrite step shows full file or exact edit. Task 8 step 7 asks for a factual verdict paragraph from real numbers — an analysis deliverable, not a code placeholder. ✓

**Type consistency (checked across tasks):**
- `RelationId = u16` defined in Task 1 (locally), moved to `graph.rs` in Task 2; `relation.rs` switches to the import in Task 2 step 3. ✓
- `NodeProps { reflection, refraction_index }` / `EdgeProps { attenuation, relation, is_portal }` identical in graph.rs (Task 2), level_file.rs (Task 4), bindings (Task 6). ✓
- `propagate(graph, vocab, seeds: &[(NodeId,f32)], query_relation: Option<RelationId>, params) -> HashMap<NodeId,f32>` identical in propagation.rs (Task 2), queries.rs (Task 2), embeddings.rs (Task 2 uses `propagate_single`), query_engine.rs (Task 3), bindings (Task 6). ✓
- `RelationVocab::{new, uniform, with_names_uniform, id_of, name, similarity, len}` used consistently in relation.rs (Task 1), propagation tests (Task 2), query_engine (Task 3), level_file (Task 4), bindings (Task 6). ✓
- Eval `Ranker.rank(seeds, query_relation, k) -> list[int]`: `NewRgdbRanker` (Task 8) matches the protocol from sub-project C Task 4. ✓
- `write_level_file`/`read_level_file_mmap` signatures change from `&PVS` to `&RelationVocab` / return `RelationVocab` — no remaining caller passes a `PVS` (pvs deleted in Task 2; `main.rs` rewritten in Task 2 does not call level_file). ✓
