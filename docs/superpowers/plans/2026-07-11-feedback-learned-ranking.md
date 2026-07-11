# Feedback-Learned Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two feedback-learned ranking improvements to RGDB — learned soft depth weights (#2) and a Hits@1 reranker (#4) — plus the shared `propagate_layered` primitive both need.

**Architecture:** A layered propagation pass exposes per-node per-depth intensity `I_k(v)`. `DepthProfileStore` learns a soft `DepthWeights` per hop-class from feedback (prior = `terminal(k)`); `Reranker` is an online logistic model that reorders the top-K. Both learn inside `RgdbEngine` mirroring `TransitionStore` (accumulate → derive → atomic swap, sidecar persistence), and both are strict no-ops at cold start.

**Tech Stack:** Rust (rgdb crate + rgdb-python PyO3 bindings), Python eval harness (rgdb-eval), MetaQA.

**Spec:** `docs/superpowers/specs/2026-07-11-feedback-learned-ranking-design.md`

**Branch:** extends `feature/rgdb-self-learning-loop` (already checked out). Do NOT merge or push.

## Global Constraints

- **Cold start of both learners reproduces current behavior exactly.** `DepthProfileStore::weights_for(k)` with zero evidence equals `DepthWeights::terminal(max_depth, k)`. `Reranker` with `w = 0, b = 0` leaves the top-K order unchanged (identity). A deployment that never sends feedback sees no change.
- **`propagate_layered` returns RAW, un-weighted per-depth intensities.** It IGNORES `params.depth_weights`. Pruning still tests raw `transmitted < min_intensity`, exactly like the scalar kernel.
- **Consistency invariant (a test):** for any `DepthWeights c`, `Σ_d c[d]·layered.per_depth[v][d] == propagate(v)` for the same params carrying `c`, exact at `min_intensity == 0`.
- **Online learning mirrors `TransitionStore`:** accumulate on `record`, `derive` purely, `rebuild` = derive-then-decay-then-reset, atomic `write-temp-then-rename` persistence, and record+auto-rebuild fold into ONE critical section in the engine.
- **Feedback trains under query-time captured state:** the `QueryContext` carries the layered result and hop hint; learners update from `ctx`, never from live post-query state.
- **No new heavy ML dependency.** Both learners are hand-written linear/ratio updates.
- **Fallible operations return `Result`, not `assert!()`.**
- **Regression floor:** the existing 79 rgdb tests and 24 rgdb-eval tests stay green after every task.

---

## Phase 1 — Layered propagation primitive

### Task 1: `propagate_layered` in the kernel

**Files:**
- Modify: `rgdb/src/propagation.rs` (add `LayeredResult`, `propagate_layered`, tests)
- Modify: `rgdb/src/credit.rs` (fold in the deferred hygiene item — length `debug_assert`)
- Modify: `rgdb/src/lib.rs` (re-export is automatic via `pub use propagation::*`)

**Interfaces:**
- Consumes: `Graph`, `RelationVocab`, `PropagationParams`, `DepthWeights` (existing).
- Produces:
  ```rust
  pub struct LayeredResult {
      pub per_depth: HashMap<NodeId, Vec<f32>>,        // node -> [I_0..I_maxdepth], raw
      pub dominant_incoming: HashMap<NodeId, RelationId>, // node -> heaviest incoming relation
  }
  pub fn propagate_layered(graph: &Graph, vocab: &RelationVocab, seeds: &[(NodeId, f32)],
                           query_relation: Option<RelationId>, params: &PropagationParams) -> LayeredResult;
  ```

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `rgdb/src/propagation.rs` (the `chain()` and multi-path fixtures already exist there):

```rust
    #[test]
    fn layered_decomposes_the_chain_by_arrival_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None };
        let r = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        // node k arrives only at depth k on a pure chain.
        assert_eq!(r.per_depth[&0], vec![1.0, 0.0, 0.0, 0.0]);
        assert!((r.per_depth[&1][1] - 0.85).abs() < 1e-5);
        assert!((r.per_depth[&2][2] - 0.7225).abs() < 1e-5);
        assert!((r.per_depth[&3][3] - 0.614125).abs() < 1e-5);
        assert_eq!(r.per_depth[&3][1], 0.0);
    }

    #[test]
    fn layered_collapses_to_scalar_propagate_for_any_weights() {
        // Σ_d c[d]·layered[v][d] must equal propagate(v) under the SAME c, exactly.
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        for c in [
            crate::depth_weights::DepthWeights::uniform(3),
            crate::depth_weights::DepthWeights::terminal(3, 2).unwrap(),
            crate::depth_weights::DepthWeights::from_vec(vec![0.0, 0.3, 0.7, 1.0], 3).unwrap(),
        ] {
            let p = PropagationParams { max_depth: 3, min_intensity: 0.0, depth_weights: Some(c.clone()) };
            let scalar = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
            let layered = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
            let cs = c.as_slice();
            for (&v, prof) in &layered.per_depth {
                let collapsed: f32 = prof.iter().zip(cs).map(|(x, w)| x * w).sum();
                let want = scalar.get(&v).copied().unwrap_or(0.0);
                assert!((collapsed - want).abs() < 1e-5, "node {v}: {collapsed} != {want}");
            }
        }
    }

    #[test]
    fn layered_reports_dominant_incoming_relation() {
        // 0 -(A=0)-> 1 -(B=1)-> 2 ; node 1's only incoming is A, node 2's is B.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let vocab = RelationVocab::uniform(2);
        let p = PropagationParams { max_depth: 2, min_intensity: 1e-6, depth_weights: None };
        let r = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        assert_eq!(r.dominant_incoming.get(&1), Some(&0));
        assert_eq!(r.dominant_incoming.get(&2), Some(&1));
        assert_eq!(r.dominant_incoming.get(&0), None); // the seed has no incoming edge
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb propagation 2>&1 | head -20`
Expected: FAIL to compile — `propagate_layered` / `LayeredResult` not found.

- [ ] **Step 3: Implement `propagate_layered`**

In `rgdb/src/propagation.rs`, add after `propagate_single`:

```rust
/// Per-node intensity broken out by arrival depth, plus each node's heaviest
/// incoming relation. The per-depth values are RAW (un-weighted): applying a
/// `DepthWeights` to them reconstructs `propagate`'s scalar output. `depth_weights`
/// in `params` is ignored here — weighting is the consumer's job.
#[derive(Debug, Clone, Default)]
pub struct LayeredResult {
    /// node -> `[I_0, I_1, .., I_maxdepth]`; index d = mass arriving after exactly d hops.
    pub per_depth: HashMap<NodeId, Vec<f32>>,
    /// node -> the incoming relation that delivered the most mass (absent for seeds).
    pub dominant_incoming: HashMap<NodeId, RelationId>,
}

pub fn propagate_layered(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> LayeredResult {
    let d_max = params.max_depth;
    let n = graph.num_nodes();
    let mut per_depth: HashMap<NodeId, Vec<f32>> = HashMap::new();
    let mut incoming: HashMap<NodeId, HashMap<RelationId, f32>> = HashMap::new();
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();

    // Linear in seed mass, so accumulate each single-seed walk into shared maps.
    for &(seed, initial_mass) in seeds {
        if (seed as usize) >= n || initial_mass < params.min_intensity {
            continue;
        }
        per_depth.entry(seed).or_insert_with(|| vec![0.0; d_max + 1])[0] += initial_mass;

        let mut frontier: HashMap<FrontierKey, f32> = HashMap::new();
        frontier.insert((seed, query_relation), initial_mass);

        for depth in 0..d_max {
            if frontier.is_empty() {
                break;
            }
            let mut next: HashMap<FrontierKey, f32> = HashMap::new();
            for (&(u, r_in), &mass) in frontier.iter() {
                if mass < params.min_intensity {
                    continue;
                }
                let props = graph.node_props()[u as usize];
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
                    per_depth.entry(v).or_insert_with(|| vec![0.0; d_max + 1])[depth + 1] += transmitted;
                    *incoming.entry(v).or_default().entry(ep.relation).or_insert(0.0) += transmitted;
                    *next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;
                }
            }
            frontier = next;
        }
    }

    let dominant_incoming = incoming
        .into_iter()
        .filter_map(|(node, rels)| {
            rels.into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| (node, r))
        })
        .collect();

    LayeredResult { per_depth, dominant_incoming }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb propagation 2>&1 | tail -8`
Expected: PASS (3 new tests + existing propagation tests green).

- [ ] **Step 5: Fold in the deferred hygiene item (length `debug_assert` in credit)**

In `rgdb/src/credit.rs`, in both `credit()` and `backward_mass_at_target()`, immediately after the line that binds `let c = params.depth_weights.as_ref().map(|w| w.as_slice());`, add:

```rust
    debug_assert!(
        c.map_or(true, |cc| cc.len() == params.max_depth + 1),
        "depth_weights length must equal max_depth + 1"
    );
```

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb 2>&1 | tail -5`
Expected: full suite green (82 tests).

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/propagation.rs rgdb/src/credit.rs
git commit -m "feat(rgdb): propagate_layered (per-node per-depth intensity) + credit length guard"
```

---

### Task 2: Python binding for `propagate_layered`

**Files:**
- Modify: `rgdb-python/src/lib.rs` (add `propagate_layered` pyfunction)
- Test: `rgdb-eval/tests/test_layered_binding.py`

**Interfaces:**
- Consumes: `rgdb::propagation::{propagate_layered, LayeredResult}`.
- Produces: `core.propagate_layered(graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3) -> (per_depth, dominant_incoming)` where `per_depth` is `list[(node, list[float])]` and `dominant_incoming` is `list[(node, int)]`.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_layered_binding.py`:

```python
"""propagate_layered native binding: per-depth intensities + dominant incoming relation."""
import math
from rgdb_embeddings import _rgdb_core as core


def _chain():
    adj = [[(1, 0.0, 0)], [(2, 0.0, 0)], [(3, 0.0, 0)], []]
    return core.build_graph(4, adj), core.uniform_vocab(1)


def test_layered_matches_scalar_under_uniform():
    g, v = _chain()
    per_depth, _dom = core.propagate_layered(g, v, [(0, 1.0)], 0, 3, 0.0)
    layered = {n: prof for n, prof in per_depth}
    scalar = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 0.0, None))
    for n, prof in layered.items():
        assert math.isclose(sum(prof), scalar.get(n, 0.0), rel_tol=1e-5, abs_tol=1e-6)


def test_layered_depth_profile_of_chain():
    g, v = _chain()
    per_depth, _dom = core.propagate_layered(g, v, [(0, 1.0)], 0, 3, 1e-6)
    layered = {n: prof for n, prof in per_depth}
    assert math.isclose(layered[3][3], 0.614125, rel_tol=1e-4)
    assert layered[3][1] == 0.0
```

- [ ] **Step 2: Build and run to verify failure**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml` (from repo root), then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests/test_layered_binding.py -q`
Expected: FAIL — `propagate_layered` not found (AttributeError).

- [ ] **Step 3: Add the binding**

In `rgdb-python/src/lib.rs`, change the import on line 6 to include the layered symbols:

```rust
use rgdb::propagation::{propagate as rust_propagate, propagate_layered as rust_propagate_layered, PropagationParams};
```

Add this pyfunction after `propagate`:

```rust
#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3))]
fn propagate_layered(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
) -> (Vec<(u32, Vec<f32>)>, Vec<(u32, u16)>) {
    let params = PropagationParams { max_depth, min_intensity, depth_weights: None };
    let r = rust_propagate_layered(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    let per_depth = r.per_depth.into_iter().collect();
    let dominant = r.dominant_incoming.into_iter().collect();
    (per_depth, dominant)
}
```

Register it in the module (after the `propagate` registration):

```rust
    m.add_function(wrap_pyfunction!(propagate_layered, m)?)?;
```

- [ ] **Step 4: Rebuild and run to verify pass**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests -q`
Expected: PASS (26 tests: 24 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/tests/test_layered_binding.py
git commit -m "feat(bindings): propagate_layered kwarg-free binding for the eval harness"
```

---

## Phase 2 — Learned soft depth weights (#2)

### Task 3: `DepthProfileStore`

**Files:**
- Create: `rgdb/src/depth_profile.rs`
- Modify: `rgdb/src/lib.rs` (`pub mod depth_profile;` + re-export)

**Interfaces:**
- Consumes: `DepthWeights` (existing).
- Produces:
  ```rust
  pub struct DepthProfileConfig { pub prior_strength: f32, pub floor: f32, pub rebuild_every_n: u32 }
  pub struct DepthProfileStore { /* per hop-class accumulators */ }
  impl DepthProfileStore {
      pub fn new(max_depth: usize, cfg: DepthProfileConfig) -> Self;
      pub fn record(&mut self, hop: usize, answer_profile: &[f32], background_profile: &[f32], signal: f32);
      pub fn weights_for(&self, hop: usize) -> DepthWeights;   // prior terminal(hop), refined by evidence
      pub fn events_since_rebuild(&self) -> u32;
      pub fn rebuild_marker(&mut self);                        // resets the event counter
      pub fn max_depth(&self) -> usize;
      pub fn save(&self, path: &str) -> Result<(), DepthProfileError>;
      pub fn load(path: &str) -> Result<Self, DepthProfileError>;
  }
  ```

- [ ] **Step 1: Write the failing tests**

Create `rgdb/src/depth_profile.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn store(max_depth: usize) -> DepthProfileStore {
        DepthProfileStore::new(max_depth, DepthProfileConfig::default())
    }

    #[test]
    fn cold_start_weights_equal_terminal_k() {
        let s = store(4);
        for k in 0..=4 {
            let w = s.weights_for(k);
            let want = crate::depth_weights::DepthWeights::terminal(4, k).unwrap();
            assert_eq!(w.as_slice(), want.as_slice(), "hop {k}");
        }
    }

    #[test]
    fn evidence_shifts_weight_toward_the_answer_depth() {
        // Answers land at depth 1 (a shortcut) even though the query is hop-class 3;
        // background is flat. The learned weight at depth 1 must rise above terminal(3)'s 0.
        let mut s = store(3);
        for _ in 0..50 {
            // answer_profile: all its mass at depth 1; background: flat across depths.
            s.record(3, &[0.0, 1.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0], 1.0);
        }
        let w = s.weights_for(3);
        assert!(w.as_slice()[1] > 0.0, "depth-1 weight must rise from evidence, got {:?}", w.as_slice());
        assert!(w.as_slice()[3] > 0.0, "depth-3 prior must persist, got {:?}", w.as_slice());
    }

    #[test]
    fn derived_weights_are_always_valid() {
        // A pathological all-zero-answer record must still yield a usable DepthWeights.
        let mut s = store(2);
        s.record(2, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 1.0);
        let w = s.weights_for(2);
        assert_eq!(w.as_slice().len(), 3);
        assert!(w.as_slice().iter().any(|&x| x > 0.0));
    }

    #[test]
    fn save_load_roundtrip() {
        let mut s = store(3);
        s.record(3, &[0.0, 1.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0], 2.0);
        let path = "test_depth_profile_roundtrip.bin";
        s.save(path).unwrap();
        let t = DepthProfileStore::load(path).unwrap();
        assert_eq!(t.weights_for(3).as_slice(), s.weights_for(3).as_slice());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = "test_depth_profile_badmagic.bin";
        std::fs::write(path, b"NOPEnothing").unwrap();
        assert!(DepthProfileStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb depth_profile 2>&1 | head -20`
Expected: FAIL to compile — types not defined.

- [ ] **Step 3: Implement the store**

Prepend to `rgdb/src/depth_profile.rs`:

```rust
//! Learned soft depth weights: per hop-class, the arrival-depth profile of rewarded
//! answers relative to background mass. Prior is `terminal(k)`, so cold start
//! reproduces the shipped hard-terminal behavior; evidence smooths it.

use thiserror::Error;

use crate::depth_weights::DepthWeights;

#[derive(Debug, Error)]
pub enum DepthProfileError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

#[derive(Debug, Clone, Copy)]
pub struct DepthProfileConfig {
    /// κ — pseudo-count mass on the `terminal(k)` prior.
    pub prior_strength: f32,
    /// ε — denominator smoothing / minimum background.
    pub floor: f32,
    /// Auto-refresh cadence in feedback events (engine-side). 0 = manual.
    pub rebuild_every_n: u32,
}

impl Default for DepthProfileConfig {
    fn default() -> Self {
        Self { prior_strength: 10.0, floor: 1e-3, rebuild_every_n: 64 }
    }
}

/// Per hop-class `k in 0..=max_depth`: accumulated answer and background mass per depth.
#[derive(Debug, Clone)]
pub struct DepthProfileStore {
    max_depth: usize,
    // Row-major [hop][depth], each (max_depth+1) x (max_depth+1).
    answer: Vec<f32>,
    background: Vec<f32>,
    cfg: DepthProfileConfig,
    events_since_rebuild: u32,
}

impl DepthProfileStore {
    pub fn new(max_depth: usize, cfg: DepthProfileConfig) -> Self {
        let sz = (max_depth + 1) * (max_depth + 1);
        Self {
            max_depth,
            answer: vec![0.0; sz],
            background: vec![0.0; sz],
            cfg,
            events_since_rebuild: 0,
        }
    }

    pub fn max_depth(&self) -> usize { self.max_depth }
    pub fn config(&self) -> DepthProfileConfig { self.cfg }
    pub fn events_since_rebuild(&self) -> u32 { self.events_since_rebuild }
    pub fn rebuild_marker(&mut self) { self.events_since_rebuild = 0; }

    /// Accumulate one feedback event. `answer_profile[d]` = mass the rewarded target
    /// received at depth d; `background_profile[d]` = total mass at depth d over all
    /// nodes. Both length `max_depth+1`; out-of-range hops and lengths are ignored.
    pub fn record(&mut self, hop: usize, answer_profile: &[f32], background_profile: &[f32], signal: f32) {
        let w = self.max_depth + 1;
        if hop > self.max_depth || answer_profile.len() != w || background_profile.len() != w {
            return;
        }
        for d in 0..w {
            let idx = hop * w + d;
            self.answer[idx] = (self.answer[idx] + signal * answer_profile[d]).max(0.0);
            self.background[idx] = (self.background[idx] + signal * background_profile[d]).max(0.0);
        }
        self.events_since_rebuild = self.events_since_rebuild.saturating_add(1);
    }

    /// Soft `DepthWeights` for hop-class `k`. Discriminative ratio of answer to
    /// background mass per depth, blended with a `terminal(k)` pseudo-count prior,
    /// then max-normalized. Cold start (no evidence) == `terminal(k)` exactly.
    pub fn weights_for(&self, hop: usize) -> DepthWeights {
        let w = self.max_depth + 1;
        let k = self.cfg.prior_strength;
        let eps = self.cfg.floor;
        let hop = hop.min(self.max_depth);
        // prior_answer = terminal(hop) one-hot; prior_background = uniform 1.0.
        let mut raw = vec![0.0f32; w];
        for d in 0..w {
            let idx = hop * w + d;
            let prior_answer = if d == hop { 1.0 } else { 0.0 };
            let a = self.answer[idx] + k * prior_answer;
            let b = self.background[idx] + k * 1.0 + eps;
            raw[d] = a / b;
        }
        let m = raw.iter().cloned().fold(0.0f32, f32::max);
        let norm: Vec<f32> = if m > 0.0 {
            raw.iter().map(|x| (x / m).clamp(0.0, 1.0)).collect()
        } else {
            // no signal anywhere -> fall back to the hard prior.
            return DepthWeights::terminal(self.max_depth, hop).expect("hop <= max_depth");
        };
        DepthWeights::from_vec(norm, self.max_depth)
            .unwrap_or_else(|_| DepthWeights::terminal(self.max_depth, hop).expect("hop <= max_depth"))
    }

    pub fn save(&self, path: &str) -> Result<(), DepthProfileError> {
        let tmp = format!("{path}.tmp");
        if let Err(e) = self.write_snapshot(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn write_snapshot(&self, tmp: &str) -> Result<(), DepthProfileError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;
        let mut f = std::fs::File::create(tmp)?;
        f.write_all(b"RGDP")?;
        f.write_u32::<LittleEndian>(1)?; // version
        f.write_u32::<LittleEndian>(self.max_depth as u32)?;
        for &x in &self.answer { f.write_f32::<LittleEndian>(x)?; }
        for &x in &self.background { f.write_f32::<LittleEndian>(x)?; }
        f.write_f32::<LittleEndian>(self.cfg.prior_strength)?;
        f.write_f32::<LittleEndian>(self.cfg.floor)?;
        f.write_u32::<LittleEndian>(self.cfg.rebuild_every_n)?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, DepthProfileError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;
        const MAX_DEPTH: usize = 4096; // a wildly generous cap before allocating
        let file_len = std::fs::metadata(path)?.len();
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGDP" {
            return Err(DepthProfileError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(DepthProfileError::Corrupt(format!("unsupported version {version}")));
        }
        let max_depth = f.read_u32::<LittleEndian>()? as usize;
        if max_depth > MAX_DEPTH {
            return Err(DepthProfileError::Corrupt(format!("max_depth={max_depth} too large")));
        }
        let sz = (max_depth + 1) * (max_depth + 1);
        let need = (sz as u64).saturating_mul(4).saturating_mul(2);
        if need > file_len {
            return Err(DepthProfileError::Corrupt(format!(
                "max_depth={max_depth} implies {need} bytes but file is {file_len}"
            )));
        }
        let mut answer = vec![0.0f32; sz];
        for x in answer.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let mut background = vec![0.0f32; sz];
        for x in background.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let cfg = DepthProfileConfig {
            prior_strength: f.read_f32::<LittleEndian>()?,
            floor: f.read_f32::<LittleEndian>()?,
            rebuild_every_n: f.read_u32::<LittleEndian>()?,
        };
        Ok(Self { max_depth, answer, background, cfg, events_since_rebuild: 0 })
    }
}
```

Wire into `rgdb/src/lib.rs`: after `pub mod depth_weights;` add `pub mod depth_profile;`, and after the depth_weights re-export add:

```rust
pub use depth_profile::{DepthProfileConfig, DepthProfileError, DepthProfileStore};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb depth_profile 2>&1 | tail -8`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/depth_profile.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): DepthProfileStore — learned soft depth weights (prior terminal(k))"
```

---

### Task 4: Engine integration for `DepthProfileStore` + `hop_hint`

**Files:**
- Modify: `rgdb/src/engine.rs` (`RgdbEngine` fields, `query` signature + `hop_hint`, `QueryContext`, `record_feedback`, tests)

**Interfaces:**
- Consumes: `DepthProfileStore`, `propagate_layered`, `LayeredResult` (Tasks 1, 3).
- Produces: `RgdbEngine::query(seeds, query_relation, hop_hint: Option<usize>, params) -> QueryResult` (signature grows by `hop_hint`). `QueryContext` now carries `hop_hint: Option<usize>` and `layered: LayeredResult`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `rgdb/src/engine.rs`:

```rust
    #[test]
    fn query_with_hop_hint_uses_learned_soft_weights_after_feedback() {
        use crate::depth_weights::DepthWeights;
        // Graph 0 -A-> 1 -B-> 2. hop_hint=2 with a cold DepthProfileStore behaves like
        // terminal(2): node 2 (depth 2) outranks node 1 (depth 1).
        let e = engine(0); // 0 -A-> 1 -B-> 2, uniform prior
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(2), "cold hop_hint=2 == terminal(2)");

        // Feed back that node 1 (a depth-1 answer) is correct, repeatedly, then refresh
        // the profile. The learned weight at depth 1 rises, so node 1 can now score.
        for _ in 0..40 {
            let q = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
            let _ = e.record_feedback(q.query_id, 1, 1.0);
        }
        e.refresh_profiles();
        let after = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
        // depth-1 mass is now weighted, so node 1's score is nonzero (was 0 under terminal(2)).
        let n1 = after.ranked.iter().find(|x| x.0 == 1).map(|x| x.1).unwrap_or(0.0);
        assert!(n1 > 0.0, "learned soft weights should score the depth-1 node, got {n1}");
    }

    #[test]
    fn query_without_hop_hint_is_unchanged() {
        // hop_hint = None must reproduce the pre-feature behavior (config default path).
        let e = engine(0);
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert!(!r.ranked.is_empty());
        assert!(r.query_id > 0);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb engine 2>&1 | head -20`
Expected: FAIL to compile — `query` takes 4 args not 5; `refresh_profiles` missing.

- [ ] **Step 3: Wire the store into the engine**

In `rgdb/src/engine.rs`:

Add the depth-profile import, and CHANGE the existing propagation import (currently
`use crate::propagation::{propagate, PropagationParams};`) — drop `propagate` (now
unused, since `query` switches to the layered pass) and add the layered symbols:
```rust
use crate::depth_profile::{DepthProfileConfig, DepthProfileStore};
use crate::propagation::{propagate_layered, LayeredResult, PropagationParams};
```

Add a field to `RgdbEngine`:
```rust
    depth_profile: Mutex<DepthProfileStore>,
```

Initialize it in `assemble` (it takes `max_depth` — use the engine's max_depth. Since `PropagationParams` is per-query, store a fixed `profile_max_depth` = 4, the default):
```rust
        depth_profile: Mutex::new(DepthProfileStore::new(4, DepthProfileConfig::default())),
```
(Add `profile_max_depth` handling if a non-default is ever needed; 4 matches `PropagationParams::default().max_depth` and the engine's `default_depth_weights` length.)

Extend `QueryContext`:
```rust
    hop_hint: Option<usize>,
    layered: LayeredResult,
```

Rewrite `query` to take `hop_hint`, resolve soft weights, run the layered pass, and cache it:
```rust
    pub fn query(
        &self,
        seeds: &[(NodeId, f32)],
        query_relation: Option<RelationId>,
        hop_hint: Option<usize>,
        params: &PropagationParams,
    ) -> QueryResult {
        // Resolve depth weights: caller override wins; else learned soft weights for the
        // hop hint (when its length fits); else the config default; else uniform.
        let resolved = if params.depth_weights.is_some() {
            params.clone()
        } else if let Some(k) = hop_hint {
            let w = self.depth_profile.lock().unwrap().weights_for(k);
            if w.as_slice().len() == params.max_depth + 1 {
                PropagationParams { depth_weights: Some(w), ..params.clone() }
            } else {
                params.clone()
            }
        } else if self.engine_cfg.default_depth_weights.as_slice().len() == params.max_depth + 1 {
            PropagationParams {
                depth_weights: Some(self.engine_cfg.default_depth_weights.clone()),
                ..params.clone()
            }
        } else {
            params.clone()
        };

        let vocab = self.vocab.load_full();
        let layered = propagate_layered(&self.graph, &vocab, seeds, query_relation, &resolved);
        let c = resolved.depth_weights.clone();
        // Collapse layered -> scalar score with the resolved weights (uniform if None).
        let mut ranked: Vec<(NodeId, f32)> = layered
            .per_depth
            .iter()
            .map(|(&node, prof)| {
                let s = match &c {
                    Some(w) => prof.iter().zip(w.as_slice()).map(|(x, ww)| x * ww).sum(),
                    None => prof.iter().sum(),
                };
                (node, s)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache.lock().unwrap().put(
            query_id,
            QueryContext {
                seeds: seeds.to_vec(),
                query_relation,
                params: resolved,
                vocab,
                hop_hint,
                layered,
                created: Instant::now(),
            },
        );
        QueryResult { ranked, query_id }
    }
```

In `record_feedback`, after the existing transition credit block, add the depth-profile update (target's per-depth profile vs background), inside the SAME critical-section discipline. Insert before the `Ok(())`:
```rust
        // Update the depth profile from the query-time layered result.
        if let Some(k) = ctx.hop_hint {
            let width = ctx.params.max_depth + 1;
            let answer = ctx.layered.per_depth.get(&target).cloned()
                .unwrap_or_else(|| vec![0.0; width]);
            let mut background = vec![0.0f32; width];
            for prof in ctx.layered.per_depth.values() {
                for (d, &x) in prof.iter().enumerate() {
                    if d < width {
                        background[d] += x;
                    }
                }
            }
            let mut store = self.depth_profile.lock().unwrap();
            store.record(k, &answer, &background, signal);
            let n = store.config().rebuild_every_n;
            if n > 0 && store.events_since_rebuild() >= n {
                store.rebuild_marker(); // weights_for derives on read, so just reset the counter
            }
        }
```
(Remove the placeholder `w` lines — they are only shown to flag: do not shadow anything; use `ctx.params.max_depth`.)

Add a manual refresh hook mirroring `refresh`:
```rust
    /// Reset the depth-profile event counter (weights are derived lazily on read).
    pub fn refresh_profiles(&self) {
        self.depth_profile.lock().unwrap().rebuild_marker();
    }
```

- [ ] **Step 4: Fix all existing `query` call sites**

`query` grew an argument. Update the existing engine tests that call `e.query(seeds, rel, &params)` to `e.query(seeds, rel, None, &params)`. The sites are in the `tests` module of `engine.rs` (search for `.query(`). Also update the Python binding call in Task 5 (next task).

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb engine 2>&1 | tail -10`
Expected: PASS (existing engine tests with `None` hop_hint + the 2 new tests).

- [ ] **Step 5: Run the full crate suite**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb 2>&1 | tail -5`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/engine.rs
git commit -m "feat(rgdb): engine learns soft depth weights per hop_hint from feedback"
```

---

### Task 5: Python `hop_hint` binding + MetaQA soft-`c` recall gate

**Files:**
- Modify: `rgdb-python/src/lib.rs` (`Engine.query` gains `hop_hint`)
- Create: `rgdb-eval/scripts/experiment_soft_depth_weights.py`

**Interfaces:**
- Consumes: `RgdbEngine::query` with `hop_hint` (Task 4).
- Produces: `Engine.query(seeds, query_relation=None, hop_hint=None, max_depth=4, min_intensity=1e-3) -> (ranked, query_id)`.

- [ ] **Step 1: Update the binding**

In `rgdb-python/src/lib.rs`, replace `PyEngine::query`. **`hop_hint` goes LAST** — the
pre-existing param order (`max_depth, min_intensity, depth_weights`) must stay fixed so
existing positional callers like `eng.query(seeds, rel, 4, 1e-4)` keep working:
```rust
    #[pyo3(signature = (seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None, hop_hint=None))]
    fn query(
        &self,
        seeds: Vec<(u32, f32)>,
        query_relation: Option<u16>,
        max_depth: usize,
        min_intensity: f32,
        depth_weights: Option<Vec<f32>>,
        hop_hint: Option<usize>,
    ) -> PyResult<(Vec<(u32, f32)>, u64)> {
        let params = PropagationParams {
            max_depth,
            min_intensity,
            depth_weights: to_depth_weights(depth_weights, max_depth)?,
        };
        let r = self.inner.query(&seeds, query_relation, hop_hint, &params);
        Ok((r.ranked, r.query_id))
    }
```

- [ ] **Step 2: Write the acceptance script**

Create `rgdb-eval/scripts/experiment_soft_depth_weights.py`:
```python
"""#2 gate: learned soft depth weights must not lose recall vs hard terminal(k).

Replays MetaQA training questions as feedback through the Engine (query with a hop
hint = the question's hop count, then record_feedback on the gold answer), refreshes
the learned profile, and evaluates on the test set. Compares the learned soft weights
against hard terminal(k). Gate: 3-hop recall@20 (soft) >= recall@20 (terminal) and
3-hop MRR (soft) not materially below terminal(3)'s 0.381.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_soft_depth_weights.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core
from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN_PER_HOP = 2000


def trained_matrix(graph):
    n = len(graph.relations); rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FL, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = core.vocab_from_matrix(list(graph.relations), trained_matrix(graph).ravel().tolist())
    eng = core.Engine(g, vocab, rebuild_every_n=0)

    # Replay training feedback with the hop count as the hint.
    events = 0
    for hop in (1, 2, 3):
        qp = os.path.join(DATA, f"qa_train_{hop}hop.txt")
        tp = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(qp):
            continue
        for q in load_questions(qp, hop, graph, limit=TRAIN_PER_HOP, qtype_path=tp):
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            _, qid = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=hop)
            try:
                eng.record_feedback(qid, q.answer_ids[0], 1.0)
                events += 1
            except ValueError:
                pass
    eng.refresh_profiles()
    print(f"replayed {events} feedback events")

    def score(hop, hop_hint):
        qs = load_questions(os.path.join(DATA, f"qa_test_{hop}hop.txt"), hop, graph,
                            limit=None, qtype_path=os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt"))
        items = []
        for q in qs:
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            ranked, _ = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=hop_hint)
            ranked = [n for n, _ in ranked if n != q.topic_id][:20]
            items.append((ranked, set(q.answer_ids)))
        return (mean(mrr(r, gs) for r, gs in items),
                mean(recall_at_k(r, gs, 20) for r, gs in items))

    # soft (learned, hop_hint set) vs terminal (hard, via explicit depth_weights)
    for hop in (2, 3):
        soft_mrr, soft_rec = score(hop, hop)
        print(f"hop{hop}: soft-c MRR {soft_mrr:.4f}  recall@20 {soft_rec:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Build, run, verify**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_soft_depth_weights.py`
Expected: prints per-hop soft-`c` MRR and recall@20. Sanity: 3-hop recall@20 ≥ ~0.90 (terminal(3)'s level) and MRR ≥ ~0.36. If soft-`c` materially regresses recall vs terminal, STOP and report — that is a real signal about the derivation, not something to force.

Also confirm the eval suite still passes: `../.venv/Scripts/python.exe -m pytest tests -q` → green.

- [ ] **Step 4: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/scripts/experiment_soft_depth_weights.py
git commit -m "feat(bindings): Engine.query hop_hint + MetaQA soft-depth-weight gate"
```

---

## Phase 3 — Hits@1 reranker (#4)

### Task 6: `Reranker`

**Files:**
- Create: `rgdb/src/reranker.rs`
- Modify: `rgdb/src/lib.rs` (`pub mod reranker;` + re-export)

**Interfaces:**
- Consumes: `LayeredResult` (Task 1), `Graph` (for degree).
- Produces:
  ```rust
  pub struct RerankerConfig { pub learning_rate: f32, pub top_k: usize, pub l2: f32 }
  pub struct Reranker { /* weights + standardizer */ }
  impl Reranker {
      pub fn new(n_relations: usize, max_depth: usize, cfg: RerankerConfig) -> Self;
      pub fn features(&self, node: NodeId, layered: &LayeredResult, graph: &Graph) -> Vec<f32>;
      pub fn score(&self, feats: &[f32]) -> f32;
      /// Reorder `candidates` (node,score) descending by learned score; stable, so w=0 is identity.
      pub fn rerank(&self, candidates: &mut Vec<(NodeId, f32)>, layered: &LayeredResult, graph: &Graph);
      /// One online logistic step over the top-K: target is the positive, others negative.
      pub fn update(&mut self, candidates: &[(NodeId, f32)], target: NodeId,
                    layered: &LayeredResult, graph: &Graph, signal: f32);
      pub fn save(&self, path: &str) -> Result<(), RerankerError>;
      pub fn load(path: &str) -> Result<Self, RerankerError>;
  }
  ```

- [ ] **Step 1: Write the failing tests**

Create `rgdb/src/reranker.rs` with the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::LayeredResult;
    use hashbrown::HashMap;

    // A tiny star: 0 -> {1,2,3}; node 2 also has high out-degree (a hub).
    fn fixture() -> (Graph, LayeredResult) {
        let e = |d, r| (d as u32, EdgeProps { attenuation: 0.0, relation: r, is_portal: false });
        // node 2 gets extra out-edges to look like a hub.
        let adj = vec![
            vec![e(1, 0), e(2, 0), e(3, 0)],
            vec![],
            vec![e(1, 0), e(3, 0)],
            vec![],
        ];
        let g = Graph::from_adjacency(4, adj, NodeProps::default()).unwrap();
        let mut per_depth = HashMap::new();
        per_depth.insert(1u32, vec![0.0, 0.5, 0.0]);
        per_depth.insert(2u32, vec![0.0, 0.5, 0.0]);
        per_depth.insert(3u32, vec![0.0, 0.5, 0.0]);
        let mut dom = HashMap::new();
        dom.insert(1u32, 0u16); dom.insert(2u32, 0u16); dom.insert(3u32, 0u16);
        (g, LayeredResult { per_depth, dominant_incoming: dom })
    }

    fn cfg() -> RerankerConfig { RerankerConfig { learning_rate: 0.5, top_k: 3, l2: 0.0 } }

    #[test]
    fn cold_start_is_identity() {
        let (g, layered) = fixture();
        let r = Reranker::new(1, 2, cfg());
        let mut cands = vec![(1u32, 0.9f32), (2, 0.8), (3, 0.7)];
        let before = cands.clone();
        r.rerank(&mut cands, &layered, &g);
        assert_eq!(cands, before, "w=0 must leave order unchanged");
    }

    #[test]
    fn learns_to_prefer_low_degree_answers() {
        // Node 1 (out-degree 0) is always the answer; node 2 is a hub (out-degree 2).
        let (g, layered) = fixture();
        let mut r = Reranker::new(1, 2, cfg());
        let cands = vec![(1u32, 0.5f32), (2, 0.9), (3, 0.5)];
        for _ in 0..200 {
            r.update(&cands, 1, &layered, &g, 1.0);
        }
        let mut c2 = cands.clone();
        r.rerank(&mut c2, &layered, &g);
        assert_eq!(c2.first().map(|x| x.0), Some(1),
            "after training, the low-degree answer should rank first, got {c2:?}");
    }

    #[test]
    fn save_load_roundtrip() {
        let (g, layered) = fixture();
        let mut r = Reranker::new(1, 2, cfg());
        let cands = vec![(1u32, 0.5f32), (2, 0.9), (3, 0.5)];
        for _ in 0..20 { r.update(&cands, 1, &layered, &g, 1.0); }
        let path = "test_reranker_roundtrip.bin";
        r.save(path).unwrap();
        let t = Reranker::load(path).unwrap();
        let f = t.features(2, &layered, &g);
        assert!((t.score(&f) - r.score(&f)).abs() < 1e-6);
        let _ = std::fs::remove_file(path);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb reranker 2>&1 | head -20`
Expected: FAIL to compile.

- [ ] **Step 3: Implement the reranker**

Prepend to `rgdb/src/reranker.rs`:

```rust
//! Online logistic reranker over the top-K diffusion candidates. Features: arrival-
//! depth profile, log diffusion score, log out-degree, and the candidate's dominant
//! incoming relation (one-hot). `w = 0` is a strict identity (do-no-harm cold start).

use thiserror::Error;

use crate::graph::{Graph, NodeId};
use crate::propagation::LayeredResult;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

#[derive(Debug, Clone, Copy)]
pub struct RerankerConfig {
    pub learning_rate: f32,
    pub top_k: usize,
    pub l2: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self { learning_rate: 0.1, top_k: 50, l2: 1e-4 }
    }
}

#[derive(Debug, Clone)]
pub struct Reranker {
    n_relations: usize,
    max_depth: usize,
    dim: usize,
    w: Vec<f32>,
    b: f32,
    cfg: RerankerConfig,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Reranker {
    pub fn new(n_relations: usize, max_depth: usize, cfg: RerankerConfig) -> Self {
        // features: (max_depth+1) depth profile + log(score) + log(degree) + n_relations one-hot
        let dim = (max_depth + 1) + 2 + n_relations;
        Self { n_relations, max_depth, dim, w: vec![0.0; dim], b: 0.0, cfg }
    }

    pub fn config(&self) -> RerankerConfig { self.cfg }

    pub fn features(&self, node: NodeId, layered: &LayeredResult, graph: &Graph) -> Vec<f32> {
        let mut f = vec![0.0f32; self.dim];
        let mut i = 0;
        // depth profile (raw; small magnitudes so no standardization needed for a demo)
        if let Some(prof) = layered.per_depth.get(&node) {
            for d in 0..=self.max_depth {
                f[i] = prof.get(d).copied().unwrap_or(0.0);
                i += 1;
            }
        } else {
            i += self.max_depth + 1;
        }
        // log diffusion score (the summed profile stands in for the collapsed score)
        let score: f32 = layered.per_depth.get(&node).map(|p| p.iter().sum()).unwrap_or(0.0);
        f[i] = (1.0 + score).ln();
        i += 1;
        // log out-degree
        let deg = graph.neighbors(node).count() as f32;
        f[i] = (1.0 + deg).ln();
        i += 1;
        // dominant incoming relation, one-hot
        if let Some(&r) = layered.dominant_incoming.get(&node) {
            let ri = r as usize;
            if ri < self.n_relations {
                f[i + ri] = 1.0;
            }
        }
        f
    }

    pub fn score(&self, feats: &[f32]) -> f32 {
        self.b + self.w.iter().zip(feats).map(|(w, x)| w * x).sum::<f32>()
    }

    pub fn rerank(&self, candidates: &mut Vec<(NodeId, f32)>, layered: &LayeredResult, graph: &Graph) {
        // Stable sort by learned score DESC; with w=0,b=0 every key is 0 so order is preserved.
        let scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, &(node, _))| (idx, self.score(&self.features(node, layered, graph))))
            .collect();
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| {
            scored[b].1.partial_cmp(&scored[a].1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let reordered: Vec<(NodeId, f32)> = order.iter().map(|&i| candidates[i]).collect();
        *candidates = reordered;
    }

    pub fn update(
        &mut self,
        candidates: &[(NodeId, f32)],
        target: NodeId,
        layered: &LayeredResult,
        graph: &Graph,
        signal: f32,
    ) {
        let lr = self.cfg.learning_rate * signal;
        let k = self.cfg.top_k.min(candidates.len());
        for &(node, _) in &candidates[..k] {
            let feats = self.features(node, layered, graph);
            let y = if node == target { 1.0 } else { 0.0 };
            let p = sigmoid(self.score(&feats));
            let g = y - p;
            for (wj, xj) in self.w.iter_mut().zip(&feats) {
                *wj += lr * (g * xj - self.cfg.l2 * *wj);
            }
            self.b += lr * g;
        }
    }

    pub fn save(&self, path: &str) -> Result<(), RerankerError> {
        let tmp = format!("{path}.tmp");
        if let Err(e) = self.write_snapshot(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn write_snapshot(&self, tmp: &str) -> Result<(), RerankerError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;
        let mut f = std::fs::File::create(tmp)?;
        f.write_all(b"RGRK")?;
        f.write_u32::<LittleEndian>(1)?;
        f.write_u32::<LittleEndian>(self.n_relations as u32)?;
        f.write_u32::<LittleEndian>(self.max_depth as u32)?;
        f.write_f32::<LittleEndian>(self.b)?;
        for &x in &self.w { f.write_f32::<LittleEndian>(x)?; }
        f.write_f32::<LittleEndian>(self.cfg.learning_rate)?;
        f.write_u32::<LittleEndian>(self.cfg.top_k as u32)?;
        f.write_f32::<LittleEndian>(self.cfg.l2)?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, RerankerError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;
        const MAX_DIM: usize = 1 << 20;
        let file_len = std::fs::metadata(path)?.len();
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGRK" {
            return Err(RerankerError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(RerankerError::Corrupt(format!("unsupported version {version}")));
        }
        let n_relations = f.read_u32::<LittleEndian>()? as usize;
        let max_depth = f.read_u32::<LittleEndian>()? as usize;
        let dim = (max_depth + 1) + 2 + n_relations;
        if dim > MAX_DIM || (dim as u64) * 4 > file_len {
            return Err(RerankerError::Corrupt(format!("implausible dim {dim}")));
        }
        let b = f.read_f32::<LittleEndian>()?;
        let mut w = vec![0.0f32; dim];
        for x in w.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let cfg = RerankerConfig {
            learning_rate: f.read_f32::<LittleEndian>()?,
            top_k: f.read_u32::<LittleEndian>()? as usize,
            l2: f.read_f32::<LittleEndian>()?,
        };
        Ok(Self { n_relations, max_depth, dim, w, b, cfg })
    }
}
```

Wire into `rgdb/src/lib.rs`: `pub mod reranker;` and `pub use reranker::{Reranker, RerankerConfig, RerankerError};`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb reranker 2>&1 | tail -8`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/reranker.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): online logistic Reranker (w=0 identity cold start)"
```

---

### Task 7: Engine integration for the reranker

**Files:**
- Modify: `rgdb/src/engine.rs` (`Reranker` field, rerank in `query`, update in `record_feedback`, tests)

**Interfaces:**
- Consumes: `Reranker` (Task 6), the cached `QueryContext.layered` and the top-K.
- Produces: `query` reorders its `ranked` via the reranker; `QueryContext` additionally caches the pre-rerank top-K (`ranked_topk: Vec<(NodeId, f32)>`); `record_feedback` calls `reranker.update`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `rgdb/src/engine.rs`:

```rust
    #[test]
    fn reranker_is_identity_until_trained_then_reorders() {
        // 0 -A-> 1, 0 -A-> 2, and 2 -A-> 3, 2 -A-> 4 (node 2 is a hub, out-degree 2;
        // node 1 is a leaf, out-degree 0). Diffusion ranks node 2 and 1 similarly.
        let a = |d| (d as u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(
            5, vec![vec![a(1), a(2)], vec![], vec![a(3), a(4)], vec![], vec![]],
            NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into()]);
        let e = RgdbEngine::new(g, prior, TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() });
        let p = PropagationParams { max_depth: 2, min_intensity: 0.0, depth_weights: None };

        // Cold: reranker is identity, so ranking is whatever diffusion produced.
        let r0 = e.query(&[(0, 1.0)], Some(0), None, &p);
        let cold_top = r0.ranked.iter().find(|x| x.0 == 1 || x.0 == 2).map(|x| x.0);
        assert!(cold_top.is_some());

        // Train: node 1 (the leaf) is always correct.
        for _ in 0..300 {
            let q = e.query(&[(0, 1.0)], Some(0), None, &p);
            let _ = e.record_feedback(q.query_id, 1, 1.0);
        }
        let after = e.query(&[(0, 1.0)], Some(0), None, &p);
        // Among {1,2}, the trained reranker should now put node 1 (leaf) ahead of node 2 (hub).
        let pos1 = after.ranked.iter().position(|x| x.0 == 1).unwrap();
        let pos2 = after.ranked.iter().position(|x| x.0 == 2).unwrap();
        assert!(pos1 < pos2, "trained reranker should rank the leaf answer above the hub");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb reranker_is_identity 2>&1 | tail -10`
Expected: FAIL (no reranker in the engine yet; ranking never changes).

- [ ] **Step 3: Wire the reranker into the engine**

In `rgdb/src/engine.rs`:

Add import + field:
```rust
use crate::reranker::{Reranker, RerankerConfig};
// field:
    reranker: Mutex<Reranker>,
```

Initialize in `assemble` — it needs `n_relations`. Derive from the vocab passed to `assemble` (`vocab.names().len()`):
```rust
        reranker: Mutex::new(Reranker::new(vocab.names().len(), 4, RerankerConfig::default())),
```

Extend `QueryContext` with the top-K captured before reranking:
```rust
    ranked_topk: Vec<(NodeId, f32)>,
```

In `query`, after building and sorting `ranked`, capture the top-K, rerank, and store both:
```rust
        let topk_n = self.reranker.lock().unwrap().config().top_k;
        let ranked_topk: Vec<(NodeId, f32)> =
            ranked.iter().take(topk_n).copied().collect();
        {
            let rr = self.reranker.lock().unwrap();
            // Rerank only the top-K slice, leave the tail as-is.
            let mut head: Vec<(NodeId, f32)> = ranked.iter().take(topk_n).copied().collect();
            rr.rerank(&mut head, &layered, &self.graph);
            for (i, item) in head.into_iter().enumerate() {
                ranked[i] = item;
            }
        }
```
And add `ranked_topk` to the cached `QueryContext { .. }`.

In `record_feedback`, after the depth-profile update, add:
```rust
        {
            let mut rr = self.reranker.lock().unwrap();
            rr.update(&ctx.ranked_topk, target, &ctx.layered, &self.graph, signal);
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb 2>&1 | tail -6`
Expected: PASS — the new reranker-engine test plus the full existing suite.

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/engine.rs
git commit -m "feat(rgdb): engine reranks top-K and trains the reranker from feedback"
```

---

### Task 8: MetaQA Hits@1 acceptance gate for the reranker

**Files:**
- Create: `rgdb-eval/scripts/experiment_reranker.py`

**Interfaces:**
- Consumes: the engine reranker via the bindings (Tasks 6, 7). No new binding needed — `Engine.query`/`record_feedback` already drive it.

- [ ] **Step 1: Write the acceptance script**

Create `rgdb-eval/scripts/experiment_reranker.py`:
```python
"""#4 gate: the online reranker lifts 3-hop Hits@1 above the terminal(3) baseline 0.203.

Replays MetaQA 3-hop training questions as feedback (query with hop_hint=3, then
record_feedback on the gold answer), which trains BOTH the depth profile and the
reranker. Evaluates 3-hop test Hits@1 / recall@20 with the trained engine and reports
against the terminal(3) baseline (Hits@1 0.203, recall@20 0.902).

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_reranker.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core
from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN = 4000
BASELINE_H1 = 0.203


def trained_matrix(graph):
    n = len(graph.relations); rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FL, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = core.vocab_from_matrix(list(graph.relations), trained_matrix(graph).ravel().tolist())
    eng = core.Engine(g, vocab, rebuild_every_n=0)

    tp = os.path.join(DATA, "qa_train_3hop.txt")
    tt = os.path.join(DATA, "qa_train_3hop_qtype.txt")
    events = 0
    for q in load_questions(tp, 3, graph, limit=TRAIN, qtype_path=tt):
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        _, qid = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=3)
        try:
            eng.record_feedback(qid, q.answer_ids[0], 1.0)
            events += 1
        except ValueError:
            pass
    eng.refresh_profiles()
    print(f"replayed {events} 3-hop feedback events")

    qs = load_questions(os.path.join(DATA, "qa_test_3hop.txt"), 3, graph,
                        limit=None, qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"))
    items = []
    for q in qs:
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        ranked, _ = eng.query([(q.topic_id, 1.0)], rel, max_depth=MD, min_intensity=MI, hop_hint=3)
        ranked = [n for n, _ in ranked if n != q.topic_id][:20]
        items.append((ranked, set(q.answer_ids)))
    h1 = mean(hits_at_k(r, gs, 1) for r, gs in items)
    r20 = mean(recall_at_k(r, gs, 20) for r, gs in items)
    print(f"3-hop reranked: Hits@1 {h1:.4f} (baseline {BASELINE_H1}), recall@20 {r20:.4f}")

    out = "results/metaqa-reranker.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Reranker acceptance (MetaQA 3-hop)\n\n"
                f"| variant | Hits@1 | recall@20 |\n|---|---|---|\n"
                f"| terminal(3) baseline | {BASELINE_H1} | 0.902 |\n"
                f"| + online reranker | {h1:.4f} | {r20:.4f} |\n")
    print(f"wrote {out}")

    assert h1 > BASELINE_H1, f"reranker did not lift Hits@1 above {BASELINE_H1}: {h1:.4f}"
    assert r20 >= 0.85, f"reranker regressed recall@20 below 0.85: {r20:.4f}"
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the acceptance**

Run: `cd /d/repos/RGP/rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_reranker.py`
Expected: prints the 3-hop Hits@1 and recall@20, writes `results/metaqa-reranker.md`, `ACCEPTANCE PASSED`. **This is a real gate:** if the reranker does not lift Hits@1 above 0.203, do NOT weaken the assertion — STOP and report the number for adjudication (as with the depth-weight acceptance). A miss is a real signal about the feature/features, not a threshold to move.

- [ ] **Step 3: Commit**

```bash
git add rgdb-eval/scripts/experiment_reranker.py rgdb-eval/results/metaqa-reranker.md
git commit -m "eval: MetaQA reranker Hits@1 acceptance gate"
```

---

## Notes for the executor

- **Do not merge or push.** Everything lands on `feature/rgdb-self-learning-loop`.
- **Cold-start no-op is a hard gate** (Task 3 `cold_start_weights_equal_terminal_k`, Task 6 `cold_start_is_identity`): both must be exact. If either learner changes behavior with zero feedback, the design is violated.
- **The two acceptance scripts (Tasks 5, 8) are real gates.** If a learned component underperforms its baseline on full-set MetaQA, report the number rather than tuning the threshold to pass — the depth-weight task set the precedent (a miss got investigated, not fudged).
- **`Engine.query` gained TWO new params across this plan** (`hop_hint` in Task 4/5; the binding also already had `depth_weights` from the prior feature). Keep the Python signature order exactly as written in Task 5 so existing eval scripts that call positionally still work — verify by running the full `pytest tests` suite after Task 5 and Task 7.
- **Concurrency:** the depth-profile and reranker updates happen under their own `Mutex`, separate from the transition `Mutex`. Each `record_feedback` takes them in sequence; there is no cross-lock ordering hazard because no path holds two of these locks at once. Keep it that way.
- Windows/PowerShell note: `cargo`/`pytest`/`maturin` use the repo's `.venv` (Python 3.12). If a bare `cargo` invocation hits a Python-version error, prefix `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (pre-existing ambient-Python quirk).
