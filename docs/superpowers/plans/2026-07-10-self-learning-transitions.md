# Data-Weighted Transitions + Self-Learning Query Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RGDB's relation-transition matrix a first-class persisted artifact, and close a feedback loop so the engine learns which relation transitions lead to correct answers.

**Architecture:** Three new Rust modules. `transitions.rs` owns learned counts and derives a similarity matrix from them (pure state + math). `credit.rs` is a pure forward–backward pass that attributes a rewarded answer to the relation transitions that carried mass to it. `engine.rs` (`RgdbEngine`) composes graph + live vocab + store + a bounded query cache, and is the surface applications call. `propagate` is unchanged and `Graph` is not modified.

**Tech Stack:** Rust (crate `rgdb`), `arc-swap` (lock-free vocab swap), `lru` (bounded query cache), `byteorder` (sidecar I/O), pyo3/maturin (bindings), Python eval harness.

**Spec:** `docs/superpowers/specs/2026-07-10-self-learning-transitions-design.md`

## Global Constraints

- `propagate` and `propagate_single` are **unchanged**. `Graph` is **not modified**. No reverse CSR.
- Derivation rule: `C'[a][b] = C[a][b] + κ·P[a][b]`; `rowmax_a = max over b != a of C'[a][b]` (**off-diagonal only**); `M[a][a] = 1.0` (pinned); `M[a][b] = max(C'[a][b]/rowmax_a, ε)` for `b != a`; if `rowmax_a == 0` then `M[a][b] = 1.0`.
- Defaults: `prior_strength κ = 10.0`, `floor ε = 0.05`, `decay γ = 1.0` (off), `rebuild_every_n = 64` (0 = manual only).
- Default prior is **uniform (all-ones)**. Cold start must produce an **exactly all-ones** matrix (pure typed PPR, do no harm).
- Signals are signed: `C[a][b] += signal · credit[a][b]`, clamped to `≥ 0`.
- `credit()` returns **L1-normalized** credits (sum to 1.0 when any flow exists), empty when the target is unreachable.
- Hops with `r_in == None` (untyped first hop) contribute **no** transition and are skipped for crediting.
- `credit()` must use the `Arc<RelationVocab>` captured **at query time**, never the currently-live one.
- The forward/backward invariant `Σ_seeds seed_mass · B_maxdepth[(seed, qr)] == propagate(...)[target]` holds **only when `min_intensity == 0.0`**. Property tests must set it to `0.0`.
- Level file stays immutable. Learned state persists to a sidecar.
- Commit after each task. `cargo test -p rgdb` green at every task boundary, with **no warnings**.

---

### Task 1: `TransitionStore` — counts and the derivation rule

**Files:**
- Create: `rgdb/src/transitions.rs`
- Modify: `rgdb/src/lib.rs` (add `pub mod transitions;`)

**Interfaces:**
- Consumes: `RelationId` (graph.rs), `RelationVocab::new` (relation.rs).
- Produces (used by Tasks 2, 4):
  - `TransitionConfig { prior_strength: f32, floor: f32, decay: f32, rebuild_every_n: u32 }` + `Default`
  - `TransitionError` (variants `BadShape`, `RelationCoverage`, `Io`, `Corrupt`)
  - `TransitionStore::new(names: Vec<String>, prior: Option<Vec<f32>>, cfg: TransitionConfig) -> Result<Self, TransitionError>`
  - `TransitionStore::{len, is_empty, names, config, counts, events_since_rebuild}`
  - `TransitionStore::record(&mut self, credits: &[(RelationId, RelationId, f32)], signal: f32)`
  - `TransitionStore::derive_vocab(&self) -> RelationVocab` (pure)
  - `TransitionStore::rebuild(&mut self) -> RelationVocab` (derive, then decay counts, reset counter)

- [ ] **Step 1: Write the failing tests**

Create `rgdb/src/transitions.rs` containing ONLY this test module for now:

```rust
//! Learned relation-transition counts and the similarity matrix derived from them.

#[cfg(test)]
mod tests {
    use super::*;

    fn store(n: usize) -> TransitionStore {
        let names: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        TransitionStore::new(names, None, TransitionConfig::default()).unwrap()
    }

    #[test]
    fn cold_start_is_exactly_uniform() {
        // uniform prior + zero evidence => every entry 1.0 => pure typed PPR
        let v = store(3).derive_vocab();
        for a in 0..3u16 {
            for b in 0..3u16 {
                assert_eq!(v.similarity(a, b), 1.0, "sim({a},{b})");
            }
        }
    }

    #[test]
    fn recorded_transition_dominates_its_row() {
        // kappa=10, prior=1 => C'[0][1] = 100+10 = 110, C'[0][2] = 0+10 = 10
        // off-diagonal rowmax = 110 => M[0][1] = 1.0, M[0][2] = 10/110 = 0.0909...
        let mut s = store(3);
        s.record(&[(0, 1, 1.0)], 100.0);
        let v = s.derive_vocab();
        assert!((v.similarity(0, 1) - 1.0).abs() < 1e-6);
        assert!((v.similarity(0, 2) - (10.0 / 110.0)).abs() < 1e-5, "got {}", v.similarity(0, 2));
    }

    #[test]
    fn diagonal_credit_does_not_shrink_off_diagonals() {
        // THE regression test: a 1-hop query dumps all credit on (r,r).
        // If the diagonal joined the row max, every off-diagonal would floor out.
        let mut s = store(3);
        s.record(&[(0, 0, 1.0)], 1000.0);
        let v = s.derive_vocab();
        assert_eq!(v.similarity(0, 0), 1.0, "diagonal pinned");
        assert!((v.similarity(0, 1) - 1.0).abs() < 1e-6, "off-diagonal untouched");
        assert!((v.similarity(0, 2) - 1.0).abs() < 1e-6, "off-diagonal untouched");
    }

    #[test]
    fn floor_applies_to_off_diagonals() {
        // C'[0][1] = 310, C'[0][2] = 10 => 10/310 = 0.032 < eps(0.05) => floored
        let mut s = store(3);
        s.record(&[(0, 1, 1.0)], 300.0);
        let v = s.derive_vocab();
        assert!((v.similarity(0, 2) - 0.05).abs() < 1e-6, "got {}", v.similarity(0, 2));
    }

    #[test]
    fn negative_signal_clamps_at_zero() {
        let mut s = store(2);
        s.record(&[(0, 1, 1.0)], -5.0);
        assert_eq!(s.counts()[0 * 2 + 1], 0.0);
    }

    #[test]
    fn rebuild_resets_counter_and_applies_decay() {
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let cfg = TransitionConfig { decay: 0.5, ..TransitionConfig::default() };
        let mut s = TransitionStore::new(names, None, cfg).unwrap();
        s.record(&[(0, 1, 1.0)], 8.0);
        assert_eq!(s.events_since_rebuild(), 1);
        let _ = s.rebuild();
        assert_eq!(s.events_since_rebuild(), 0);
        assert_eq!(s.counts()[1], 4.0, "decay applied after derive");
    }

    #[test]
    fn bad_prior_shape_rejected() {
        let names = vec!["a".into(), "b".into()];
        assert!(TransitionStore::new(names, Some(vec![1.0; 3]), TransitionConfig::default()).is_err());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rgdb transitions:: 2>&1 | tail -5`
Expected: compile error — `cannot find TransitionStore / TransitionConfig`.

- [ ] **Step 3: Implement the store**

Prepend to `rgdb/src/transitions.rs` (above the test module):

```rust
use thiserror::Error;

use crate::graph::RelationId;
use crate::relation::RelationVocab;

#[derive(Debug, Error)]
pub enum TransitionError {
    #[error("matrix must be {expected} entries ({n}x{n}), got {got}")]
    BadShape { n: usize, expected: usize, got: usize },
    #[error("snapshot covers {store_len} relations but the graph uses relation id {graph_max}")]
    RelationCoverage { store_len: usize, graph_max: usize },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

/// Knobs for how learned evidence turns into a similarity matrix.
#[derive(Debug, Clone, Copy)]
pub struct TransitionConfig {
    /// κ — pseudo-count mass placed on the prior. "How much evidence before I stop trusting it."
    pub prior_strength: f32,
    /// ε — minimum off-diagonal similarity. Never fully kill a path.
    pub floor: f32,
    /// γ — multiplicative decay applied to counts on each rebuild. 1.0 = off.
    pub decay: f32,
    /// Auto-refresh cadence in feedback events. 0 = manual only.
    pub rebuild_every_n: u32,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self { prior_strength: 10.0, floor: 0.05, decay: 1.0, rebuild_every_n: 64 }
    }
}

/// Accumulated per-transition evidence, plus the prior it blends with.
#[derive(Debug, Clone)]
pub struct TransitionStore {
    names: Vec<String>,
    counts: Vec<f32>, // n*n, row-major
    prior: Vec<f32>,  // n*n, row-major
    cfg: TransitionConfig,
    events_since_rebuild: u32,
}

impl TransitionStore {
    /// `prior = None` means the uniform (all-ones) do-no-harm prior.
    pub fn new(
        names: Vec<String>,
        prior: Option<Vec<f32>>,
        cfg: TransitionConfig,
    ) -> Result<Self, TransitionError> {
        let n = names.len();
        let expected = n * n;
        let prior = match prior {
            Some(p) if p.len() != expected => {
                return Err(TransitionError::BadShape { n, expected, got: p.len() })
            }
            Some(p) => p,
            None => vec![1.0; expected],
        };
        Ok(Self { names, counts: vec![0.0; expected], prior, cfg, events_since_rebuild: 0 })
    }

    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }
    pub fn names(&self) -> &[String] { &self.names }
    pub fn counts(&self) -> &[f32] { &self.counts }
    pub fn prior(&self) -> &[f32] { &self.prior }
    pub fn config(&self) -> TransitionConfig { self.cfg }
    pub fn events_since_rebuild(&self) -> u32 { self.events_since_rebuild }

    /// Accumulate one feedback event's (already L1-normalized) credits.
    pub fn record(&mut self, credits: &[(RelationId, RelationId, f32)], signal: f32) {
        let n = self.names.len();
        for &(a, b, c) in credits {
            let (ai, bi) = (a as usize, b as usize);
            if ai >= n || bi >= n {
                continue;
            }
            let idx = ai * n + bi;
            self.counts[idx] = (self.counts[idx] + signal * c).max(0.0);
        }
        self.events_since_rebuild = self.events_since_rebuild.saturating_add(1);
    }

    /// Pure derivation. The row max is taken over OFF-DIAGONAL entries only — a
    /// 1-hop query dumps all its credit on (r,r), and letting the diagonal into
    /// the row max would floor every off-diagonal and collapse the matrix back to
    /// "penalize every relation change".
    pub fn derive_vocab(&self) -> RelationVocab {
        let n = self.names.len();
        let k = self.cfg.prior_strength;
        let eps = self.cfg.floor;
        let blended = |a: usize, b: usize| self.counts[a * n + b] + k * self.prior[a * n + b];

        let mut m = vec![0.0f32; n * n];
        for a in 0..n {
            let mut rowmax = 0.0f32;
            for b in 0..n {
                if b != a {
                    rowmax = rowmax.max(blended(a, b));
                }
            }
            for b in 0..n {
                m[a * n + b] = if b == a {
                    1.0
                } else if rowmax > 0.0 {
                    (blended(a, b) / rowmax).clamp(0.0, 1.0).max(eps)
                } else {
                    1.0
                };
            }
        }
        RelationVocab::new(self.names.clone(), m).expect("n*n by construction")
    }

    /// Derive the current matrix, then decay counts and reset the event counter.
    pub fn rebuild(&mut self) -> RelationVocab {
        let vocab = self.derive_vocab();
        if self.cfg.decay != 1.0 {
            for c in self.counts.iter_mut() {
                *c *= self.cfg.decay;
            }
        }
        self.events_since_rebuild = 0;
        vocab
    }
}
```

- [ ] **Step 4: Register the module**

In `rgdb/src/lib.rs`, add after `pub mod relation;`:

```rust
pub mod transitions;
```

- [ ] **Step 5: Run to verify the tests pass**

Run: `cargo test -p rgdb transitions:: 2>&1 | tail -8`
Expected: `test result: ok. 7 passed`.

Run: `cargo build -p rgdb 2>&1 | tail -2`
Expected: `Finished`, no warnings.

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/transitions.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): TransitionStore with off-diagonal row-max derivation"
```

---

### Task 2: `TransitionStore` persistence (sidecar snapshot)

**Files:**
- Modify: `rgdb/src/transitions.rs` (add `save`/`load` + tests)

**Interfaces:**
- Consumes: `TransitionStore` fields from Task 1.
- Produces (used by Task 4):
  - `TransitionStore::save(&self, path: &str) -> Result<(), TransitionError>` — atomic (temp + rename)
  - `TransitionStore::load(path: &str) -> Result<Self, TransitionError>`

**Format** (little-endian): magic `b"RGTR"`, `version: u32 = 1`, `n: u32`, then `n` names (each `u32` byte-length + UTF-8 bytes), then `n*n` `f32` counts, then `n*n` `f32` prior, then `prior_strength: f32`, `floor: f32`, `decay: f32`, `rebuild_every_n: u32`.

- [ ] **Step 1: Write the failing tests**

Add these tests inside the existing `mod tests` in `rgdb/src/transitions.rs`:

```rust
    #[test]
    fn save_load_roundtrip() {
        let mut s = TransitionStore::new(
            vec!["isa".into(), "causes".into()],
            Some(vec![1.0, 0.3, 0.3, 1.0]),
            TransitionConfig { prior_strength: 7.0, floor: 0.02, decay: 0.9, rebuild_every_n: 5 },
        ).unwrap();
        s.record(&[(0, 1, 1.0)], 3.0);

        let path = "test_transitions_roundtrip.bin";
        s.save(path).unwrap();
        let t = TransitionStore::load(path).unwrap();

        assert_eq!(t.names(), s.names());
        assert_eq!(t.counts(), s.counts());
        assert_eq!(t.prior(), s.prior());
        assert_eq!(t.config().prior_strength, 7.0);
        assert_eq!(t.config().floor, 0.02);
        assert_eq!(t.config().decay, 0.9);
        assert_eq!(t.config().rebuild_every_n, 5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = "test_transitions_badmagic.bin";
        std::fs::write(path, b"NOPEnothing").unwrap();
        assert!(TransitionStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_truncated_file() {
        let path = "test_transitions_trunc.bin";
        std::fs::write(path, b"RGTR\x01\x00\x00\x00").unwrap(); // magic + version, nothing else
        assert!(TransitionStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rgdb transitions::tests::save_load_roundtrip 2>&1 | tail -5`
Expected: compile error — no method `save`.

- [ ] **Step 3: Implement persistence**

Add to the `impl TransitionStore` block in `rgdb/src/transitions.rs`:

```rust
    /// Atomic write: serialize to `<path>.tmp`, then rename over `path`.
    pub fn save(&self, path: &str) -> Result<(), TransitionError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;

        let tmp = format!("{path}.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(b"RGTR")?;
            f.write_u32::<LittleEndian>(1)?; // version
            f.write_u32::<LittleEndian>(self.names.len() as u32)?;
            for name in &self.names {
                let b = name.as_bytes();
                f.write_u32::<LittleEndian>(b.len() as u32)?;
                f.write_all(b)?;
            }
            for &c in &self.counts {
                f.write_f32::<LittleEndian>(c)?;
            }
            for &p in &self.prior {
                f.write_f32::<LittleEndian>(p)?;
            }
            f.write_f32::<LittleEndian>(self.cfg.prior_strength)?;
            f.write_f32::<LittleEndian>(self.cfg.floor)?;
            f.write_f32::<LittleEndian>(self.cfg.decay)?;
            f.write_u32::<LittleEndian>(self.cfg.rebuild_every_n)?;
            f.flush()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, TransitionError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;

        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGTR" {
            return Err(TransitionError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(TransitionError::Corrupt(format!("unsupported version {version}")));
        }
        let n = f.read_u32::<LittleEndian>()? as usize;

        let mut names = Vec::with_capacity(n);
        for _ in 0..n {
            let len = f.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            f.read_exact(&mut buf)?;
            names.push(String::from_utf8_lossy(&buf).into_owned());
        }
        let mut counts = vec![0.0f32; n * n];
        for c in counts.iter_mut() {
            *c = f.read_f32::<LittleEndian>()?;
        }
        let mut prior = vec![0.0f32; n * n];
        for p in prior.iter_mut() {
            *p = f.read_f32::<LittleEndian>()?;
        }
        let cfg = TransitionConfig {
            prior_strength: f.read_f32::<LittleEndian>()?,
            floor: f.read_f32::<LittleEndian>()?,
            decay: f.read_f32::<LittleEndian>()?,
            rebuild_every_n: f.read_u32::<LittleEndian>()?,
        };
        Ok(Self { names, counts, prior, cfg, events_since_rebuild: 0 })
    }
```

Note: a truncated file surfaces as `TransitionError::Io` via `read_exact`/`read_f32` — `is_err()` holds, which is what the test asserts.

- [ ] **Step 4: Run to verify the tests pass**

Run: `cargo test -p rgdb transitions:: 2>&1 | tail -8`
Expected: `test result: ok. 10 passed`.

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/transitions.rs
git commit -m "feat(rgdb): atomic sidecar persistence for TransitionStore"
```

---

### Task 3: `credit()` — the forward–backward credit pass

**Files:**
- Create: `rgdb/src/credit.rs`
- Modify: `rgdb/src/lib.rs` (add `pub mod credit;`)

**Interfaces:**
- Consumes: `Graph`, `NodeId`, `RelationId`, `PropagationParams`, `RelationVocab`, `propagate` (for the invariant test).
- Produces (used by Task 4):
  - `credit(graph, vocab, seeds: &[(NodeId,f32)], query_relation: Option<RelationId>, target: NodeId, params: &PropagationParams) -> Vec<(RelationId, RelationId, f32)>` — L1-normalized; empty if unreachable.
  - `backward_mass_at_target(graph, vocab, seeds, query_relation, target, params) -> f32` — diagnostic; equals `propagate(...)[target]` exactly when `min_intensity == 0.0`.

**Hand-computed expectations** (defaults: `reflection = 0.85`, `refraction_index = 1.0`, attenuation 0 so `p = 1`, hop weight `w = 0.85 · sim`):
- Chain `0→1→2→3` (all relation 0, uniform vocab), seed 0, target 3: every hop stays on relation 0, so all credit lands on the diagonal → `[(0,0,1.0)]`. `backward_mass_at_target == 0.614125` (= `0.85³`), matching the kernel's known chain value.
- Two-relation `0 -(A=0)-> 1 -(B=1)-> 2` with `sim(A,B) = 0.5`, seed 0, `query_relation = A`, target 2: `flow(A→A) = 1·0.85·B₃[(1,A)] = 0.36125`, `flow(A→B) = 0.85·0.425·1 = 0.36125` → credits `(0,0) = 0.5`, `(0,1) = 0.5`. `backward_mass_at_target == 0.36125`, matching the kernel's known refraction value.

- [ ] **Step 1: Write the failing tests**

Create `rgdb/src/credit.rs` containing ONLY this test module for now:

```rust
//! Backward credit pass: attribute a rewarded answer to relation transitions.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::{propagate, PropagationParams};
    use crate::relation::RelationVocab;

    fn exact() -> PropagationParams {
        // The forward/backward invariant is exact only without pruning.
        PropagationParams { max_depth: 4, min_intensity: 0.0 }
    }

    fn chain() -> Graph {
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1)], vec![e(2)], vec![e(3)], vec![]];
        Graph::from_adjacency(4, adj, NodeProps::default()).unwrap()
    }

    /// 0 -(A)-> 1 -(B)-> 2, with sim(A,B) = 0.5
    fn two_relation() -> (Graph, RelationVocab) {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::new(vec!["A".into(), "B".into()], vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        (g, v)
    }

    #[test]
    fn credits_are_l1_normalized() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let c = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact());
        let total: f32 = c.iter().map(|&(_, _, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn chain_credits_only_the_diagonal() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let c = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact());
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].0, c[0].1), (0, 0));
        assert!((c[0].2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn two_relation_splits_credit_evenly() {
        let (g, v) = two_relation();
        let mut c = credit(&g, &v, &[(0, 1.0)], Some(0), 2, &exact());
        c.sort_by_key(|&(a, b, _)| (a, b));
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].0, c[0].1), (0, 0));
        assert!((c[0].2 - 0.5).abs() < 1e-5, "diag got {}", c[0].2);
        assert_eq!((c[1].0, c[1].1), (0, 1));
        assert!((c[1].2 - 0.5).abs() < 1e-5, "off-diag got {}", c[1].2);
    }

    #[test]
    fn unreachable_target_yields_no_credit() {
        // node 3 has no path from node 2 in this 4-node graph with only 0->1
        let e = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(4, vec![vec![e], vec![], vec![], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        assert!(credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact()).is_empty());
    }

    #[test]
    fn untyped_query_skips_the_first_hop_transition() {
        // query_relation = None => the single hop has r_in = None => no transition
        let e = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(2, vec![vec![e], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        assert!(credit(&g, &v, &[(0, 1.0)], None, 1, &exact()).is_empty());
    }

    // THE cross-check: backward value at the seed == forward mass at the target.
    #[test]
    fn backward_matches_forward_on_chain() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&3).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 3, &p);
        assert!((fwd - 0.614125).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_with_refraction() {
        let (g, v) = two_relation();
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&2).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 2, &p);
        assert!((fwd - 0.36125).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_on_multipath() {
        // 0 -> 1 -> 2 and 0 -> 2 (two paths to node 2)
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![e(1), e(2)], vec![e(2)], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&2).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 2, &p);
        assert!((fwd - 0.78625).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rgdb credit:: 2>&1 | tail -5`
Expected: compile error — `cannot find function credit`.

- [ ] **Step 3: Implement the credit pass**

Prepend to `rgdb/src/credit.rs` (above the test module):

```rust
use hashbrown::{HashMap, HashSet};

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::PropagationParams;
use crate::relation::RelationVocab;

/// A propagation state: a node, plus the relation of the edge we arrived by.
/// `None` only on the seed of an untyped query.
type State = (NodeId, Option<RelationId>);

/// Σ over `u`'s out-edges of `(1 - attenuation)`, memoized.
fn out_weight_sum(graph: &Graph, u: NodeId, cache: &mut HashMap<NodeId, f32>) -> f32 {
    if let Some(&d) = cache.get(&u) {
        return d;
    }
    let mut s = 0.0f32;
    for (_v, ep) in graph.neighbors(u) {
        s += (1.0 - ep.attenuation).max(0.0);
    }
    cache.insert(u, s);
    s
}

/// Weight of the hop `(u, r_in) --[r_out]--> v`, identical to the forward kernel's.
fn hop_weight(
    graph: &Graph,
    vocab: &RelationVocab,
    u: NodeId,
    r_in: Option<RelationId>,
    r_out: RelationId,
    base: f32,
    denom: f32,
) -> f32 {
    if base <= 0.0 || denom <= 0.0 {
        return 0.0;
    }
    let props = graph.node_props()[u as usize];
    if props.reflection <= 0.0 {
        return 0.0;
    }
    let p = base / denom;
    let sim = match r_in {
        Some(a) => vocab.similarity(a, r_out),
        None => 1.0,
    };
    let sim_term = if props.refraction_index == 1.0 { sim } else { sim.powf(props.refraction_index) };
    props.reflection * p * sim_term
}

/// `F[k]` = mass at each state after exactly `k` hops. Mirrors the forward kernel.
fn forward(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
    denom_cache: &mut HashMap<NodeId, f32>,
) -> Vec<HashMap<State, f32>> {
    let d = params.max_depth;
    let n = graph.num_nodes();
    let mut f: Vec<HashMap<State, f32>> = vec![HashMap::new(); d + 1];

    for &(s, m) in seeds {
        if (s as usize) < n && m >= params.min_intensity && m > 0.0 {
            *f[0].entry((s, query_relation)).or_insert(0.0) += m;
        }
    }

    for k in 0..d {
        let current: Vec<(State, f32)> = f[k].iter().map(|(&s, &m)| (s, m)).collect();
        for ((u, r_in), mass) in current {
            if mass <= 0.0 || mass < params.min_intensity {
                continue;
            }
            let denom = out_weight_sum(graph, u, denom_cache);
            if denom <= 0.0 {
                continue;
            }
            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                let w = hop_weight(graph, vocab, u, r_in, ep.relation, base, denom);
                let t = mass * w;
                if t <= 0.0 || t < params.min_intensity {
                    continue;
                }
                *f[k + 1].entry((v, Some(ep.relation))).or_insert(0.0) += t;
            }
        }
    }
    f
}

/// `B[j][(v,r)]` = total weight of continuations from `(v,r)` that arrive at `target`
/// within `j` more hops. Backward in *depth budget*, forward in *graph direction* —
/// so this needs only out-adjacency, never a reverse index.
fn backward(
    graph: &Graph,
    vocab: &RelationVocab,
    target: NodeId,
    ball: &[State],
    params: &PropagationParams,
    denom_cache: &mut HashMap<NodeId, f32>,
) -> Vec<HashMap<State, f32>> {
    let d = params.max_depth;
    let mut b: Vec<HashMap<State, f32>> = vec![HashMap::new(); d + 1];

    for &st in ball {
        b[0].insert(st, if st.0 == target { 1.0 } else { 0.0 });
    }

    for j in 1..=d {
        let mut cur: HashMap<State, f32> = HashMap::with_capacity(ball.len());
        for &(v, r) in ball {
            let mut acc = if v == target { 1.0 } else { 0.0 };
            let denom = out_weight_sum(graph, v, denom_cache);
            if denom > 0.0 {
                for (x, ep) in graph.neighbors(v) {
                    let base = (1.0 - ep.attenuation).max(0.0);
                    let w = hop_weight(graph, vocab, v, r, ep.relation, base, denom);
                    if w <= 0.0 {
                        continue;
                    }
                    if let Some(&bx) = b[j - 1].get(&(x, Some(ep.relation))) {
                        acc += w * bx;
                    }
                }
            }
            cur.insert((v, r), acc);
        }
        b[j] = cur;
    }
    b
}

fn ball_of(f: &[HashMap<State, f32>]) -> Vec<State> {
    let mut set: HashSet<State> = HashSet::new();
    for m in f {
        for k in m.keys() {
            set.insert(*k);
        }
    }
    set.into_iter().collect()
}

/// L1-normalized credit per `(r_in, r_out)` transition for a rewarded `target`.
/// Empty when the target is unreachable within `max_depth`.
pub fn credit(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    target: NodeId,
    params: &PropagationParams,
) -> Vec<(RelationId, RelationId, f32)> {
    let d = params.max_depth;
    if (target as usize) >= graph.num_nodes() || d == 0 {
        return Vec::new();
    }

    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();
    let f = forward(graph, vocab, seeds, query_relation, params, &mut denom_cache);
    let ball = ball_of(&f);
    let b = backward(graph, vocab, target, &ball, params, &mut denom_cache);

    let mut flow: HashMap<(RelationId, RelationId), f32> = HashMap::new();
    for k in 0..d {
        let budget = d - k - 1;
        let states: Vec<(State, f32)> = f[k].iter().map(|(&s, &m)| (s, m)).collect();
        for ((u, r_in), mass) in states {
            if mass <= 0.0 {
                continue;
            }
            // An untyped first hop has no source relation, so it credits nothing.
            let r_from = match r_in {
                Some(a) => a,
                None => continue,
            };
            let denom = out_weight_sum(graph, u, &mut denom_cache);
            if denom <= 0.0 {
                continue;
            }
            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                let w = hop_weight(graph, vocab, u, r_in, ep.relation, base, denom);
                if w <= 0.0 {
                    continue;
                }
                let bv = b[budget].get(&(v, Some(ep.relation))).copied().unwrap_or(0.0);
                if bv <= 0.0 {
                    continue;
                }
                *flow.entry((r_from, ep.relation)).or_insert(0.0) += mass * w * bv;
            }
        }
    }

    let total: f32 = flow.values().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    flow.into_iter().map(|((a, bb), v)| (a, bb, v / total)).collect()
}

/// Diagnostic: `Σ_seeds seed_mass · B_maxdepth[(seed, query_relation)]`.
/// Equals `propagate(...)[target]` exactly when `params.min_intensity == 0.0`.
pub fn backward_mass_at_target(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    target: NodeId,
    params: &PropagationParams,
) -> f32 {
    let d = params.max_depth;
    if (target as usize) >= graph.num_nodes() {
        return 0.0;
    }
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();
    let f = forward(graph, vocab, seeds, query_relation, params, &mut denom_cache);
    let ball = ball_of(&f);
    let b = backward(graph, vocab, target, &ball, params, &mut denom_cache);

    seeds
        .iter()
        .map(|&(s, m)| m * b[d].get(&(s, query_relation)).copied().unwrap_or(0.0))
        .sum()
}
```

- [ ] **Step 4: Register the module**

In `rgdb/src/lib.rs`, add after `pub mod transitions;`:

```rust
pub mod credit;
```

- [ ] **Step 5: Run to verify the tests pass**

Run: `cargo test -p rgdb credit:: 2>&1 | tail -12`
Expected: `test result: ok. 8 passed`. In particular the three `backward_matches_forward_*` tests must pass — they tie the new backward walk to the already-verified forward kernel.

Run: `cargo build -p rgdb 2>&1 | tail -2`
Expected: `Finished`, no warnings.

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/credit.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): exact forward-backward credit pass (no reverse CSR needed)"
```

---

### Task 4: `RgdbEngine` — the database surface

**Files:**
- Modify: `rgdb/Cargo.toml` (add `arc-swap`, `lru`)
- Modify: `rgdb/src/relation.rs` (add `names()`, `matrix()` accessors)
- Create: `rgdb/src/engine.rs`
- Modify: `rgdb/src/lib.rs` (add `pub mod engine;` + re-exports)

**Interfaces:**
- Consumes: `TransitionStore`/`TransitionConfig`/`TransitionError` (Task 1–2), `credit()` (Task 3), `propagate`, `Graph`, `RelationVocab`.
- Produces (used by Task 5):
  - `pub type QueryId = u64;`
  - `QueryResult { ranked: Vec<(NodeId, f32)>, query_id: QueryId }`
  - `EngineConfig { cache_capacity: usize, cache_ttl: Duration }` + `Default` (4096, 3600s)
  - `FeedbackError` (`UnknownQuery`, `InvalidTarget`, `TargetUnreachable`, `Transition`)
  - `RgdbEngine::{new, with_engine_config, query, record_feedback, refresh, save, load, vocab, matrix}`

- [ ] **Step 1: Add dependencies**

In `rgdb/Cargo.toml`, under `[dependencies]`, add:

```toml
arc-swap = "1.7"
lru = "0.12"
```

- [ ] **Step 2: Add `RelationVocab` accessors**

In `rgdb/src/relation.rs`, add inside `impl RelationVocab`:

```rust
    /// Relation names, index == RelationId.
    pub fn names(&self) -> &[String] { &self.names }

    /// Flat row-major n*n similarity matrix.
    pub fn matrix(&self) -> &[f32] { &self.similarity }
```

- [ ] **Step 3: Write the failing tests**

Create `rgdb/src/engine.rs` containing ONLY this test module for now:

```rust
//! `RgdbEngine`: graph + live vocab + learned transitions + feedback loop.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::PropagationParams;
    use crate::relation::RelationVocab;
    use crate::transitions::TransitionConfig;

    /// 0 -(A=0)-> 1 -(B=1)-> 2
    fn engine(rebuild_every_n: u32) -> RgdbEngine {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let cfg = TransitionConfig { rebuild_every_n, ..TransitionConfig::default() };
        RgdbEngine::new(g, prior, cfg)
    }

    fn params() -> PropagationParams { PropagationParams { max_depth: 4, min_intensity: 0.0 } }

    #[test]
    fn cold_start_matrix_is_uniform() {
        let e = engine(0);
        for a in 0..2u16 {
            for b in 0..2u16 {
                assert_eq!(e.vocab().similarity(a, b), 1.0);
            }
        }
    }

    #[test]
    fn query_returns_ranked_results_and_an_id() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        assert!(r.query_id > 0);
        assert!(!r.ranked.is_empty());
        // sorted descending by score
        for w in r.ranked.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn feedback_on_unknown_query_errors() {
        let e = engine(0);
        assert!(matches!(e.record_feedback(9999, 2, 1.0), Err(FeedbackError::UnknownQuery(_))));
    }

    #[test]
    fn feedback_on_out_of_bounds_target_errors() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        assert!(matches!(e.record_feedback(r.query_id, 99, 1.0), Err(FeedbackError::InvalidTarget(_))));
    }

    #[test]
    fn feedback_on_unreachable_target_errors() {
        // seed at node 2 (a sink): nothing is reachable
        let e = engine(0);
        let r = e.query(&[(2, 1.0)], Some(0), &params());
        assert!(matches!(e.record_feedback(r.query_id, 0, 1.0), Err(FeedbackError::TargetUnreachable(_))));
    }

    #[test]
    fn feedback_then_refresh_changes_the_matrix() {
        let e = engine(0); // manual refresh only
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        // not refreshed yet -> still uniform
        assert_eq!(e.vocab().similarity(0, 1), 1.0);
        e.refresh();
        // A->B was credited; A->A (diagonal) stays pinned
        assert_eq!(e.vocab().similarity(0, 0), 1.0);
        assert_eq!(e.vocab().similarity(0, 1), 1.0, "credited transition becomes the row max");
        // B has no outgoing evidence, so its row stays uniform
        assert_eq!(e.vocab().similarity(1, 0), 1.0);
    }

    #[test]
    fn auto_refresh_fires_at_rebuild_every_n() {
        let e = engine(1); // refresh after every event
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        assert_eq!(e.events_since_rebuild(), 0, "auto-refresh reset the counter");
    }

    #[test]
    fn save_load_roundtrip_preserves_learning() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        let path = "test_engine_roundtrip.transitions";
        e.save(path).unwrap();

        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let e2 = RgdbEngine::load(g, path).unwrap();
        assert_eq!(e2.counts_snapshot(), e.counts_snapshot());
        let _ = std::fs::remove_file(path);
    }
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `cargo test -p rgdb engine:: 2>&1 | tail -5`
Expected: compile error — `cannot find RgdbEngine`.

- [ ] **Step 5: Implement the engine**

Prepend to `rgdb/src/engine.rs` (above the test module):

```rust
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use lru::LruCache;
use thiserror::Error;

use crate::credit::credit;
use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate, PropagationParams};
use crate::relation::RelationVocab;
use crate::transitions::{TransitionConfig, TransitionError, TransitionStore};

pub type QueryId = u64;

#[derive(Debug, Error)]
pub enum FeedbackError {
    #[error("unknown or expired query id {0}")]
    UnknownQuery(QueryId),
    #[error("target node {0} is out of bounds")]
    InvalidTarget(NodeId),
    #[error("target node {0} is unreachable from the query's seeds within max_depth")]
    TargetUnreachable(NodeId),
    #[error(transparent)]
    Transition(#[from] TransitionError),
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub ranked: Vec<(NodeId, f32)>,
    pub query_id: QueryId,
}

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub cache_capacity: usize,
    pub cache_ttl: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { cache_capacity: 4096, cache_ttl: Duration::from_secs(3600) }
    }
}

/// Everything `credit()` needs to attribute a later feedback event, including the
/// vocab that actually produced the ranking (a `refresh()` may swap the live one).
#[derive(Clone)]
struct QueryContext {
    seeds: Vec<(NodeId, f32)>,
    query_relation: Option<RelationId>,
    params: PropagationParams,
    vocab: Arc<RelationVocab>,
    created: Instant,
}

pub struct RgdbEngine {
    graph: Graph,
    vocab: ArcSwap<RelationVocab>,
    store: Mutex<TransitionStore>,
    cache: Mutex<LruCache<QueryId, QueryContext>>,
    next_id: AtomicU64,
    engine_cfg: EngineConfig,
}

impl RgdbEngine {
    /// `vocab_prior` supplies both the relation names and the prior matrix.
    /// Use `RelationVocab::with_names_uniform(names)` for the do-no-harm default.
    pub fn new(graph: Graph, vocab_prior: RelationVocab, cfg: TransitionConfig) -> Self {
        Self::with_engine_config(graph, vocab_prior, cfg, EngineConfig::default())
    }

    pub fn with_engine_config(
        graph: Graph,
        vocab_prior: RelationVocab,
        cfg: TransitionConfig,
        engine_cfg: EngineConfig,
    ) -> Self {
        let store = TransitionStore::new(
            vocab_prior.names().to_vec(),
            Some(vocab_prior.matrix().to_vec()),
            cfg,
        )
        .expect("prior came from a RelationVocab, so its shape is n*n");
        let initial = store.derive_vocab();
        Self::assemble(graph, store, initial, engine_cfg)
    }

    fn assemble(graph: Graph, store: TransitionStore, vocab: RelationVocab, engine_cfg: EngineConfig) -> Self {
        let cap = NonZeroUsize::new(engine_cfg.cache_capacity.max(1)).unwrap();
        Self {
            graph,
            vocab: ArcSwap::from_pointee(vocab),
            store: Mutex::new(store),
            cache: Mutex::new(LruCache::new(cap)),
            next_id: AtomicU64::new(1),
            engine_cfg,
        }
    }

    /// Restore learned state from a sidecar, validating relation coverage.
    pub fn load(graph: Graph, path: &str) -> Result<Self, TransitionError> {
        let store = TransitionStore::load(path)?;
        let graph_max = graph.edge_props().iter().map(|e| e.relation as usize).max().unwrap_or(0);
        if store.len() <= graph_max {
            return Err(TransitionError::RelationCoverage { store_len: store.len(), graph_max });
        }
        let vocab = store.derive_vocab();
        Ok(Self::assemble(graph, store, vocab, EngineConfig::default()))
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    pub fn vocab(&self) -> Arc<RelationVocab> { self.vocab.load_full() }
    pub fn matrix(&self) -> Vec<f32> { self.vocab().matrix().to_vec() }
    pub fn counts_snapshot(&self) -> Vec<f32> { self.store.lock().unwrap().counts().to_vec() }
    pub fn events_since_rebuild(&self) -> u32 { self.store.lock().unwrap().events_since_rebuild() }

    pub fn query(
        &self,
        seeds: &[(NodeId, f32)],
        query_relation: Option<RelationId>,
        params: &PropagationParams,
    ) -> QueryResult {
        let vocab = self.vocab.load_full();
        let totals = propagate(&self.graph, &vocab, seeds, query_relation, params);
        let mut ranked: Vec<(NodeId, f32)> = totals.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache.lock().unwrap().put(
            query_id,
            QueryContext {
                seeds: seeds.to_vec(),
                query_relation,
                params: *params,
                vocab,
                created: Instant::now(),
            },
        );
        QueryResult { ranked, query_id }
    }

    /// Attribute `target` back to the relation transitions that carried mass to it.
    /// `signal` is signed: negative reports a wrong answer.
    pub fn record_feedback(
        &self,
        query_id: QueryId,
        target: NodeId,
        signal: f32,
    ) -> Result<(), FeedbackError> {
        let ctx = {
            let mut cache = self.cache.lock().unwrap();
            match cache.get(&query_id) {
                Some(c) if c.created.elapsed() <= self.engine_cfg.cache_ttl => c.clone(),
                Some(_) => {
                    cache.pop(&query_id);
                    return Err(FeedbackError::UnknownQuery(query_id));
                }
                None => return Err(FeedbackError::UnknownQuery(query_id)),
            }
        };

        if (target as usize) >= self.graph.num_nodes() {
            return Err(FeedbackError::InvalidTarget(target));
        }

        // Credit under the vocab that produced the ranking, not the live one.
        let credits = credit(
            &self.graph,
            &ctx.vocab,
            &ctx.seeds,
            ctx.query_relation,
            target,
            &ctx.params,
        );
        if credits.is_empty() {
            return Err(FeedbackError::TargetUnreachable(target));
        }

        let should_refresh = {
            let mut store = self.store.lock().unwrap();
            store.record(&credits, signal);
            let n = store.config().rebuild_every_n;
            n > 0 && store.events_since_rebuild() >= n
        };
        if should_refresh {
            self.refresh();
        }
        Ok(())
    }

    /// Rebuild the similarity matrix from counts and atomically swap it in.
    pub fn refresh(&self) {
        let vocab = { self.store.lock().unwrap().rebuild() };
        self.vocab.store(Arc::new(vocab));
    }

    pub fn save(&self, path: &str) -> Result<(), TransitionError> {
        self.store.lock().unwrap().save(path)
    }
}
```

- [ ] **Step 6: Register the module**

In `rgdb/src/lib.rs`, add after `pub mod credit;`:

```rust
pub mod engine;
```

and add to the re-export list at the bottom:

```rust
pub use engine::{EngineConfig, FeedbackError, QueryId, QueryResult, RgdbEngine};
pub use transitions::{TransitionConfig, TransitionError, TransitionStore};
```

- [ ] **Step 7: Run to verify the tests pass**

Run: `cargo test -p rgdb engine:: 2>&1 | tail -12`
Expected: `test result: ok. 8 passed`.

Run: `cargo test -p rgdb 2>&1 | grep "test result" | head -1`
Expected: all green (27 pre-existing + the new ones).

Run: `cargo build -p rgdb 2>&1 | tail -2`
Expected: `Finished`, no warnings.

- [ ] **Step 8: Commit**

```bash
git add rgdb/Cargo.toml rgdb/src/relation.rs rgdb/src/engine.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): RgdbEngine with feedback loop, atomic vocab swap, sidecar persistence"
```

---

### Task 5: Python bindings for the engine

**Files:**
- Modify: `rgdb-python/src/lib.rs`
- Create: `rgdb-eval/scripts/smoke_engine.py`

**Interfaces:**
- Consumes: `RgdbEngine`, `TransitionConfig`, `PropagationParams`, `RelationVocab` (Task 4).
- Produces (used by Task 6), on the Python module `rgdb_embeddings._rgdb_core`:
  - `Engine(graph, vocab, prior_strength=10.0, floor=0.05, decay=1.0, rebuild_every_n=64)`
  - `engine.query(seeds, query_relation=None, max_depth=4, min_intensity=1e-3) -> (list[(int,float)], int)`
  - `engine.record_feedback(query_id, target, signal=1.0) -> None` (raises `ValueError` on error)
  - `engine.refresh() -> None`
  - `engine.matrix() -> list[float]` (flat n*n)
  - `engine.save(path)` / `Engine.load(graph, path)`

- [ ] **Step 1: Add the `Engine` pyclass**

In `rgdb-python/src/lib.rs`, add these imports near the top:

```rust
use rgdb::engine::{RgdbEngine, QueryId};
use rgdb::transitions::TransitionConfig;
```

Then add this pyclass (place it after `PyVocab`, before `#[pymodule]`):

```rust
#[pyclass(name = "Engine")]
struct PyEngine { inner: RgdbEngine }

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (graph, vocab, prior_strength=10.0, floor=0.05, decay=1.0, rebuild_every_n=64))]
    fn new(graph: &PyGraph, vocab: &PyVocab, prior_strength: f32, floor: f32, decay: f32, rebuild_every_n: u32) -> Self {
        let cfg = TransitionConfig { prior_strength, floor, decay, rebuild_every_n };
        PyEngine { inner: RgdbEngine::new(graph.inner.clone(), vocab.inner.clone(), cfg) }
    }

    #[staticmethod]
    fn load(graph: &PyGraph, path: &str) -> PyResult<Self> {
        RgdbEngine::load(graph.inner.clone(), path)
            .map(|inner| PyEngine { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (seeds, query_relation=None, max_depth=4, min_intensity=1e-3))]
    fn query(&self, seeds: Vec<(u32, f32)>, query_relation: Option<u16>, max_depth: usize, min_intensity: f32) -> (Vec<(u32, f32)>, u64) {
        let params = PropagationParams { max_depth, min_intensity };
        let r = self.inner.query(&seeds, query_relation, &params);
        (r.ranked, r.query_id)
    }

    #[pyo3(signature = (query_id, target, signal=1.0))]
    fn record_feedback(&self, query_id: u64, target: u32, signal: f32) -> PyResult<()> {
        self.inner
            .record_feedback(query_id as QueryId, target, signal)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn refresh(&self) { self.inner.refresh(); }
    fn matrix(&self) -> Vec<f32> { self.inner.matrix() }
    fn counts(&self) -> Vec<f32> { self.inner.counts_snapshot() }
    fn events_since_rebuild(&self) -> u32 { self.inner.events_since_rebuild() }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }
}
```

`PyVocab` must derive `Clone` on its inner `RelationVocab` (it already does) and `PyGraph`'s `inner: Graph` is `Clone`.

Register it in `#[pymodule] fn _rgdb_core`:

```rust
    m.add_class::<PyEngine>()?;
```

- [ ] **Step 2: Write the smoke script**

Create `rgdb-eval/scripts/smoke_engine.py`:

```python
"""Exercise the learning loop end-to-end through the native bindings."""
from rgdb_embeddings import _rgdb_core as core


def main() -> None:
    # 0 -(A=0)-> 1 -(B=1)-> 2
    g = core.build_graph(3, [[(1, 0.0, 0)], [(2, 0.0, 1)], []])
    vocab = core.vocab_from_matrix(["A", "B"], [1.0, 1.0, 1.0, 1.0])  # uniform prior
    eng = core.Engine(g, vocab, rebuild_every_n=0)  # manual refresh

    # cold start: matrix must be exactly all-ones (typed PPR, do no harm)
    assert eng.matrix() == [1.0, 1.0, 1.0, 1.0], eng.matrix()

    ranked, qid = eng.query([(0, 1.0)], 0, 4, 0.0)
    assert ranked and qid > 0, (ranked, qid)

    eng.record_feedback(qid, 2, 100.0)
    eng.refresh()

    m = eng.matrix()
    assert m[0] == 1.0, "diagonal pinned"
    assert m[3] == 1.0, "diagonal pinned"

    # unknown query id must raise, never silently drop
    try:
        eng.record_feedback(999999, 2, 1.0)
        raise AssertionError("expected ValueError for unknown query id")
    except ValueError:
        pass

    print("engine smoke ok; matrix:", [round(x, 4) for x in m])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Build and run**

Run:
```bash
.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml
.venv/Scripts/python.exe rgdb-eval/scripts/smoke_engine.py
```
Expected: `engine smoke ok; matrix: [1.0, ...]` with no assertion failures.

- [ ] **Step 4: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/scripts/smoke_engine.py
git commit -m "feat(bindings): expose RgdbEngine (query/record_feedback/refresh/save)"
```

---

### Task 6: MetaQA online-learning acceptance experiment

**Files:**
- Create: `rgdb-eval/scripts/experiment_online_learning.py`
- Create: `rgdb-eval/results/metaqa-online-learning.md` (generated)

**Interfaces:**
- Consumes: `core.Engine` (Task 5), `load_kb`/`load_questions` (`rgdb_eval.metaqa`), `evaluate`/`to_markdown` (`rgdb_eval.report`), `PPRRanker`, `NewRgdbRanker`.
- Produces: a committed results table comparing the **online-learned** matrix against the uniform cold start and the **offline-trained** reference.

**Background:** This validates the whole loop against numbers already independently measured. Reference points from `results/metaqa-trained-refraction.md`: 2-hop MRR — uniform 0.211, offline-trained **0.275**; 1-hop MRR — uniform 0.973, offline-trained 0.992.

**Honest acceptance:** the replay credits transitions along *all* seed→answer paths, not only the gold chain, so it need not reproduce the offline number exactly. The hard assertion is directional; the exact figures are reported for comparison.

- [ ] **Step 1: Write the experiment**

Create `rgdb-eval/scripts/experiment_online_learning.py`:

```python
"""Acceptance: does the online feedback loop learn what offline training learned?

Replays MetaQA *training* questions through the engine as feedback events
(query -> record_feedback(gold answer)), then evaluates the resulting matrix on
the *test* questions. Compares against the uniform cold start and the offline
trained matrix.

Run: .venv/Scripts/python.exe rgdb-eval/scripts/experiment_online_learning.py
"""
from __future__ import annotations
import os
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.rankers.rgdb_new import NewRgdbRanker
from rgdb_eval.report import evaluate, to_markdown

DATA = "data/MetaQA"
TRAIN_EVENTS = 4000   # feedback events to replay
TEST_LIMIT = 1000     # test questions per hop


def replay_feedback(graph) -> list[float]:
    """Drive the engine with training questions; return the learned flat matrix."""
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)

    n = len(graph.relations)
    uniform = core.vocab_from_matrix(list(graph.relations), [1.0] * (n * n))
    eng = core.Engine(g, uniform, rebuild_every_n=0)  # manual refresh at the end

    # cold start must be exactly uniform
    assert all(abs(x - 1.0) < 1e-6 for x in eng.matrix()), "cold start not uniform"

    events = 0
    skipped = 0
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_train_{hop}hop.txt")
        qtype = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(qpath):
            continue
        per_hop = TRAIN_EVENTS // 3
        for q in load_questions(qpath, hop, graph, limit=per_hop, qtype_path=qtype):
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            _, qid = eng.query([(q.topic_id, 1.0)], rel, 4, 1e-4)
            try:
                eng.record_feedback(qid, q.answer_ids[0], 1.0)
                events += 1
            except ValueError:
                skipped += 1  # unreachable within max_depth
    eng.refresh()
    print(f"replayed {events} feedback events ({skipped} skipped as unreachable)")
    return eng.matrix()


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {len(graph.relations)} relations")

    learned = replay_feedback(graph)
    n = len(graph.relations)
    learned_np = np.asarray(learned, dtype=np.float32).reshape(n, n)
    off_diag = learned_np[~np.eye(n, dtype=bool)]
    print(f"learned matrix: off-diagonal mean {off_diag.mean():.3f}, min {off_diag.min():.3f}")

    questions = []
    for hop in (1, 2, 3):
        qpath = os.path.join(DATA, f"qa_test_{hop}hop.txt")
        qtype = os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt")
        if os.path.exists(qpath):
            questions += load_questions(qpath, hop, graph, limit=TEST_LIMIT, qtype_path=qtype)

    contenders = [
        PPRRanker(graph),
        NewRgdbRanker(graph, vocab_mode="uniform"),
        NewRgdbRanker(graph, sim_matrix=learned_np, name="rgdb-new-online-learned"),
    ]
    rows = []
    for c in contenders:
        rows.append(evaluate(c, questions))
        print(f"  scored {c.name}")

    md = "# Online-learning acceptance (MetaQA)\n" + to_markdown(rows)
    out = "results/metaqa-online-learning.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}\n")
    print(md)

    # Directional acceptance: the loop must beat the uniform cold start at 2-hop.
    by_name = {r["ranker"]: r for r in rows}
    uni2 = by_name["rgdb-new-uniform"]["hop2"]["mrr"]
    on2 = by_name["rgdb-new-online-learned"]["hop2"]["mrr"]
    on1 = by_name["rgdb-new-online-learned"]["hop1"]["mrr"]
    print(f"2-hop MRR: uniform {uni2:.3f} -> online-learned {on2:.3f} (offline reference 0.275)")
    print(f"1-hop MRR: online-learned {on1:.3f} (offline reference 0.992)")
    assert on2 > uni2, f"online learning did not improve 2-hop MRR ({on2:.3f} <= {uni2:.3f})"
    assert on1 >= 0.90, f"online learning damaged 1-hop MRR ({on1:.3f})"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run:
```bash
cd rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_online_learning.py 2>&1 | tail -30
```
Expected: prints the replay count, the learned matrix summary, a per-hop table, and both assertions pass. `2-hop MRR: uniform 0.211 -> online-learned <X>`.

If either assertion fails, that is a **finding, not a plan failure**: record the actual numbers in the results file and report them rather than weakening the assertion.

- [ ] **Step 3: Append the verdict**

Append to `rgdb-eval/results/metaqa-online-learning.md` a short paragraph stating, with the actual numbers: whether the online loop improved 2-hop MRR over the uniform cold start, how close it came to the offline-trained 0.275, and whether the 1-hop win survived. Note the credit pass rewards all seed→answer paths (not just gold chains), which is why online and offline need not match exactly.

- [ ] **Step 4: Commit**

```bash
git add rgdb-eval/scripts/experiment_online_learning.py rgdb-eval/results/metaqa-online-learning.md
git commit -m "eval: online-learning acceptance experiment (feedback replay vs offline-trained)"
```

---

## Self-Review

**Spec coverage:**
- `TransitionStore` (counts, off-diagonal row-max derivation, pinned diagonal, floor, signed+clamped signals, decay, event counter) → Task 1. ✓
- Sidecar persistence, atomic temp+rename, corrupt/truncated rejection → Task 2. ✓
- `credit()` forward–backward, L1 normalization, untyped-hop skip, unreachable→empty, the `min_intensity == 0.0` invariant, **no reverse CSR / no `Graph` change** → Task 3. ✓
- `RgdbEngine` (ArcSwap vocab, Mutex store, LRU+TTL cache, `QueryId`, credit under the query-time vocab, auto-refresh at `rebuild_every_n`, relation-coverage load check, all four `FeedbackError` variants) → Task 4. ✓
- `RelationVocab::{names, matrix}` accessors → Task 4 Step 2. ✓
- Python bindings → Task 5. ✓
- MetaQA replay acceptance → Task 6. ✓
- Cold start is exactly uniform → asserted in Task 1 (`cold_start_is_exactly_uniform`), Task 4 (`cold_start_matrix_is_uniform`), Task 5 (smoke), and Task 6 (replay guard). ✓
- The diagonal-credit regression → Task 1 (`diagonal_credit_does_not_shrink_off_diagonals`). ✓

**Placeholder scan:** none. Every step carries complete code and exact commands. Task 6's assertions are directional by design, with the failure mode explicitly documented as a finding to report rather than a threshold to weaken.

**Type consistency:**
- `TransitionConfig` field names identical across Tasks 1, 2, 4, 5. ✓
- `credit(graph, vocab, seeds, query_relation, target, params) -> Vec<(RelationId, RelationId, f32)>` — defined Task 3, consumed Task 4. ✓
- `TransitionStore::{new, record, derive_vocab, rebuild, save, load, counts, names, config, events_since_rebuild}` — defined Tasks 1–2, consumed Task 4. ✓
- `RelationVocab::{names, matrix}` — added Task 4 Step 2, used by `RgdbEngine::new` in the same task. ✓
- `RgdbEngine::{query, record_feedback, refresh, save, load, matrix, counts_snapshot, events_since_rebuild}` — defined Task 4, consumed Tasks 5–6. ✓
- Python `Engine.{query, record_feedback, refresh, matrix, counts, save, load}` — defined Task 5, consumed Task 6. ✓
- `NewRgdbRanker(graph, sim_matrix=..., name=...)` — the `sim_matrix`/`name` params already exist (added during the trained-refraction experiment). ✓
