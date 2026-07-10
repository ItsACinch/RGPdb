# Depth-Aware Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-hop scoring coefficients `c_k` to RGDB propagation so a node's score is `Σ_k c_k · I_k(v)` (mass weighted by arrival depth), turning multi-hop retrieval from a regression into a win while reproducing today's behavior bit-for-bit by default.

**Architecture:** A validated `DepthWeights` newtype carried inside `PropagationParams`. The kernel weights the *readout* (`totals`) by arrival depth while leaving *flow* (`next` frontier) and *pruning* on raw mass. The backward credit pass is generalized to exact-remaining-length so credit attributes exactly the paths that contribute to the weighted score. The engine gains a config default; Python bindings gain a `depth_weights` kwarg.

**Tech Stack:** Rust (rgdb crate + rgdb-python PyO3 bindings), Python eval harness (rgdb-eval), MetaQA dataset.

**Spec:** `docs/superpowers/specs/2026-07-10-depth-aware-scoring-design.md`

**Branch:** extends `feature/rgdb-self-learning-loop` (already checked out). Do NOT merge or push; all work lands on this branch for the user to review.

## Global Constraints

- **`c = uniform` must be bit-identical to today.** With all-ones weights, `1.0 * x == x` under IEEE-754, so `propagate()` output must be *exactly* equal to the current kernel's, not approximately. Assert exact equality in tests.
- **Readout is weighted; flow is not; pruning tests raw flow.** `totals[v] += c[depth+1] * transmitted`. The `next` frontier accumulates raw `transmitted`. The `transmitted < min_intensity` and `mass < min_intensity` guards test the unweighted value. Weighting flow or pruning would make `terminal(k)` reach nothing.
- **`DepthWeights` length is always `max_depth + 1`**, indexed by arrival depth (index 0 = seed mass).
- **Non-negative coefficients only.** `from_vec` rejects negative, non-finite, and all-zero vectors. Downstream (`intensity_to_distance`'s `-ln`, RAG fusion, credit→counts) assumes scores ≥ 0.
- **Out-of-the-box engine behavior is unchanged.** The `EngineConfig` default is `uniform`, applied only when its length matches the query's `max_depth + 1`; otherwise the query runs uniform.
- **Fallible operations return `Result`, not `assert!()`** (existing crate convention).
- **Regression floor:** the existing 63 rgdb tests and 20 rgdb-eval tests must stay green after every task.

---

### Task 1: `DepthWeights` newtype and validation

**Files:**
- Create: `rgdb/src/depth_weights.rs`
- Modify: `rgdb/src/lib.rs` (add module + re-export)

**Interfaces:**
- Produces: `DepthWeights` with `uniform(max_depth: usize) -> Self`, `terminal(max_depth: usize, k: usize) -> Result<Self, DepthWeightsError>`, `from_vec(v: Vec<f32>, max_depth: usize) -> Result<Self, DepthWeightsError>`, `as_slice(&self) -> &[f32]`. And `DepthWeightsError` enum. Both re-exported at crate root via `rgdb::depth_weights::*` and `rgdb::{DepthWeights, DepthWeightsError}`.

- [ ] **Step 1: Write the failing tests**

Create `rgdb/src/depth_weights.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_all_ones_length_max_depth_plus_one() {
        let w = DepthWeights::uniform(4);
        assert_eq!(w.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn terminal_is_one_hot_at_k() {
        let w = DepthWeights::terminal(4, 3).unwrap();
        assert_eq!(w.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn terminal_beyond_max_depth_errors() {
        assert_eq!(
            DepthWeights::terminal(2, 3),
            Err(DepthWeightsError::TerminalExceedsMaxDepth { k: 3, max_depth: 2 })
        );
    }

    #[test]
    fn from_vec_accepts_valid() {
        let w = DepthWeights::from_vec(vec![0.0, 0.5, 1.0], 2).unwrap();
        assert_eq!(w.as_slice(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, 1.0], 4),
            Err(DepthWeightsError::WrongLength { expected: 5, got: 2 })
        );
    }

    #[test]
    fn from_vec_rejects_negative() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, -0.1, 1.0], 2),
            Err(DepthWeightsError::Negative { index: 1, value: -0.1 })
        );
    }

    #[test]
    fn from_vec_rejects_non_finite() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, f32::NAN, 1.0], 2),
            Err(DepthWeightsError::NotFinite { index: 1 })
        );
    }

    #[test]
    fn from_vec_rejects_all_zero() {
        assert_eq!(
            DepthWeights::from_vec(vec![0.0, 0.0, 0.0], 2),
            Err(DepthWeightsError::AllZero)
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rgdb && cargo test depth_weights`
Expected: FAIL to compile (`DepthWeights` / `DepthWeightsError` not found). This is the expected "red".

- [ ] **Step 3: Write the implementation**

Prepend to `rgdb/src/depth_weights.rs` (above the test module):

```rust
//! Per-hop scoring coefficients for depth-aware propagation readout.
//!
//! A node's score is `Σ_k coefficients[k] · (mass arriving at exactly k hops)`.
//! Index 0 weights the seed's own mass. Length is always `max_depth + 1`.

/// Validation error for [`DepthWeights`].
#[derive(Debug, Clone, PartialEq)]
pub enum DepthWeightsError {
    /// Vector length was not `max_depth + 1`.
    WrongLength { expected: usize, got: usize },
    /// A coefficient was negative (scores must stay >= 0 for downstream consumers).
    Negative { index: usize, value: f32 },
    /// A coefficient was NaN or infinite.
    NotFinite { index: usize },
    /// Every coefficient was zero, so nothing would ever be scored.
    AllZero,
    /// `terminal(max_depth, k)` was asked for `k > max_depth`.
    TerminalExceedsMaxDepth { k: usize, max_depth: usize },
}

impl std::fmt::Display for DepthWeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongLength { expected, got } => {
                write!(f, "depth weights must have length {expected} (max_depth+1), got {got}")
            }
            Self::Negative { index, value } => {
                write!(f, "depth weight at index {index} is negative ({value})")
            }
            Self::NotFinite { index } => {
                write!(f, "depth weight at index {index} is not finite")
            }
            Self::AllZero => write!(f, "depth weights are all zero; nothing would be scored"),
            Self::TerminalExceedsMaxDepth { k, max_depth } => {
                write!(f, "terminal depth {k} exceeds max_depth {max_depth}")
            }
        }
    }
}

impl std::error::Error for DepthWeightsError {}

/// Per-hop scoring coefficients, indexed by arrival depth (0 = seed mass).
#[derive(Debug, Clone, PartialEq)]
pub struct DepthWeights(Vec<f32>);

impl DepthWeights {
    /// All-ones: reproduces the depth-blind readout (today's behavior).
    pub fn uniform(max_depth: usize) -> Self {
        DepthWeights(vec![1.0; max_depth + 1])
    }

    /// 1.0 at arrival depth `k`, 0.0 elsewhere.
    pub fn terminal(max_depth: usize, k: usize) -> Result<Self, DepthWeightsError> {
        if k > max_depth {
            return Err(DepthWeightsError::TerminalExceedsMaxDepth { k, max_depth });
        }
        let mut v = vec![0.0; max_depth + 1];
        v[k] = 1.0;
        Ok(DepthWeights(v))
    }

    /// Validated: length `max_depth + 1`, every entry finite and >= 0, at least one > 0.
    pub fn from_vec(v: Vec<f32>, max_depth: usize) -> Result<Self, DepthWeightsError> {
        let expected = max_depth + 1;
        if v.len() != expected {
            return Err(DepthWeightsError::WrongLength { expected, got: v.len() });
        }
        let mut any_positive = false;
        for (i, &x) in v.iter().enumerate() {
            if !x.is_finite() {
                return Err(DepthWeightsError::NotFinite { index: i });
            }
            if x < 0.0 {
                return Err(DepthWeightsError::Negative { index: i, value: x });
            }
            if x > 0.0 {
                any_positive = true;
            }
        }
        if !any_positive {
            return Err(DepthWeightsError::AllZero);
        }
        Ok(DepthWeights(v))
    }

    /// The coefficient slice, length `max_depth + 1`.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}
```

Then wire it into `rgdb/src/lib.rs`. After line `pub mod propagation;` (line 2), add:

```rust
pub mod depth_weights;
```

And after the `pub use propagation::*;` line (line 22), add:

```rust
pub use depth_weights::{DepthWeights, DepthWeightsError};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rgdb && cargo test depth_weights`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/depth_weights.rs rgdb/src/lib.rs
git commit -m "feat(rgdb): DepthWeights newtype with validation"
```

---

### Task 2: Carry `depth_weights` in `PropagationParams` and weight the kernel readout

**Files:**
- Modify: `rgdb/src/propagation.rs` (struct, Default, `propagate_single`, tests)
- Modify: `rgdb/src/engine.rs:145` (`*params` → `params.clone()`; test literals)
- Modify: `rgdb/src/credit.rs` (test literals only, to compile)
- Modify: `rgdb/src/rag/query_engine.rs:82` (literal, to compile)
- Modify: `rgdb-python/src/lib.rs:81,107` (literals, to compile — real wiring is Task 5)

**Interfaces:**
- Consumes: `DepthWeights` from Task 1.
- Produces: `PropagationParams { max_depth: usize, min_intensity: f32, depth_weights: Option<DepthWeights> }` (no longer `Copy`; is `Clone`). `None` means uniform. Precondition for `propagate`/`propagate_single`: if `depth_weights` is `Some`, its length equals `max_depth + 1`.

- [ ] **Step 1: Write the failing tests**

Add these to the `tests` module in `rgdb/src/propagation.rs` (the `chain()` fixture already exists there):

```rust
    #[test]
    fn uniform_weights_are_bit_identical_to_none() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let none = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None };
        let uni = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::uniform(3)),
        };
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &none);
        let b = propagate_single(&g, &vocab, 0, 1.0, Some(0), &uni);
        // EXACT equality, not approximate: 1.0 * x == x.
        for node in 0..4u32 {
            assert_eq!(a.get(&node), b.get(&node), "node {node}");
        }
    }

    #[test]
    fn terminal_weights_score_only_the_arrival_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(3, 1).unwrap()),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // Only node 1 (arrives at depth 1) is scored; node 3 arrives at depth 3.
        assert!((t[&1] - 0.85).abs() < 1e-6, "node 1 = {}", t[&1]);
        assert_eq!(t.get(&3).copied().unwrap_or(0.0), 0.0, "node 3 scored 0 under terminal(1)");
    }

    #[test]
    fn flow_is_not_weighted_so_terminal_still_reaches_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(3, 3).unwrap()),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // Node 3 is only reachable by flowing through depth-1 and depth-2 nodes that
        // score ZERO. If flow were weighted, node 3 would be unreachable.
        assert!((t[&3] - 0.614125).abs() < 1e-5, "node 3 = {}", t[&3]);
        assert_eq!(t.get(&1).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn pruning_tests_raw_flow_not_weighted_score() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        // c_3 = 0.001 makes the weighted score of node 3 tiny (0.000614), but its raw
        // flow (0.614) is well above min_intensity, so it must NOT be pruned.
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 0.1,
            depth_weights: Some(
                crate::depth_weights::DepthWeights::from_vec(vec![1.0, 1.0, 1.0, 0.001], 3).unwrap(),
            ),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        assert!((t[&3] - 0.001 * 0.614125).abs() < 1e-7, "node 3 = {}", t[&3]);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rgdb && cargo test -p rgdb propagation 2>&1 | head -30`
Expected: FAIL to compile — `PropagationParams` has no `depth_weights` field.

- [ ] **Step 3: Change the struct and the kernel**

In `rgdb/src/propagation.rs`, add the import near the top (after `use crate::relation::RelationVocab;`):

```rust
use crate::depth_weights::DepthWeights;
```

Change the struct derive and add the field (currently lines 9-15):

```rust
/// Propagation parameters.
#[derive(Debug, Clone)]
pub struct PropagationParams {
    /// Maximum number of hops.
    pub max_depth: usize,
    /// Minimum mass to keep propagating (also prunes tiny contributions).
    pub min_intensity: f32,
    /// Per-hop scoring coefficients (index = arrival depth, 0 = seed). `None` =
    /// uniform (all-ones). When `Some`, length MUST equal `max_depth + 1`.
    pub depth_weights: Option<DepthWeights>,
}
```

Update the `Default` impl (currently lines 17-21):

```rust
impl Default for PropagationParams {
    fn default() -> Self {
        Self { max_depth: 4, min_intensity: 1e-3, depth_weights: None }
    }
}
```

In `propagate_single`, hoist the coefficient slice and add a debug guard. Immediately after the early-return guard (current line 60, before `*totals.entry(seed)...`), insert:

```rust
    let dw = params.depth_weights.as_ref().map(|w| w.as_slice());
    debug_assert!(
        dw.map_or(true, |c| c.len() == params.max_depth + 1),
        "depth_weights length must equal max_depth + 1"
    );
    let weight_at = |depth: usize| -> f32 { dw.map_or(1.0, |c| c[depth]) };
```

Change the seed insertion (current line 62) to weight it by `c[0]`:

```rust
    *totals.entry(seed).or_insert(0.0) += weight_at(0) * initial_mass;
```

Change the loop header (current line 71) to name the depth:

```rust
    for depth in 0..params.max_depth {
```

Change the readout line (current line 115) to weight `totals` while leaving `next` and the prune raw. The block starting at the `transmitted` computation should read:

```rust
                let transmitted = mass * refl * p * sim_term;
                if transmitted < params.min_intensity {
                    continue;
                }
                // READOUT is weighted by arrival depth; FLOW stays raw so mass keeps
                // propagating through depths that score zero.
                *totals.entry(v).or_insert(0.0) += weight_at(depth + 1) * transmitted;
                *next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;
```

- [ ] **Step 4: Make the whole workspace compile again**

Adding the field breaks every `PropagationParams { .. }` literal and dropping `Copy` breaks one deref. Fix each site by adding `depth_weights: None`:

- `rgdb/src/propagation.rs` — the 4 existing test literals at (pre-edit) lines 153, 174, 194, 207: change `PropagationParams { max_depth: 4, min_intensity: 1e-6 }` → `PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None }`.
- `rgdb/src/credit.rs` — 4 literals: `exact()` (`{ max_depth: 4, min_intensity: 0.0 }`), and the three `{ max_depth: 2/2/1, min_intensity: 0.0 }` at (pre-edit) lines 365, 380, 394. Add `, depth_weights: None` to each.
- `rgdb/src/engine.rs` — `params()` helper (pre-edit line 244) and the two `{ max_depth: 2, min_intensity: 0.0 }` at lines 314, 416: add `, depth_weights: None`. AND change line 145 `params: *params,` → `params: params.clone(),` (PropagationParams is no longer `Copy`).
- `rgdb/src/rag/query_engine.rs:82` — `PropagationParams { max_depth: config.max_hops, min_intensity: config.min_relevance }` → add `, depth_weights: None`.
- `rgdb-python/src/lib.rs` — lines 81 and 107 `PropagationParams { max_depth, min_intensity }` → add `, depth_weights: None` (real kwarg wiring lands in Task 5).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd rgdb && cargo test -p rgdb 2>&1 | tail -15`
Expected: PASS (all existing tests + 4 new ones green).

Run: `cargo build --workspace 2>&1 | tail -5`
Expected: workspace compiles (rgdb-python included).

- [ ] **Step 6: Commit**

```bash
git add rgdb/src/propagation.rs rgdb/src/engine.rs rgdb/src/credit.rs rgdb/src/rag/query_engine.rs rgdb-python/src/lib.rs
git commit -m "feat(rgdb): depth-weighted readout in propagate (flow + pruning stay raw)"
```

---

### Task 3: Make the credit pass `c`-aware (exact-remaining-length backward)

**Files:**
- Modify: `rgdb/src/credit.rs` (`backward`, `credit`, `backward_mass_at_target`, tests)

**Interfaces:**
- Consumes: `PropagationParams.depth_weights` from Task 2.
- Produces: `credit(...)` and `backward_mass_at_target(...)` (unchanged signatures) now honor `params.depth_weights`. Generalized invariant: `Σ_seeds mass · Σ_{L=0}^{max_depth} c_L · B_exact[L][(seed, query_relation)] == propagate(..., params)[target]` exactly at `min_intensity == 0.0`. Reduces to the current invariant at `c = None`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `rgdb/src/credit.rs`:

```rust
    /// 0 -A-> 3 (length 1) and 0 -A-> 1 -B-> 3 (length 2). Target 3 is reachable at
    /// two DIFFERENT path lengths, so depth weights change both the forward readout
    /// and the credit attribution — the discriminating fixture.
    fn two_lengths() -> (Graph, RelationVocab) {
        let a3 = (3u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let a1 = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let b3 = (3u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(
            4,
            vec![vec![a3, a1], vec![b3], vec![], vec![]],
            NodeProps::default(),
        )
        .unwrap();
        let v = RelationVocab::new(vec!["A".into(), "B".into()], vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        (g, v)
    }

    fn params_with(c: Option<crate::depth_weights::DepthWeights>) -> PropagationParams {
        PropagationParams { max_depth: 2, min_intensity: 0.0, depth_weights: c }
    }

    #[test]
    fn generalized_invariant_holds_under_depth_weights() {
        use crate::depth_weights::DepthWeights;
        let (g, v) = two_lengths();
        // (depth weights, expected forward mass at target 3)
        //   terminal(1): only the length-1 direct path  = 0.85 * 0.5 = 0.425
        //   terminal(2): only the length-2 path 0-A->1-B->3 = 0.425 * 0.425 = 0.180625
        //   [0,1,1]:     both                              = 0.605625
        let cases = [
            (DepthWeights::terminal(2, 1).unwrap(), 0.425_f32),
            (DepthWeights::terminal(2, 2).unwrap(), 0.180625_f32),
            (DepthWeights::from_vec(vec![0.0, 1.0, 1.0], 2).unwrap(), 0.605625_f32),
        ];
        for (c, expected_fwd) in cases {
            let p = params_with(Some(c));
            let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&3).unwrap();
            let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 3, &p);
            assert!((fwd - expected_fwd).abs() < 1e-5, "fwd {fwd} != expected {expected_fwd}");
            assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
        }
    }

    #[test]
    fn terminal_credit_selects_paths_by_length() {
        use crate::depth_weights::DepthWeights;
        let (g, v) = two_lengths();
        // terminal(1): only the direct 0-A->3 hop earns credit -> (A, A) at 100%.
        let c1 = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &params_with(Some(DepthWeights::terminal(2, 1).unwrap())));
        assert_eq!(c1.len(), 1);
        assert_eq!((c1[0].0, c1[0].1), (0, 0));
        assert!((c1[0].2 - 1.0).abs() < 1e-5);

        // terminal(2): only the 0-A->1-B->3 path -> credit split across (A,A) and (A,B).
        let mut c2 = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &params_with(Some(DepthWeights::terminal(2, 2).unwrap())));
        c2.sort_by_key(|&(a, b, _)| (a, b));
        assert_eq!(c2.len(), 2);
        assert_eq!((c2[0].0, c2[0].1), (0, 0));
        assert_eq!((c2[1].0, c2[1].1), (0, 1));
        assert!((c2[0].2 - 0.5).abs() < 1e-5, "(A,A) got {}", c2[0].2);
        assert!((c2[1].2 - 0.5).abs() < 1e-5, "(A,B) got {}", c2[1].2);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rgdb && cargo test -p rgdb credit 2>&1 | tail -25`
Expected: FAIL. `generalized_invariant_holds_under_depth_weights` fails because `backward_mass_at_target` still returns only `b[d]` (at-most semantics, ignores `c`), and `terminal_credit_selects_paths_by_length` fails because `credit` ignores `c`.

- [ ] **Step 3: Make `backward` exact-remaining-length**

In `rgdb/src/credit.rs`, the `backward` function currently re-adds `[v == target]` at every level, giving "at most `j` hops". Change it to *exactly* `j` remaining hops by dropping that re-add. In the `for j in 1..=d` loop (current lines 113-133), change:

```rust
        for &(v, r) in ball {
            let mut acc = if v == target { 1.0 } else { 0.0 };
```

to:

```rust
        for &(v, r) in ball {
            // Exact remaining length: no target re-add. B[j] is mass over
            // continuations of EXACTLY j more hops. (b[0] still seeds [v==target].)
            let mut acc = 0.0f32;
```

Leave the `b[0]` initialization (lines 109-111) unchanged — it correctly sets `B[0][(v,r)] = [v == target]`.

- [ ] **Step 4: Make `credit` weight flows by total path length via a `G_k` precompute**

In `credit` (current lines 164-219), after `let b = backward(...);` (line 180) and before the `let mut flow` block, insert the `G_k` precompute:

```rust
    // c[L] weights a path of total length L. A hop at forward position k+1 with j
    // hops remaining sits on a length-(k+1+j) path. Precompute per forward level k:
    //   G_k[(v,r)] = Σ_{j=0}^{d-k-1} c[k+1+j] · B[j][(v,r)]
    // so the flow loop stays O(edges) rather than O(edges · d).
    let c = params.depth_weights.as_ref().map(|w| w.as_slice());
    let weight_at = |depth: usize| -> f32 { c.map_or(1.0, |cc| cc[depth]) };
    let mut gk: Vec<HashMap<State, f32>> = vec![HashMap::new(); d];
    for k in 0..d {
        let budget = d - k - 1;
        let mut acc: HashMap<State, f32> = HashMap::new();
        for j in 0..=budget {
            let wj = weight_at(k + 1 + j);
            if wj == 0.0 {
                continue;
            }
            for (&st, &val) in &b[j] {
                *acc.entry(st).or_insert(0.0) += wj * val;
            }
        }
        gk[k] = acc;
    }
```

Then in the flow loop (current lines 183-212), remove the `let budget = d - k - 1;` line and change the backward lookup from `b[budget]` to `gk[k]`:

```rust
    let mut flow: HashMap<(RelationId, RelationId), f32> = HashMap::new();
    for k in 0..d {
        let states: Vec<(State, f32)> = f[k].iter().map(|(&s, &m)| (s, m)).collect();
        for ((u, r_in), mass) in states {
            if mass <= 0.0 {
                continue;
            }
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
                let bv = gk[k].get(&(v, Some(ep.relation))).copied().unwrap_or(0.0);
                if bv <= 0.0 {
                    continue;
                }
                *flow.entry((r_from, ep.relation)).or_insert(0.0) += mass * w * bv;
            }
        }
    }
```

- [ ] **Step 5: Make `backward_mass_at_target` `c`-aware**

Because `backward` is now exact-length, `b[d]` alone is only the length-`d` mass. The diagnostic must sum all lengths, weighted by `c`. Replace the final expression of `backward_mass_at_target` (current lines 240-243):

```rust
    let c = params.depth_weights.as_ref().map(|w| w.as_slice());
    let weight_at = |depth: usize| -> f32 { c.map_or(1.0, |cc| cc[depth]) };
    seeds
        .iter()
        .map(|&(s, m)| {
            let per_seed: f32 = (0..=d)
                .map(|l| weight_at(l) * b[l].get(&(s, query_relation)).copied().unwrap_or(0.0))
                .sum();
            m * per_seed
        })
        .sum()
```

Also update its doc comment (current lines 221-222) to reflect the generalization:

```rust
/// Diagnostic: `Σ_seeds seed_mass · Σ_L depth_weights[L] · B_exact[L][(seed, query_relation)]`.
/// Equals `propagate(..., params)[target]` exactly when `params.min_intensity == 0.0`.
/// At `depth_weights = None` (uniform) this reduces to the at-most-`max_depth` mass.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd rgdb && cargo test -p rgdb credit 2>&1 | tail -20`
Expected: PASS. Both new tests pass, AND the existing `backward_matches_forward_on_chain`, `backward_matches_forward_with_refraction`, `credits_are_l1_normalized`, `chain_credits_only_the_diagonal`, `two_relation_splits_credit_evenly` stay green (they use `c = None`, which reduces to today's behavior).

Run: `cd rgdb && cargo test -p rgdb 2>&1 | tail -5`
Expected: full suite green.

- [ ] **Step 7: Commit**

```bash
git add rgdb/src/credit.rs
git commit -m "feat(rgdb): c-aware credit pass (exact-length backward + G_k length weighting)"
```

---

### Task 4: Engine config default and query-time resolution

**Files:**
- Modify: `rgdb/src/engine.rs` (`EngineConfig`, `query`, tests)

**Interfaces:**
- Consumes: `DepthWeights` (Task 1), `PropagationParams.depth_weights` (Task 2), `c`-aware `credit` (Task 3).
- Produces: `EngineConfig { cache_capacity, cache_ttl, default_depth_weights: DepthWeights }` (no longer `Copy`; is `Clone`). `query` resolves weights: caller's `Some` wins; else the config default is applied iff its length matches `max_depth + 1`; else uniform. The resolved weights are stored in `QueryContext.params`, so `record_feedback` credits under the exact weights the query used.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `rgdb/src/engine.rs`. The `engine(..)` helper builds `0 -A-> 1 -B-> 2`; extend with a config default and assert it changes the ranking, and that a caller override wins:

```rust
    #[test]
    fn engine_default_depth_weights_are_applied_when_caller_omits_them() {
        use crate::depth_weights::DepthWeights;
        // Graph 0 -A-> 1 -B-> 2. terminal(2) scores ONLY depth-2 arrivals, so node 2
        // (depth 2) ranks above node 1 (depth 1); under uniform, node 1 outranks it.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let e = RgdbEngine::with_engine_config(
            g,
            prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
            EngineConfig {
                cache_capacity: 8,
                cache_ttl: Duration::from_secs(3600),
                default_depth_weights: DepthWeights::terminal(4, 2).unwrap(),
            },
        );
        // Caller passes None -> engine applies its terminal(2) default (max_depth 4 matches).
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(2), "terminal(2) default ranks node 2 first");
    }

    #[test]
    fn caller_depth_weights_override_the_engine_default() {
        use crate::depth_weights::DepthWeights;
        let e = engine(0); // 0 -A-> 1 -B-> 2, default config = uniform
        // Caller forces terminal(1): node 1 (depth 1) must rank first.
        let p = PropagationParams {
            max_depth: 4,
            min_intensity: 0.0,
            depth_weights: Some(DepthWeights::terminal(4, 1).unwrap()),
        };
        let r = e.query(&[(0, 1.0)], Some(0), &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(1), "caller terminal(1) ranks node 1 first");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rgdb && cargo test -p rgdb engine 2>&1 | tail -20`
Expected: FAIL to compile — `EngineConfig` has no `default_depth_weights` field.

- [ ] **Step 3: Add the config field**

In `rgdb/src/engine.rs`, add the import near the top (after `use crate::credit::credit;`):

```rust
use crate::depth_weights::DepthWeights;
```

Change `EngineConfig` (currently lines 38-48) — drop `Copy`, add the field, set the default:

```rust
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub cache_capacity: usize,
    pub cache_ttl: Duration,
    /// Applied to queries whose caller omits `depth_weights`, but only when its
    /// length matches the query's `max_depth + 1`; otherwise the query runs uniform.
    /// Defaults to `uniform` (all-ones), so out-of-the-box behavior is unchanged.
    pub default_depth_weights: DepthWeights,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            cache_capacity: 4096,
            cache_ttl: Duration::from_secs(3600),
            default_depth_weights: DepthWeights::uniform(4),
        }
    }
}
```

- [ ] **Step 4: Resolve weights in `query`**

In `RgdbEngine::query` (currently lines 128-151), resolve the weights before propagating and store the resolved params in the cached context. Replace the body:

```rust
    pub fn query(
        &self,
        seeds: &[(NodeId, f32)],
        query_relation: Option<RelationId>,
        params: &PropagationParams,
    ) -> QueryResult {
        // Caller's weights win; else apply the config default when its length fits
        // this query's max_depth; else leave uniform.
        let resolved = if params.depth_weights.is_some() {
            params.clone()
        } else if self.engine_cfg.default_depth_weights.as_slice().len() == params.max_depth + 1 {
            PropagationParams {
                depth_weights: Some(self.engine_cfg.default_depth_weights.clone()),
                ..params.clone()
            }
        } else {
            params.clone()
        };

        let vocab = self.vocab.load_full();
        let totals = propagate(&self.graph, &vocab, seeds, query_relation, &resolved);
        let mut ranked: Vec<(NodeId, f32)> = totals.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache.lock().unwrap().put(
            query_id,
            QueryContext {
                seeds: seeds.to_vec(),
                query_relation,
                params: resolved,
                vocab,
                created: Instant::now(),
            },
        );
        QueryResult { ranked, query_id }
    }
```

(`record_feedback` already credits under `ctx.params`, which now carries the resolved `depth_weights` — no change needed there.)

- [ ] **Step 5: Fix the one `EngineConfig` literal that lost `Copy`**

The `expired_query_context_errors_and_is_evicted` test (current line ~371) builds an `EngineConfig` literal. Add the field:

```rust
            EngineConfig {
                cache_capacity: 8,
                cache_ttl: Duration::from_millis(1),
                default_depth_weights: DepthWeights::uniform(4),
            },
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd rgdb && cargo test -p rgdb 2>&1 | tail -8`
Expected: PASS — both new engine tests plus the full existing suite (the uniform default at max_depth 4 keeps `params()`-based tests, which use `max_depth: 4/2`, bit-identical: at max_depth 4 the default all-ones applies as a no-op; at max_depth 2 the length-5 default is skipped → uniform).

- [ ] **Step 7: Commit**

```bash
git add rgdb/src/engine.rs
git commit -m "feat(rgdb): EngineConfig default_depth_weights + query-time resolution"
```

---

### Task 5: Python bindings — `depth_weights` kwarg

**Files:**
- Modify: `rgdb-python/src/lib.rs` (`propagate`, `PyEngine::query`)

**Interfaces:**
- Consumes: `DepthWeights::from_vec`, `PropagationParams.depth_weights`, `EngineConfig` default (Tasks 1-4).
- Produces: `core.propagate(graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None)` and `Engine.query(seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None)`, where `depth_weights` is `list[float] | None`. Invalid weights raise `ValueError`. Both now return `PyResult<..>`.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_depth_weights_binding.py`:

```python
"""depth_weights kwarg on the native propagate() / Engine.query() bindings."""
import math
import pytest
from rgdb_embeddings import _rgdb_core as core


def _chain():
    # 0 -> 1 -> 2 -> 3, single relation, no attenuation.
    adj = [[(1, 0.0, 0)], [(2, 0.0, 0)], [(3, 0.0, 0)], []]
    g = core.build_graph(4, adj)
    v = core.uniform_vocab(1)
    return g, v


def test_uniform_weights_match_none():
    g, v = _chain()
    a = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, None))
    b = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, 1.0, 1.0, 1.0]))
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k]  # exact


def test_terminal_scores_only_arrival_depth():
    g, v = _chain()
    t = dict(core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [0.0, 0.0, 0.0, 1.0]))
    assert math.isclose(t.get(3, 0.0), 0.614125, rel_tol=1e-4)
    assert t.get(1, 0.0) == 0.0


def test_wrong_length_raises_value_error():
    g, v = _chain()
    with pytest.raises(ValueError):
        core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, 1.0])  # needs length 4


def test_negative_weight_raises_value_error():
    g, v = _chain()
    with pytest.raises(ValueError):
        core.propagate(g, v, [(0, 1.0)], 0, 3, 1e-6, [1.0, -1.0, 1.0, 1.0])
```

- [ ] **Step 2: Build the bindings and run to verify failure**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml` (from repo root; the existing dev workflow), then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests/test_depth_weights_binding.py -q`
Expected: FAIL — `propagate()` takes no `depth_weights` argument (TypeError), so the tests error.

- [ ] **Step 3: Wire the kwarg**

In `rgdb-python/src/lib.rs`, add the import (after line 6):

```rust
use rgdb::depth_weights::DepthWeights;
```

Add a small helper above the `propagate` pyfunction:

```rust
fn to_depth_weights(
    depth_weights: Option<Vec<f32>>,
    max_depth: usize,
) -> PyResult<Option<DepthWeights>> {
    match depth_weights {
        None => Ok(None),
        Some(v) => DepthWeights::from_vec(v, max_depth)
            .map(Some)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}"))),
    }
}
```

Replace the `propagate` pyfunction (currently lines 71-84):

```rust
#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None))]
fn propagate(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
    depth_weights: Option<Vec<f32>>,
) -> PyResult<Vec<(u32, f32)>> {
    let params = PropagationParams {
        max_depth,
        min_intensity,
        depth_weights: to_depth_weights(depth_weights, max_depth)?,
    };
    let totals = rust_propagate(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    Ok(totals.into_iter().collect())
}
```

Replace `PyEngine::query` (currently lines 105-110):

```rust
    #[pyo3(signature = (seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None))]
    fn query(
        &self,
        seeds: Vec<(u32, f32)>,
        query_relation: Option<u16>,
        max_depth: usize,
        min_intensity: f32,
        depth_weights: Option<Vec<f32>>,
    ) -> PyResult<(Vec<(u32, f32)>, u64)> {
        let params = PropagationParams {
            max_depth,
            min_intensity,
            depth_weights: to_depth_weights(depth_weights, max_depth)?,
        };
        let r = self.inner.query(&seeds, query_relation, &params);
        Ok((r.ranked, r.query_id))
    }
```

- [ ] **Step 4: Rebuild and run to verify pass**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests/test_depth_weights_binding.py -q`
Expected: PASS (4 tests).

Run the existing eval suite to confirm the added positional/kwarg is backward-compatible: `../.venv/Scripts/python.exe -m pytest tests -q`
Expected: PASS (24 tests: 20 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/tests/test_depth_weights_binding.py
git commit -m "feat(bindings): depth_weights kwarg on propagate + Engine.query"
```

---

### Task 6: MetaQA acceptance — `terminal(k)` per hop meets the success criteria

**Files:**
- Create: `rgdb-eval/scripts/experiment_depth_weights_acceptance.py`
- Create (output): `rgdb-eval/results/metaqa-depth-weights.md`

**Interfaces:**
- Consumes: the `depth_weights` binding (Task 5), the existing trained-matrix builder pattern, and MetaQA loaders.
- Produces: an assert-gated script (like `experiment_online_learning.py`) that runs `terminal(k)` per hop against the trained matrix and enforces the spec's success criteria.

- [ ] **Step 1: Write the acceptance script**

Create `rgdb-eval/scripts/experiment_depth_weights_acceptance.py`:

```python
"""Acceptance: depth-weighted terminal(k) readout meets the spec's success criteria.

For each hop k in {1,2,3}, score MetaQA test questions with the trained transition
matrix, seeded with the first-hop relation, using depth_weights = terminal(k) via the
native binding. Asserts the floors from
docs/superpowers/specs/2026-07-10-depth-aware-scoring-design.md.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_depth_weights_acceptance.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, mrr, K_VALUES

DATA = "data/MetaQA"
LIMIT = 1000
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05

# spec success criteria (floors)
THRESHOLDS = {1: 0.989, 2: 0.60, 3: 0.39}


def build_trained(graph) -> np.ndarray:
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M


def terminal(k: int) -> list[float]:
    v = [0.0] * (MAX_DEPTH + 1)
    v[k] = 1.0
    return v


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n} relations")

    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    trained = build_trained(graph)
    vocab = core.vocab_from_matrix(list(graph.relations), trained.ravel().tolist())
    k_max = max(K_VALUES)

    rows = []
    failures = []
    for hop in (1, 2, 3):
        qs = load_questions(
            os.path.join(DATA, f"qa_test_{hop}hop.txt"), hop, graph, limit=LIMIT,
            qtype_path=os.path.join(DATA, f"qa_test_{hop}hop_qtype.txt"),
        )
        w = terminal(hop)
        items = []
        for q in qs:
            rel = graph.relation_to_id.get(q.relation) if q.relation else None
            totals = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel,
                                         MAX_DEPTH, MIN_INTENSITY, w))
            totals.pop(q.topic_id, None)
            ranked = [nid for nid, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
            items.append((ranked, set(q.answer_ids)))
        m = mean(mrr(r, gs) for r, gs in items)
        h1 = mean(hits_at_k(r, gs, 1) for r, gs in items)
        rows.append((hop, len(items), h1, m))
        status = "PASS" if m >= THRESHOLDS[hop] else "FAIL"
        if m < THRESHOLDS[hop]:
            failures.append((hop, m, THRESHOLDS[hop]))
        print(f"  hop{hop}: terminal({hop}) MRR {m:.3f} (>= {THRESHOLDS[hop]}) Hits@1 {h1:.3f}  [{status}]")

    md = ["# Depth-weighted terminal(k) acceptance (MetaQA)\n",
          "\ntrained matrix, first-hop relation seed, depth_weights = terminal(k)\n",
          "\n| hop | n | hits@1 | mrr | floor |", "|---|---|---|---|---|"]
    for hop, n_q, h1, m in rows:
        md.append(f"| {hop} | {n_q} | {h1:.3f} | {m:.3f} | {THRESHOLDS[hop]} |")
    out = "results/metaqa-depth-weights.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwrote {out}")

    assert not failures, f"depth-weight acceptance failed: {failures}"
    print("\nACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the acceptance script**

Run: `cd rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_depth_weights_acceptance.py`
Expected: prints per-hop PASS lines and `ACCEPTANCE PASSED`; writes `results/metaqa-depth-weights.md`. Exit code 0. (If the `.metaqa` data is absent, the script errors on `load_kb` — that is an environment gap, not a code failure; note it in the report and rerun where the data exists, as with the other `experiment_*.py`.)

- [ ] **Step 3: Commit**

```bash
git add rgdb-eval/scripts/experiment_depth_weights_acceptance.py rgdb-eval/results/metaqa-depth-weights.md
git commit -m "eval: MetaQA acceptance for depth-weighted terminal(k) readout"
```

---

## Notes for the executor

- **Do not merge or push.** Everything lands on `feature/rgdb-self-learning-loop`.
- **Bit-exactness is a hard gate** (Task 2 Step 1, Task 5 Step 1): assert `==`, never `abs() < eps`, for the uniform-vs-none comparison.
- The two big correctness risks are both covered by a single discriminating test each: `generalized_invariant_holds_under_depth_weights` (Task 3) for the credit math, and `flow_is_not_weighted_so_terminal_still_reaches_depth` (Task 2) for the readout/flow split. If either is weakened to a non-discriminating fixture, the coverage is lost.
- Windows/PowerShell note: the `cargo`/`pytest`/`maturin` commands assume the repo's existing `.venv` (Python 3.12) and the standard dev build. Adjust the `maturin develop` invocation to match however the bindings were last built in this session.
