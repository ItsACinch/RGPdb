# Depth-Aware Scoring for RGDB Propagation — Design

**Date:** 2026-07-10
**Branch:** extends `feature/rgdb-self-learning-loop` (kept as-is, not merged)
**Status:** approved design, pending implementation plan

## Summary

Add per-hop scoring coefficients `c_k` to `propagate()`: the score of a node
becomes a weighted sum over the depths at which mass arrives at it,
`score(v) = Σ_k c_k · I_k(v)`, where `I_k(v)` is the mass reaching `v` at exactly
`k` hops. Today's kernel is the special case `c = [1, 1, …, 1]` (all arrival
depths summed equally), which this change reproduces bit-for-bit by default.

The coefficients are a **scoring** readout, not a **transmission** weight: mass
continues to flow through depths that score zero, so a terminal-only weighting
(`c = [0,0,0,1]`) still reaches depth-3 nodes.

## Motivation — the measured problem

Multi-hop retrieval collapses on RGDB not because the relation model is weak but
because the kernel sums mass over all arrival depths, and each hop multiplies mass
by `reflection · p(u→v) · sim ≤ 1`. On any single path a hop-1 node therefore
outscores the hop-3 answer on that same path — for *any* assignment of the existing
per-node / per-relation coefficients. Two independent measurements establish this:

1. **A perfect per-question relation schedule ranks a depth-1 node first in 99.8%
   of 3-hop questions** (`experiment_arrival_depth`-adjacent run), while 95% of the
   gold answers sit at depth 3. The relation model is not the bottleneck; the
   depth-blind readout is.

2. **The fix is a per-hop coefficient, and it is an interaction with the relation
   model, not a substitute for it.** On MetaQA 3-hop, trained matrix, seeded with
   the first-hop relation (`experiment_depth_control`, `experiment_arrival_depth`):

   | 3-hop MRR | no depth control | arrival-depth `terminal(3)` |
   |---|---:|---:|
   | uniform (refraction off) | 0.264 | — |
   | trained matrix | 0.235 | **0.398** |

   The trained matrix *without* depth control is worse than no refraction at all
   (0.235 < 0.264): sharpening relation coherence concentrates mass along the
   correct chain, making the hop-1/hop-2 intermediates on that chain stronger
   competitors to the answer. With depth control the same matrix wins.

### Why arrival depth, not shortest-path distance

The earlier evidence (0.419) came from masking candidates to graph shortest-path
distance == 3. That needs a per-query BFS over the reached ball — `O(ball)`, and
the ball reaches 6,641 nodes at depth 3 on this graph, so it does not scale.

Arrival-depth weighting is `O(1)` (the walk already visits per depth) and was
measured to reproduce the win. On MetaQA 3-hop, trained matrix, 300q:

| readout | hits@1 | hits@20 | mrr |
|---|---:|---:|---:|
| `c=[1,1,1,1]` (today) | 0.143 | 0.697 | 0.235 |
| `c=[0,0,1,0]` terminal-3 | 0.223 | **0.910** | **0.398** |
| `c=[-1,-1,1,1]` subtract-near | 0.200 | 0.880 | 0.386 |
| shortest-path==3 mask (0.419 evidence) | **0.243** | 0.853 | 0.416 |

Arrival-depth `terminal(3)` captures 96% of the mask's MRR win and beats it on
Hits@20, because the mask discards the ~13% of gold answers reachable by a shortcut
edge (shortest-path < 3) whereas arrival-depth keeps them. Arrival depth is *not*
shortest-path distance — 42.8% of depth-3 arrival mass lands on distance-1 nodes
via backtracking (movie→actor→movie→actor) — but zeroing `c_1` strips those nodes
of their large depth-1 mass, leaving them to compete only on their small backtrack
mass, which is why it works anyway.

### Do-no-harm: `terminal(k)` improves every hop

Trained matrix, seeded with the first-hop relation, 300q/hop:

| hop | `c=all-ones` (today) MRR / H@1 | `terminal(k)` MRR / H@1 |
|---|---:|---:|
| 1 | 0.989 / 0.983 | **0.995 / 0.993** |
| 2 | 0.280 / 0.000 | **0.640 / 0.483** |
| 3 | 0.235 / 0.143 | **0.398 / 0.223** |

The 2-hop Hits@1 jump from 0.000 to 0.483 is the headline: the "diffusion cannot do
Hits@1 past one hop" ceiling observed throughout this investigation was an artifact
of the unweighted sum, not a property of diffusion. For reference, untyped-PPR's
2-hop MRR is 0.219 and 3-hop is 0.279; `terminal(k)` beats both at every hop.

## Non-goals

- **Learning `c_k` from feedback.** No evidence a learned `c` beats `terminal(k)`,
  and the supervision design is separable. The seam is left open (see below) exactly
  as `TransitionStore` was, but nothing learns `c` in this work.
- **Hop-count prediction.** The caller supplies `k` (or the full weight vector),
  precisely as it already supplies `query_relation` from intent classification. RGDB
  never guesses hop count.
- **Negative coefficients.** Measured worse than non-negative (`-1,-1,1,1` → 0.386 vs
  `0,0,1,0` → 0.398), and non-negativity is relied on downstream (see Component 1).

## Architecture

Three units change; all are on the already-built self-learning branch.

### Component 1 — `DepthWeights` and the kernel readout (`propagation.rs`)

A validated newtype, length `max_depth + 1`, indexed by arrival depth. Index 0 is
the seed's own mass; index `k` is mass arriving after exactly `k` hops.

```rust
#[derive(Debug, Clone)]
pub struct DepthWeights(Vec<f32>);

#[derive(Debug, Clone, PartialEq)]
pub enum DepthWeightsError {
    WrongLength { expected: usize, got: usize },
    Negative { index: usize, value: f32 },
    NotFinite { index: usize },
    AllZero,
    TerminalExceedsMaxDepth { k: usize, max_depth: usize },
}

impl DepthWeights {
    /// [1.0; max_depth + 1] — reproduces today's behavior exactly.
    pub fn uniform(max_depth: usize) -> Self;

    /// 1.0 at index k, 0.0 elsewhere. Errors if k > max_depth.
    pub fn terminal(max_depth: usize, k: usize) -> Result<Self, DepthWeightsError>;

    /// Validated: length == max_depth+1, all finite, all >= 0, not all zero.
    pub fn from_vec(v: Vec<f32>, max_depth: usize) -> Result<Self, DepthWeightsError>;

    pub fn as_slice(&self) -> &[f32];
}
```

Validation enforces: correct length, every entry finite, every entry `>= 0`, and at
least one entry `> 0` (an all-zero vector scores nothing and is a caller bug).

**Non-negativity is load-bearing, not stylistic.** Three downstream consumers
assume scores are `>= 0`: `intensity_to_distance` computes `-ln(I + eps)`; the RAG
fusion multiplies graph score into a calibrated blend; and the credit pass feeds
flows into `TransitionStore` counts, which must not go negative. Forbidding negative
`c_k` protects all three, and the data says we lose nothing.

`PropagationParams` carries the weights:

```rust
#[derive(Debug, Clone)]          // was: Copy — now Clone only
pub struct PropagationParams {
    pub max_depth: usize,
    pub min_intensity: f32,
    pub depth_weights: Option<DepthWeights>,   // None => uniform
}
```

**Why inside `PropagationParams`, and the `Copy` trade-off.** `PropagationParams` is
only ever passed as `&PropagationParams`, and `queries.rs`, `embeddings.rs`, and
`main.rs` merely forward it. Placing `depth_weights` here means **no signature churn**
across the crate — only `propagation.rs`, `credit.rs`, and `engine.rs` read the new
field. The price is dropping `Copy` (a `Vec` is not `Copy`); `Clone` is retained.
Grep confirms no code relies on `PropagationParams: Copy` (every site takes a
reference or constructs a fresh value). This was weighed against passing
`depth_weights` as a separate argument to keep `Copy`; the in-struct form was chosen
for zero churn and because the weights are conceptually a propagation parameter.

The kernel change is at the readout site (`propagation.rs`, currently ~line 115),
using the loop's existing depth index (currently discarded at ~line 71):

```rust
// c = params.depth_weights.as_ref().map(|w| w.as_slice());  // hoisted before the loop
// ... inside `for depth in 0..max_depth`, inside the neighbor loop:
let w = c.map_or(1.0, |c| c[depth + 1]);            // arrival depth of v is depth+1
*totals.entry(v).or_insert(0.0) += w * transmitted; // READOUT: weighted
*next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;  // FLOW: unweighted
```

The seed's own mass (`totals[seed] += initial_mass` before the loop) is scaled by
`c[0]`. Two invariants the implementation must preserve:

- **Flow is never weighted.** `next` accumulates raw `transmitted`. Weighting the
  flow would make `terminal(3)` prune everything at depth 1 and reach nothing.
- **Pruning compares against raw flow.** The `transmitted < min_intensity` guard
  (and the frontier `mass < min_intensity` guard) must test the unweighted value,
  never `w * transmitted`. Otherwise a small `c_k` would spuriously prune live paths.

**Bit-exactness.** With `c = uniform`, `w == 1.0` and `1.0 * x == x` under IEEE-754,
so the output is identical to today's, not merely close. This is a testable promise.

### Component 2 — `c`-aware credit pass (`credit.rs`)

The credit pass must attribute exactly the paths that contribute to the score. Two
changes.

**Exact-remaining-length backward.** Today `backward()` computes "mass reaching the
target in *at most* `j` hops" by re-adding `[v == target]` at every level. Change it
to *exactly* `j` remaining hops by dropping that re-add:

```
B[0][(v, r)] = [v == target]
B[j][(v, r)] = Σ_{v→x} w(v→x | r) · B[j-1][(x, rel(v→x))]      (no target re-add)
```

**Length-weighted flow.** A hop at forward position `k+1` with exactly `j` hops
remaining lies on a path of total length `k + 1 + j`, so it earns `c[k+1+j]`:

```
flow(r_in, r_out) += F[k][(u, r_in)] · w(u→v) · B[j][(v, r_out)] · c[k+1+j]
                     for all 0 <= j <= max_depth - k - 1
```

To keep the hot loop `O(edges · max_depth)` rather than `O(edges · max_depth²)`,
precompute per forward level `k`:

```
G_k[(v, r)] = Σ_{j=0}^{max_depth-k-1} c[k+1+j] · B[j][(v, r)]
```

Then the inner accumulation is `F[k][(u,r_in)] · w · G_k[(v,r_out)]`, structurally
identical to today's single-`B` loop.

**Generalized invariant** (the acceptance test for this component):

```
Σ_seeds mass · ( c_0·[seed==target] + Σ_{L>=1} c_L · B_exact[L][(seed, query_relation)] )
    == propagate(graph, vocab, seeds, query_relation, params_with_c)[target]
```

exact when `min_intensity == 0.0`. At `c = uniform` this reduces to the existing
invariant (`Σ seed_mass · B_maxdepth[(seed, qr)] == propagate[target]`), because
summing `B_exact` over all lengths equals the at-most-maxdepth `B`.

`backward_mass_at_target` and any public helper that assumed at-most-`j` semantics
must be updated or documented; downstream callers in `engine.rs` use the top-level
`credit()` result, not the intermediate `B`, so the blast radius is internal.

### Component 3 — engine default, bindings, query plumbing (`engine.rs`, `rgdb-python`)

`EngineConfig` gains a default, applied only when the caller omits weights:

```rust
pub struct EngineConfig {
    pub cache_capacity: usize,
    pub cache_ttl: Duration,
    pub default_depth_weights: DepthWeights,   // defaults to uniform(max_depth)
}
```

Resolution order at query time: caller-supplied `depth_weights` if `Some`, else
`config.default_depth_weights`. Out of the box the default is `uniform`, so existing
behavior is unchanged until an operator configures it or a caller overrides.

`QueryContext` already captures the vocab the query ran under; it must also capture
the resolved `DepthWeights`, so `record_feedback` runs the `c`-aware credit pass with
the exact weights the query used — never the current config default, which may have
changed. This mirrors the existing "credit under the query-time vocab" rule.

**Bindings.** Additive, keyword-defaulted so existing Python callers are unaffected:

```python
core.propagate(graph, vocab, seeds, query_relation=None,
               max_depth=4, min_intensity=1e-3, depth_weights=None)   # list[float] | None
Engine.query(seeds, query_relation=None, max_depth=4, min_intensity=1e-3,
             depth_weights=None)                                       # list[float] | None
```

A Python-side `DepthWeightsError` maps to `ValueError` (consistent with the existing
`FeedbackError → ValueError` mapping; distinct exception types remain a logged
fix-later ticket for the whole crate, not this feature).

## Data flow

```
caller (intent classifier) ──k or c──▶ propagate()/Engine.query()
                                          │
              resolve: caller c ?? config.default ?? uniform
                                          │
   walk depth 0..max_depth:  totals[v] += c[depth+1]·transmitted   (readout)
                             next[v]   +=            transmitted    (flow, unweighted)
                                          │
                                   ranked results
                                          │
   Engine.record_feedback(query_id, target, signal)
                                          │
        QueryContext.depth_weights (captured at query time)
                                          │
   c-aware credit(): flow(r_in,r_out) weighted by c[path_length]
                                          │
                              TransitionStore counts
```

## Testing strategy

Rust unit tests (the crate's own suite):

- `DepthWeights` validation: wrong length, negative, non-finite, all-zero, `terminal`
  with `k > max_depth` each return the specific error variant.
- `uniform` bit-exactness: on the existing `chain()` and multi-path fixtures,
  `propagate` with `depth_weights = Some(uniform)` and with `None` produce identical
  maps (assert exact equality, not approximate).
- Terminal readout on the chain fixture: `terminal(k)` yields a node's mass only at
  its arrival depth; a hand-computed 3-node chain checks the exact value.
- Flow-not-weighted: a fixture where `terminal(last)` still reaches the terminal node
  (proves mass flowed through zero-scored intermediate depths).
- Pruning uses raw flow: a fixture with small `c_k` and a `min_intensity` between
  `c_k·transmitted` and `transmitted` reaches the node (proves the guard tests raw).
- **`c`-aware credit invariant**: on an asymmetric multi-relation fixture with a
  non-uniform `c`, at `min_intensity = 0.0`, the generalized invariant holds to `1e-5`.
  Write this test first — it is the one that catches a wrong exact-length backward.
- Credit at `c = uniform` reduces to the existing credit result (regression guard).

Python acceptance (`rgdb-eval`, promote the exploratory scripts to a checked-in
experiment): on MetaQA 1000q/hop with the trained matrix, seeded with the first-hop
relation, assert the success criteria below.

## Success criteria

Measured on MetaQA 1000q/hop, trained matrix, seeded with the first-hop relation,
each hop-`k` bucket scored with `depth_weights = terminal(max_depth, k)`. Thresholds
are floors set below the 300q exploratory measurements to absorb full-set variance;
the parenthetical "from" is the current `c = all-ones` value for that hop.

| check | requirement |
|---|---|
| `c = uniform` | `propagate` output bit-identical to current kernel |
| 1-hop MRR, `terminal(1)` | `>= 0.989` (all-ones floor; expected ~0.995) |
| 2-hop MRR, `terminal(2)` | `>= 0.60` (from 0.280) |
| 3-hop MRR, `terminal(3)` | `>= 0.39` (from 0.235) |
| credit invariant | generalized invariant holds for a non-uniform `c` at `min_intensity = 0` |
| existing suite | all current Rust + eval tests still green |

## Open seam (deliberately not built)

Learning `c_k` from feedback. The credit pass's forward frontier `F[k]` already
indexes mass by depth, so the per-depth arrival profile of a rewarded answer is
available at feedback time — the supervision a `c`-fitter would need. This design
does not build it, does not add a knob for it, and takes no dependency on it. It is
recorded so the option is not lost, exactly as the deferred query-conditioned
schedule (Option C) was. Sequence remains: **B (done) → depth-aware scoring (this) →
re-open C**, where C's gate is now "a predicted schedule beats `terminal(k)` under
the same depth control," not "beats untyped-PPR."
