# Feedback-Learned Ranking Improvements for RGDB — Design

**Date:** 2026-07-11
**Branch:** extends `feature/rgdb-self-learning-loop`
**Status:** approved design, pending implementation plan

## Summary

Two feedback-learned ranking improvements built on the just-completed depth-aware
scoring, plus the shared kernel primitive both require:

- **A. Layered propagation primitive** — the kernel optionally returns per-node
  *per-depth* intensity `I_k(v)` instead of only the depth-weighted sum.
- **#2. Learned soft depth weights (`DepthProfileStore`)** — the engine learns a
  smoothed `DepthWeights` per hop-class from feedback, replacing the hard
  `terminal(k)`; improves *recall* by keeping answers at shortcut distances.
- **#4. Hits@1 reranker (`Reranker`)** — an online-learned linear model reorders the
  top-K by depth/score/degree/relation features; improves *precision-at-1*.

Both learn online from the existing `RgdbEngine` feedback loop, mirroring
`TransitionStore` (accumulate → derive → atomic swap), with no new heavy ML
dependencies. Both are strict no-ops at cold start (do-no-harm).

Two other candidates from the same discussion are **out of scope**:
- **#1 second-order transition matrix — dropped, with proof.** A measure-first
  entropy gate (`scripts/investigate_2nd_order_gate.py`) showed the information gain
  from conditioning the third relation on the first is **0.000 bits**
  (`H(r3|r2) = H(r3|r1,r2) = 2.294`) on MetaQA: the 3-hop chains share a fixed
  `movie→person→movie` skeleton, so `r1` is determined by `r2`. A second-order model
  would carry extra frontier state and buy nothing. The residual `r3` ambiguity is
  question-conditioned, which only #3 can address.
- **#3 query-conditioned per-hop schedule (Option C) — deferred** to its own effort;
  it needs a schedule predictor (RotatE + the feedback loop). See
  `2026-07-10-deferred-query-conditioned-refraction.md`.

## Motivation

Depth-aware scoring (`terminal(k)`) lifted 3-hop MRR to 0.381 (from 0.235) and 2-hop
to 0.612 (from 0.266). Two gaps remain in the *deployable* setup:

1. **`terminal(k)` is a hard one-hot** that zeroes the ~13% of answers reachable at a
   shortcut distance `< k`. A learned soft profile recovers them (#2).
2. **3-hop Hits@1 is only 0.203** even with `terminal(3)` — the answer is usually in
   the top-20 but not first, because diffusion still ranks some intermediates high.
   A reranker over per-candidate features targets exactly this (#4).

Neither closes the 0.381→0.927 gold-schedule ceiling — the entropy result confirms
that ceiling is pure question-conditioning (#3). These two squeeze the deployable
regime: #2 on recall, #4 on precision-at-1.

## Component A — Layered propagation primitive

`propagate` today returns `HashMap<NodeId, f32>`, the depth-weighted sum
`Σ_k c_k·I_k(v)`. #2 and #4 both need the un-collapsed vector.

```rust
/// Per reached node, the intensity arriving at each depth: index d = mass arriving
/// after exactly d hops (index 0 = seed mass). Length max_depth+1.
pub fn propagate_layered(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> HashMap<NodeId, Vec<f32>>;
```

Same walk as `propagate_single`, accumulating a `vec![0.0; max_depth+1]` per node
instead of a scalar; each arrival adds RAW `transmitted` at index `depth+1` (the
per-depth analogue of the weighted readout — the layered result is un-weighted so a
consumer can apply any `c`). `params.depth_weights` is IGNORED by the layered pass
(it returns the raw per-depth decomposition; weighting is a consumer concern).

**Consistency invariant (a test):** for any `DepthWeights c`,
`Σ_d c[d]·layered[v][d] == propagate(v)` for the same params carrying `c`, exactly at
`min_intensity == 0`. This ties the new primitive to the shipped kernel.

The scalar `propagate` stays as the fast path (a `Vec<f32>` per node costs more than a
scalar; only paid when features are needed). Pruning still tests raw `transmitted`, as
in the scalar kernel.

> **MEASURED OUTCOME (2026-07-11): #2's learning is a dead end; the plumbing is kept,
> the learning is disabled by default.** The full-set MetaQA gate
> (`scripts/investigate_soft_depth_derivation.py`, `results/metaqa-soft-depth-weights.md`)
> showed BOTH the discriminative derivation below and a generative alternative lose to
> hard `terminal(k)` at 3 hops (MRR 0.28 / recall@20 0.49 vs terminal(3)'s 0.38 / 0.53).
> Root cause: the 3-hop answer's arrival mass is ~76% at depth 1 (it is usually also
> reachable via a 1-hop shortcut), but so are the distractors — any `c` derived from
> answer-mass up-weights depth 1 and drowns the answer. `terminal(k)` wins by isolating
> the depth where answer and distractors SEPARATE, not where the answer's mass is. No
> single per-depth vector learned from answer-mass statistics can beat `terminal(k)`;
> separating same-depth answer from distractor needs a per-NODE model — which is #4.
>
> **Disposition:** keep the `hop_hint` API (its cold start IS `terminal(k)`, a useful
> ergonomic) and the `DepthProfileStore` type (the seam), but gate the feedback-driven
> depth-profile update behind `EngineConfig.depth_profile_learning`, default **false**.
> Default behavior: `hop_hint` → `terminal(k)`, no drift. The 2-hop case is healthy
> (soft-c matches terminal(2) at 0.61), so the seam is not worthless — but it is off by
> default until a per-node-aware derivation exists. The derivation below is retained as
> the opt-in path; it is NOT the recommended default.

## Component #2 — Learned soft depth weights (`DepthProfileStore`)

A sibling of `TransitionStore`, in `rgdb/src/depth_profile.rs`.

**State:** for each hop-class `k` in `0..=max_depth`, two length-`(max_depth+1)`
accumulators — `answer_mass[k][d]` and `background_mass[k][d]` — plus the config.

**Record (on feedback):** given the rewarded target `t`, the query's `hop_hint = k`,
signal `s`, and the query's layered result `L` (captured in the `QueryContext`):
```
for d in 0..=max_depth:
    answer_mass[k][d]     += s * L[t][d]
    background_mass[k][d] += s * Σ_v L[v][d]
```

**Derive `DepthWeights` for hop-class `k`:**
```
raw[d]  = (answer_mass[k][d] + κ·prior_answer[k][d])
        / (background_mass[k][d] + κ·prior_background[k][d] + ε)
c[d]    = raw[d] / max_d raw[d]              // max-normalize into [0,1]
if all c[d] == 0: c = terminal(k)            // never emit an invalid DepthWeights
```
The **prior is `terminal(k)`**: `prior_answer[k] = terminal(k)`,
`prior_background[k]` uniform. With `κ` large and no evidence, `raw ∝ terminal(k)` and
`c == terminal(k)` — cold start is exactly the shipped 0.381 behavior. Evidence
smooths it toward the empirical answer/background ratio, which up-weights depths where
rewarded answers concentrate relative to distractors and keeps shortcut-distance
answers that hard `terminal(k)` zeroes.

`DepthProfileConfig { prior_strength: f32, floor: f32, rebuild_every_n: u32 }`.
Persistence: a sidecar mirroring `TransitionStore`'s (magic, version, per-`k`
accumulators, config), atomic `write-temp-then-rename`.

## Component #4 — Hits@1 reranker (`Reranker`)

An online linear model in `rgdb/src/reranker.rs`.

**Features `φ(v)`** for a candidate `v` in the top-K, from the layered result:
- `L[v][0..=max_depth]` — the arrival-depth profile (max_depth+1 features)
- `ln(1 + score(v))` — the collapsed diffusion score
- `ln(1 + out_degree(v))` — hubs are rarely specific answers
- one-hot of `v`'s **dominant incoming relation** (argmax over the relation that
  delivered the most mass to `v`; the layered pass tracks this as a side table).
  Learnable without a schedule — the model learns which incoming relations correlate
  with correct answers.

All non-one-hot features are standardized (running mean/var) for stable SGD.

**Score & rerank:** `s(v) = w·φ(v) + b`; the engine reorders the top-K by descending
`s`. **`w = 0, b = 0` at cold start ⇒ constant `s` ⇒ the top-K order is unchanged**
(stable sort) — a strict no-op until trained.

**Online update (on feedback):** pointwise logistic SGD over the query's top-K —
the rewarded target is label 1, the other top-K candidates are label 0:
```
for v in top_k:
    y = 1 if v == target else 0
    p = sigmoid(w·φ(v) + b)
    w += lr * (y - p) * φ(v);  b += lr * (y - p)
```
`RerankerConfig { learning_rate: f32, top_k: usize, l2: f32 }` (L2 shrinkage per
step). Persisted in the same sidecar family (weights vector + standardizer state).

**Scope boundary:** the reranker only *reorders* the retrieved top-K — it cannot pull
in a node diffusion ranked below K. That is #2's job (soft `c` changes the scores that
determine the top-K). They are complementary: #2 → recall, #4 → precision-at-1.

## Engine integration

`RgdbEngine` gains a `DepthProfileStore` and a `Reranker` alongside `TransitionStore`,
and the query path becomes:

```
query(seeds, query_relation, hop_hint: Option<usize>, params):
    c        = hop_hint.map(|k| depth_profile.weights_for(k))     // learned soft c
                        .unwrap_or(config default)                 // else current behavior
    layered  = propagate_layered(graph, vocab, seeds, query_relation, params)
                                                                  // raw per-depth; c NOT applied here
    scored   = collapse(layered, c)                               // Σ_d c[d]·L[v][d]
    top_k    = top_k(scored)
    ranked   = reranker.rerank(top_k, layered, graph)             // no-op at cold start
    cache QueryContext { …, hop_hint, layered }                   // for feedback
    return ranked
```

`record_feedback(query_id, target, signal)` additionally:
- updates `DepthProfileStore` from `ctx.layered` and `ctx.hop_hint`,
- updates `Reranker` from `ctx.layered` and the cached top-K,
both inside the existing single critical section, with the same
`rebuild_every_n`/atomic-swap discipline as the transition rebuild. Feedback trains
under the weights/profile captured at query time (the existing "credit under
query-time state" rule), so `ctx` must carry the layered result and hop hint.

`QueryContext` grows by the layered map (bounded by the reached ball, ~789 nodes on
MetaQA) and the hop hint. The LRU/TTL bounds memory as today.

### API changes

- `RgdbEngine::query` gains `hop_hint: Option<usize>` (additive; `None` = today's
  behavior). Python `Engine.query` gains `hop_hint=None`.
- `propagate_layered` is exposed in the Python bindings for the eval harness:
  `core.propagate_layered(...) -> list[(node, list[float])]`.

## Non-goals / do-no-harm

- No new heavy ML dependency; both learners are hand-written linear/ratio updates.
- Cold start of BOTH components reproduces the current shipped behavior exactly
  (soft `c` == `terminal(k)`; reranker == identity). A deployment that never sends
  feedback sees no change.
- #1 is not built (entropy gate failed). #3 stays deferred.
- Fold in one deferred hygiene item from the depth-aware final review while touching
  these files: add the length `debug_assert` to `credit()` / `backward_mass_at_target`
  to match `propagate_single`.

## Testing strategy

Rust unit tests:
- **Primitive consistency:** `Σ_d c[d]·layered[v][d] == propagate(v)` for `c = uniform`
  and `c = terminal(k)` on the existing chain / multi-path fixtures, exact at
  `min_intensity = 0`.
- **Layered decomposition:** on the `0→1→2→3` chain, `layered[3] = [0,0,0,0.614…]`
  (node 3 arrives only at depth 3).
- **`DepthProfileStore` cold start:** `weights_for(k)` with no evidence equals
  `terminal(k)` exactly.
- **`DepthProfileStore` learning:** synthetic feedback where the rewarded target's
  mass sits at depth 2 while background is flat → derived `c` peaks at depth 2.
- **`Reranker` cold start:** `w=0` leaves the top-K order unchanged (identity).
- **`Reranker` learning:** synthetic feedback where the answer is always the
  lowest-degree candidate → after N updates the reranker ranks low-degree first.
- **Engine:** a query with `hop_hint` and no feedback returns the same ranking as
  today; after feedback, the learned components change the ranking; feedback trains
  under the query-time layered result (not a later one).

Python acceptance (`rgdb-eval`, full test set, trained matrix):
- **#2 gate:** learned soft `c` (replayed from training feedback) achieves 3-hop
  recall@20 ≥ `terminal(3)`'s and does not regress MRR.
- **#4 gate:** the reranker (replayed from training feedback) lifts 3-hop **Hits@1
  above 0.203** while not regressing recall@20.
- Report honestly if either gate is not met (as with the depth-weight acceptance).

## Phasing

One spec, three phases in the implementation plan, each independently testable:
1. Layered primitive + binding + consistency tests.
2. `DepthProfileStore` + engine `hop_hint` wiring + soft-`c` gate.
3. `Reranker` + engine rerank wiring + Hits@1 gate.
