# Query-Conditioned Reranker (Follow-up #3) — Design

**Date:** 2026-07-11
**Branch:** extends `feature/rgdb-self-learning-loop`
**Status:** approved design, pending implementation plan

## Summary

Give the online `Reranker` a **query-conditioned** feature: whether a candidate's
dominant incoming relation equals the query's **expected final relation** (the last
element of the per-query `schedule`). This is the signal the reranker lacked — its
existing features are global (query-agnostic), which is why it regressed 3-hop Hits@1.
The match feature is inert when no schedule is present, so the do-no-harm default is
preserved and the reranker stays off by default.

## Why (validated by a measure-first gate)

`rgdb-eval/scripts/experiment_reranker_schedule_gate.py` (commit `f41bc46`) trained an
offline logistic reranker over the layered output on **degraded** schedules (each
relation corrupted with probability `p`, a proxy for imperfect prediction) and measured
3-hop Hits@1:

| corruption `p` | schedule-alone | reranker + match feature |
|---:|---:|---:|
| 0.00 | 0.779 | **0.872** |
| 0.25 | 0.457 | **0.556** |
| 0.40 | 0.319 | **0.429** |

The match feature recovers **+9–12 Hits@1 points** across the degraded regime, and the
gap does not collapse as corruption grows — so its value is real in the realistic
(imperfect-schedule) regime, not just the easy case.

**Load-bearing finding (gate concern #2):** the reranker *without* the match feature
underperforms schedule-alone at **every** `p`, including `p=0`. So the existing global
features (log out-degree, incoming-relation one-hot) are net-negative in isolation —
the match feature is what flips the reranker from harmful to beneficial, not a marginal
add.

**Honesty caveat:** the corruption is synthetic uniform relation substitution, a proxy
for a real predictor's (correlated) error pattern, not a simulation of it. MetaQA is
templated. This gate isolates whether the match feature carries signal in the degraded
regime; it does.

## Component 1 — the match feature and expected-relation threading

`Reranker` (`rgdb/src/reranker.rs`) gains the query's expected final relation as an
input to feature extraction and training:

```rust
pub fn features(&self, node, layered, graph, expected_relation: Option<RelationId>) -> Vec<f32>;
pub fn rerank(&self, candidates, layered, graph, expected_relation: Option<RelationId>);
pub fn update(&mut self, candidates, target, layered, graph, expected_relation: Option<RelationId>, signal);
```

The appended feature (the LAST feature slot, so `dim` grows by 1 to
`(max_depth+1) + 2 + n_relations + 1`):

```rust
let match_feat = match expected_relation {
    Some(er) => if layered.dominant_incoming.get(&node) == Some(&er) { 1.0 } else { 0.0 },
    None => 0.0,
};
```

`expected_relation = None` (no schedule, or an empty schedule) → the feature is
constantly `0.0`, contributing nothing — so the reranker's behavior is exactly its
pre-#3 behavior when no schedule is supplied.

**Feature-set decision.** This build ADDS the match feature to the EXISTING feature set
(the configuration the gate validated at 0.872 / 0.556 / 0.429). The gate's concern #2
(the global features are net-negative in isolation) is recorded as a follow-up: a *lean*
feature set (per-depth + log-score + match, dropping log-degree and the incoming
one-hot) is a candidate simplification, but it was NOT the configuration measured
through-the-engine here, so it is not shipped now. The acceptance script reports a lean
variant's offline number for reference so the pruning decision is data-backed when taken.

## Component 2 — engine wiring

`RgdbEngine::query`: derive the expected relation from the schedule already in params and
pass it to the reranker (only relevant when `reranker_enabled`):

```rust
let expected_relation = resolved.schedule.as_ref().and_then(|s| s.last().copied());
// ... in the reranker block:
rr.rerank(&mut head, &layered, &self.graph, expected_relation);
```

`record_feedback`: the `QueryContext` already caches `params` (which carries the
schedule), so the expected relation is re-derivable at feedback time:

```rust
let expected_relation = ctx.params.schedule.as_ref().and_then(|s| s.last().copied());
rr.update(&ctx.ranked_topk, target, &ctx.layered, &self.graph, expected_relation, signal);
```

No new `QueryContext` field is needed (the schedule rides in the cached `params`). No new
engine `query`/`record_feedback` argument. The reranker gate (`reranker_enabled`, default
false) is unchanged.

## Do-no-harm

- Reranker still off by default (`EngineConfig.reranker_enabled = false`).
- `w = 0` cold start is still a strict identity (the extra feature has weight 0 too).
- With no schedule, the match feature is `0.0` — the reranker behaves exactly as before
  #3 (so this change cannot regress the no-schedule path, enabled or not).
- No new Rust dependency; no binding signature change for callers (the schedule already
  flows through the existing `schedule` kwarg; the reranker reads it internally).

## Testing strategy

Rust unit tests (`reranker.rs`):
- **Match feature fires:** on a fixture where a candidate's `dominant_incoming` equals
  `expected_relation`, `features(...)` has `1.0` in the match slot; a non-matching
  candidate has `0.0`; `expected_relation = None` gives `0.0` for all.
- **Cold-start identity preserved:** `w = 0` leaves order unchanged even with an
  `expected_relation` supplied (the new feature has weight 0).
- **Learns the match signal:** synthetic feedback where the gold answer is the only
  candidate whose incoming relation matches `expected_relation` → after training, the
  reranker ranks it first (the match feature gets positive weight).
- **`dim` and persistence:** `save`/`load` roundtrip with the new feature width.

Engine test (`engine.rs`):
- With `reranker_enabled = true` and a schedule set, feedback trains the reranker under
  the expected relation from `ctx.params.schedule`; a matching candidate is promoted.

Python acceptance (`rgdb-eval`):
- `experiment_reranker_schedule_api.py`: through the ENGINE (`reranker_enabled` on),
  train on degraded 3-hop schedules via `record_feedback`, evaluate 3-hop Hits@1 vs
  schedule-alone at `p ∈ {0.0, 0.25, 0.4}`. Gate: reranker+match beats schedule-alone
  at `p = 0.25` (target ≈ 0.556 from the offline gate; allow a floor of 0.50 for
  online-vs-offline variance). The results doc reports the offline lean-variant number
  for the pruning follow-up and carries the synthetic-corruption caveat.

## Non-goals

- No lean feature-set switch shipped (recorded as a follow-up; acceptance reports its
  offline number).
- No change to the schedule mechanism, depth weights, or the off-by-default gates.
- No production text→schedule predictor (caller-side; the reference predictor exists).

## Phasing (for the plan)

1. `Reranker`: add `expected_relation` to `features`/`rerank`/`update`, append the match
   feature, bump `dim`/persistence; unit tests (match fires, identity preserved, learns
   the signal, roundtrip).
2. Engine: derive `expected_relation` from the schedule in `query` and `record_feedback`,
   pass to the reranker; engine test.
3. Eval: `experiment_reranker_schedule_api.py` acceptance through the engine + results
   doc (with the lean-variant reference and the caveat).
