# Query-Conditioned Schedule Seam (Production #3) — Design

**Date:** 2026-07-11
**Branch:** extends `feature/rgdb-self-learning-loop`
**Status:** approved design, pending implementation plan
**Supersedes the build half of:** `2026-07-10-deferred-query-conditioned-refraction.md` (Option C)

## Summary

Make a per-query relation **schedule** a first-class scoring input to the kernel: at
hop `k`, an edge scores 1.0 if its relation matches `schedule[k]`, else a small floor,
replacing the vocab similarity for scheduled hops. This is the deployable half of
feature #3 — the mechanism that turns a predicted reasoning chain into a ranking. The
**predictor is caller-side** (the application supplies the schedule); the reference
implementation is a MetaQA question-text classifier in the eval harness. RotatE / a
production text predictor is explicitly out of scope.

## Why (validated)

The measure-first gate (`rgdb-eval/scripts/experiment_predicted_schedule.py`, commit
`7b19f8e`) established, on MetaQA 3-hop full test set, with `depth_weights = terminal(3)`
held constant:

| condition | 3-hop MRR |
|---|---:|
| baseline (deployed terminal(3), no schedule) | 0.381 |
| predicted-schedule (100% classifier) | **0.878** |
| gold-schedule | 0.878 |

Degradation sweep (corrupt each schedule relation independently with prob `p`):

| `p` | 3-hop MRR |
|---|---:|
| 0.0 | 0.878 |
| 0.25 | 0.460 |
| 0.5 | 0.235 |

The mechanism stays above the 0.381 baseline through ~25–40% per-relation schedule
error and crosses below near `p=0.5`. So the schedule mechanism is **robust, not
brittle** — it needs a predictor better than ~60–75% per-relation accuracy to be a net
win. **Honesty caveat (carried into the results doc):** MetaQA has 15 fixed, distinctly
phrased 3-hop templates, so the reference classifier's ~100% accuracy is a
template-matching artifact, NOT evidence that real open-ended question→schedule
prediction is easy. The degradation robustness is the transferable evidence; the
predictor's real-world accuracy is application-specific and untestable on MetaQA.

`0.878` is the *deployable* ceiling (schedule + the shipped `depth_weights`, no oracle
mask). The higher `0.927` seen earlier used an oracle graph-distance mask and is not
deployable.

## Component 1 — the schedule parameter and kernel scoring

`PropagationParams` gains a field (mirroring how `depth_weights` was added, so the
`propagate` / `propagate_layered` / `credit` signatures do NOT change):

```rust
pub struct PropagationParams {
    pub max_depth: usize,
    pub min_intensity: f32,
    pub depth_weights: Option<DepthWeights>,
    /// Per-hop expected relation chain. `schedule[k]` is the relation expected at hop
    /// `k` (first hop = index 0). `None` = today's behavior (vocab similarity). When
    /// `Some`, it REPLACES vocab similarity for scheduled hops. Length may be < or >
    /// `max_depth`; hops past its end fall back to the vocab.
    pub schedule: Option<Vec<RelationId>>,
}
```

In the per-hop relation scoring of BOTH `propagate_single` and `propagate_layered`, the
`sim_term` computation becomes:

```rust
const SCHEDULE_FLOOR: f32 = 0.05; // matches the validated gate; module constant, not configurable (YAGNI)

let sim = match r_in {
    Some(a) => vocab.similarity(a, ep.relation),
    None => 1.0,
};
let sim_term = match params.schedule.as_deref() {
    Some(s) if depth < s.len() => {
        if ep.relation == s[depth] { 1.0 } else { SCHEDULE_FLOOR }
    }
    _ => if rix == 1.0 { sim } else { sim.powf(rix) },
};
```

Notes:
- `depth` is the kernel's existing hop index (0 = first hop from the seed). `schedule[depth]`
  is the expected relation for that hop.
- When a schedule is set, `query_relation` and the refraction exponent are unused for
  scheduled hops — the schedule is trusted over the global/vocab relation model. This is
  the point: the schedule is query-specific and overrides the global matrix.
- `schedule = None` yields `sim_term` identical to today (bit-for-bit): the `match` falls
  to the existing `sim`/`sim.powf(rix)` path.
- The schedule enters the RAW `transmitted` mass (it is a relation-scoring factor), so
  `propagate_layered` applies it too and its per-depth output already reflects it. The
  `Σ_d c[d]·layered[v][d] == propagate(v)` consistency invariant continues to hold with
  a schedule set (both apply the same `sim_term`).
- Composition with `depth_weights` is orthogonal: `sim_term` scores the relation,
  `depth_weights[depth+1]` scores the arrival depth; the readout multiplies them.
- Pruning still tests raw `transmitted` (unchanged).

## Component 2 — credit / transition learning stays schedule-agnostic

`credit()` and `backward_mass_at_target()` IGNORE `params.schedule`. Rationale: the
transition store learns a **global** relation-similarity matrix; a per-query schedule is
a different, overriding mechanism, and entangling the two would complicate the credit
math for no deployable benefit (when you have a schedule you are not relying on the
global matrix). Documented consequence: the forward↔backward credit invariant
(`Σ_seeds mass · Σ_L c_L · B_exact[L][(seed, qr)] == propagate(...)[target]`) holds only
when `schedule = None` (in addition to the existing `min_intensity == 0` condition). A
one-line doc note on `credit()` / `backward_mass_at_target()` states this. No logic
change to `credit()` is required — it simply never reads the new field.

## Component 3 — engine, bindings, reference predictor

**Engine:** `RgdbEngine::query` already takes `&PropagationParams`, so a caller supplies
the schedule via params — no new `query` argument. The existing depth-weight resolution
rebuilds params to inject learned/`hop_hint` weights; it MUST preserve `params.schedule`
(carry it through the rebuilt `PropagationParams`). `QueryContext` already caches the
resolved params, so feedback (should it be sent) sees the schedule that ran — though per
Component 2, transition credit ignores it.

**Bindings:** `propagate` and `Engine.query` gain a `schedule=None` kwarg
(`list[int] | None`), placed LAST (after the existing kwargs) so existing positional
callers are unaffected. It is validated only for relation-id range implicitly (ids are
`u16`; out-of-vocab ids simply never match an edge and score the floor — no error path
needed). `propagate_layered`'s binding may also take `schedule` for eval use.

**Reference predictor (eval harness only):** the caller-side reference is the MetaQA
question-text classifier — `sklearn` `TfidfVectorizer` (char+word n-grams) +
`LogisticRegression`, with the topic entity masked to a constant token — mapping a
question to its qtype, then `qtype_to_relation_sequence` to a schedule. It lives in
`rgdb-eval` as the reference/test predictor; it is NOT part of the Rust core, and the
core takes no ML dependency.

## Do-no-harm

- `schedule = None` everywhere → behavior identical to the current kernel (bit-exact for
  the scoring path).
- No new Rust dependency. The reference predictor's `sklearn` dependency is confined to
  the eval harness (already used there for the gate).

## Testing strategy

Rust unit tests (`propagation.rs`):
- **`schedule=None` bit-identical:** on the existing fixtures, `propagate_single` with
  `schedule: None` equals the pre-feature output exactly (`assert_eq!`).
- **Schedule scores by hop:** on `0 -A-> 1 -B-> 2`, a schedule `[A, B]` gives node 2 the
  full decayed mass (both hops match); a schedule `[A, A]` floors the second hop (node 2
  gets `refl^2 · p · SCHEDULE_FLOOR`), a hand-computed exact value.
- **Fallback past schedule length:** a length-1 schedule scores hop 0 by the schedule and
  hop 1 by the vocab (verify hop-2 node uses vocab sim, not the floor).
- **Composition with `depth_weights`:** schedule `[A,B]` + `terminal(2)` scores only the
  depth-2 node, at the schedule-weighted mass.
- **Layered consistency with a schedule:** `Σ_d c[d]·layered[v][d] == propagate(v)` with a
  schedule set, at `min_intensity=0`.

Python (`rgdb-eval`):
- **Binding roundtrip:** `core.propagate(..., schedule=[...])` runs and a matching
  schedule reproduces the expected ranking; `schedule=None` matches the no-schedule call.
- **Acceptance gate:** `experiment_schedule_api.py` — build the reference predictor,
  produce predicted schedules, score 3-hop test THROUGH the production `propagate`/engine
  `schedule` param with `depth_weights=terminal(3)`, and assert 3-hop MRR > 0.381
  (expected ~0.878). Writes `results/metaqa-schedule-api.md` with the numbers and the
  templating caveat. If it does NOT beat 0.381, report the number — do not tune.

## Non-goals

- No RotatE / GNN / learned production predictor. The seam is predictor-agnostic; the
  application supplies schedules.
- No schedule-aware credit / transition learning (Component 2).
- No new engine `query` argument (schedule rides in `PropagationParams`).
- `SCHEDULE_FLOOR` is a fixed constant, not configurable.

## Phasing (for the plan)

1. Kernel: `PropagationParams.schedule` + scheduled `sim_term` in `propagate_single` and
   `propagate_layered`; unit tests (bit-exact None, scheduling, fallback, composition,
   layered consistency); credit doc note. Patch the `PropagationParams { .. }` literals.
2. Bindings: `schedule` kwarg on `propagate` (+ `propagate_layered`) and `Engine.query`;
   engine param-resolution preserves `schedule`; binding roundtrip test.
3. Eval: reference predictor module + `experiment_schedule_api.py` acceptance gate + a
   results doc with the caveat.
