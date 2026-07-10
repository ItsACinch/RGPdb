# Data-Weighted Transitions + Self-Learning Query Loop — Design Spec

**Date:** 2026-07-10
**Status:** Approved for planning (design sections 1–3 approved by user; sections 4–6
approved by delegation, with two open questions resolved autonomously — see
"Decisions made without the user" below and please review).
**Branch:** `feature/rgdb-self-learning-loop`
**Defers:** [Option C — query-conditioned per-hop refraction](./2026-07-10-deferred-query-conditioned-refraction.md)

---

## Motivation

Refraction penalizes relation *changes*. Measured on MetaQA (1000 questions/hop,
query relation from gold qtypes):

| hop | name-embedded refraction | trained transition matrix | untyped-ppr |
|-----|-------------------------:|--------------------------:|------------:|
| 1 (MRR) | **0.999** | 0.992 | 0.956 |
| 2 (MRR) | 0.200 *(worst)* | **0.275** *(best)* | 0.219 |
| 3 (MRR) | 0.210 | 0.227 | **0.279** |

The mechanism is sound but **untrained**: a static, name-derived similarity matrix
has no notion of which relation transitions matter for the task, so it penalizes the
compositional steps multi-hop reasoning requires. Rebuilding that matrix from
observed reasoning transitions took 2-hop MRR from worst (0.200) to best (0.275)
while preserving the 1-hop win.

This spec productionalizes that result (**A**) and closes the loop so the engine
learns from its own use (**B**).

## Scope

**In:**
- **A.** The relation-transition matrix as a first-class, persisted, queryable
  artifact — not an experiment script.
- **B.** A self-learning loop: applications push feedback into the database; the
  engine attributes credit to relation transitions and sharpens its matrix.

**Out (deferred):**
- **C.** Query-conditioned per-hop relation schedules / RotatE composition, for the
  residual 3-hop gap. Unvalidated. See the linked doc — it carries a mandatory
  validation experiment that must pass before any of it is designed. Note that **B
  is a prerequisite for C**: the credit pass recovers the reasoning chains a
  schedule predictor would train on.

---

## Architecture

Three new units in the Rust core, each independently testable.

### `transitions.rs` — `TransitionStore`
Owns the learned state. Knows nothing about graphs or queries — pure state + math.

```rust
pub struct TransitionConfig {
    pub prior_strength: f32,   // κ: pseudo-count mass on the prior. default 10.0
    pub floor: f32,            // ε: min off-diagonal similarity. default 0.05
    pub decay: f32,            // γ: multiplicative decay on rebuild. default 1.0 (off)
    pub rebuild_every_n: u32,  // auto-refresh cadence. default 64; 0 = manual only
}

pub struct TransitionStore { /* counts: Vec<f32> (n*n), prior: Vec<f32>, names, config */ }

impl TransitionStore {
    pub fn new(relation_names: Vec<String>, prior: Option<Vec<f32>>, cfg: TransitionConfig) -> Result<Self, TransitionError>;
    pub fn record(&mut self, credits: &[(RelationId, RelationId, f32)], signal: f32);
    pub fn rebuild(&self) -> RelationVocab;
    pub fn save(&self, path: &str) -> Result<(), TransitionError>;
    pub fn load(path: &str) -> Result<Self, TransitionError>;
    pub fn events_since_rebuild(&self) -> u32;
}
```

**Derivation rule (counts → similarity matrix).** The prior enters as pseudo-counts,
which makes cold start fall out for free:

```
C'[a][b] = C[a][b] + κ · P[a][b]          // pseudo-count blend
M[a][b]  = C'[a][b] / max_b C'[a][b]      // row-max normalize
M[a][a]  = 1.0                            // diagonal pinned
M[a][b]  = max(M[a][b], ε)   (a != b)     // floor: never fully kill a path
```

With the default uniform prior (`P` all-ones) and zero evidence, every row
normalizes to all-ones → the matrix is **exactly** uniform → pure typed PPR, zero
refraction penalty. **Do no harm before you have evidence.** As credit accumulates,
`C` dominates `κ·P` and the matrix converges on the learned transitions. `κ` is
literally "how much evidence before I stop trusting the prior."

The diagonal must stay pinned at 1.0: `sim(query_relation, same_relation) = 1` is
what makes the 1-hop case work (0.999). Never learn it away.

Signals are **signed**: `C[a][b] += signal · credit[a][b]`, clamped to `≥ 0`. An
application can report "this answer was wrong" as negative reinforcement.

### `credit.rs` — the backward credit pass
Pure function, no state. The novel core.

```rust
/// Returns L1-normalized per-transition credit (sums to 1.0 when any flow exists).
pub fn credit(
    graph: &Graph, vocab: &RelationVocab,
    seeds: &[(NodeId, f32)], query_relation: Option<RelationId>,
    target: NodeId, params: &PropagationParams,
) -> Vec<(RelationId, RelationId, f32)>;
```

This is the **forward–backward algorithm**, and it is exact — not a heuristic.

- Forward: `F_k[(u, r_in)]` = mass at state `(u, r_in)` after `k` hops. `propagate`
  already computes this internally; `credit` recomputes it retaining per-depth
  frontiers.
- Backward: `B_j[(v, r)]` = total weight of all continuations from state `(v, r)`
  reaching `target` within `j` more hops. Base case `B_0[(target, ·)] = 1`.

Flow through edge `u→v` carrying transition `(r_in → r_out)` at hop `k+1`:

```
flow = F_k[(u, r_in)] · w(u→v, r_in→r_out) · B_{maxdepth−k−1}[(v, r_out)]
w    = reflection(u) · p(u→v) · sim(r_in, r_out)^refraction_index(u)
```

Summing over all edges and grouping by `(r_in, r_out)` yields flow per transition,
in exact proportion to how much of the target's mass traversed that transition.

**Credits are L1-normalized:** `credit[a][b] = flow[a][b] / Σ flow`. Each feedback
event therefore contributes exactly `signal` total evidence, so a high-mass target
cannot dominate the counts simply by being well-connected. If `Σ flow == 0` the
target is unreachable and `credit()` returns empty.

**Correctness invariant (the property test).** Note that `Σ flow` is *not* the
target's mass — the decomposition counts each path once per edge, so `Σ flow` equals
the **length-weighted** mass `Σ_paths length·weight`. The clean cross-check between
the two passes is instead:

```
Σ_seeds  seed_mass · B_maxdepth[(seed, query_relation)]  ==  propagate(...)[target]
```

The backward value at the seed state is by construction the total weight of every
seed→target path within `max_depth` — i.e. exactly the forward mass at the target.
Asserting this ties the backward walk to the already-tested forward kernel and
catches essentially any bookkeeping error in either direction.

The backward walk needs **in-edges**, so `Graph` gains a lazily-built, cached reverse
CSR (`O(E)` once). This is the only structural change to an existing type. The credit
pass runs **only on feedback** — normal queries pay nothing.

### `engine.rs` — `RgdbEngine`
The database surface. Composes `Graph`, the live vocab, the store, and a bounded
query-context cache.

```rust
pub type QueryId = u64;  // monotonic, engine-issued

pub struct QueryResult { pub ranked: Vec<(NodeId, f32)>, pub query_id: QueryId }

pub struct RgdbEngine {
    graph: Graph,
    vocab: ArcSwap<RelationVocab>,           // live matrix; lock-free reads
    store: Mutex<TransitionStore>,           // counts; tiny, writes serialized
    cache: Mutex<LruCache<QueryId, QueryContext>>,  // bounded + TTL
    next_id: AtomicU64,
}

impl RgdbEngine {
    /// `vocab_prior` supplies BOTH the relation names and the prior matrix.
    /// Pass `RelationVocab::with_names_uniform(names)` for the do-no-harm default.
    pub fn new(graph: Graph, vocab_prior: RelationVocab, cfg: TransitionConfig) -> Self;
    pub fn query(&self, seeds: &[(NodeId, f32)], query_relation: Option<RelationId>, params: &PropagationParams) -> QueryResult;
    pub fn record_feedback(&self, query_id: QueryId, target: NodeId, signal: f32) -> Result<(), FeedbackError>;
    pub fn refresh(&self);
    pub fn save(&self, path: &str) -> Result<(), TransitionError>;
    pub fn load(graph: Graph, path: &str) -> Result<Self, TransitionError>;
}
```

- **Interior mutability, so `query` and `record_feedback` both take `&self`** and the
  engine is `Sync`: the live vocab is an `ArcSwap<RelationVocab>` (queries clone a
  cheap `Arc`; the swap is lock-free), while the counts and the query cache sit behind
  short-lived `Mutex`es. The matrix is `n_rel²` floats (324 for MetaQA), so rebuilds
  are effectively free — cadence is a *semantics* decision, never a perf one.
- `RelationVocab` gains two additive accessors so the store can be built from a prior
  and re-serialized: `names(&self) -> &[String]` and `matrix(&self) -> &[f32]`.
- `TransitionStore::new(names, prior_matrix, cfg)` is what `RgdbEngine::new` calls
  after decomposing `vocab_prior`.
- `propagate` is **unchanged** — it still takes `&RelationVocab`. The hot path is
  untouched, and the seam for deferred C stays open.
- `refresh()` rebuilds and atomically swaps. It also fires automatically every
  `rebuild_every_n` feedback events.

## Data flow

```
app: query(seeds, query_relation)  ──►  propagate(graph, live_vocab, …)
                                        └─► QueryResult { ranked, query_id }
                                              (context cached under query_id)

app: record_feedback(query_id, target, signal)
        └─► resolve context (seeds, query_relation, params)
        └─► credit()  ── forward+backward ──► per-transition credits
        └─► store.record(credits, signal)      // C += signal·credit, clamp ≥0
        └─► if events_since_rebuild ≥ rebuild_every_n → refresh()

refresh(): vocab' = store.rebuild();  live_vocab.swap(vocab')
save():    counts + prior + config + relation names + checksum → sidecar
```

## Persistence

**Sidecar snapshot** `<level>.transitions`: magic, version, `n_relations`, relation
**names**, counts matrix, prior matrix, config, checksum. Written temp-then-rename so
a crash mid-write cannot corrupt it. The relation names are validated against the
graph's vocabulary on load — a transition matrix is meaningless against a different
relation set, and silently accepting one would corrupt every subsequent query.

**The level file stays immutable.** It is build-once and mmap'd; learned state is
small, hot, and mutable. Mixing them would destroy that property.

**Optional append-only feedback log** `<level>.feedback`: `(timestamp, seeds,
query_relation, target, signal)` records. Enables re-deriving the matrix later under a
different `κ`/`ε`/decay policy without having lost history. Off by default.

## Query-context cache — and its consequences

`query()` returns an opaque `query_id`; contexts live in a bounded LRU with a TTL.
This buys the ergonomic feedback API. It costs three things, stated plainly:

1. Feedback after eviction/TTL returns `FeedbackError::UnknownQuery` — an **explicit
   error, never a silent drop**.
2. In-flight contexts are **lost on restart**. Already-recorded counts are safe (they
   are persisted); only un-acknowledged queries vanish.
3. It is **single-process**: a `query_id` from one process will not resolve in another.

The escape hatch is additive and deliberately not built now (YAGNI): `credit()` is
already stateless, so a `record_feedback_with_context(seeds, query_relation, target,
signal)` overload is a thin wrapper if any of the above bites.

## Error handling

| Error | When | Behavior |
|---|---|---|
| `FeedbackError::UnknownQuery(id)` | evicted / expired / never issued | return error; record nothing |
| `FeedbackError::InvalidTarget(node)` | target out of graph bounds | return error; record nothing |
| `FeedbackError::TargetUnreachable` | no seed→target path within `max_depth` (credits all zero) | return error; record nothing |
| `TransitionError::VocabMismatch` | sidecar relation names ≠ graph relations | refuse to load |
| `TransitionError::BadShape` | counts/prior not `n×n` | refuse to construct |

Nothing is ever silently dropped. A feedback event either updates counts or returns
an error saying why it didn't.

## Testing

- **Unit — `TransitionStore`:** derivation rule (pseudo-count blend, row-max
  normalization, pinned diagonal, floor); **uniform prior + zero evidence yields an
  exactly all-ones matrix** (cold start == typed PPR); signed signals clamp at 0;
  save/load roundtrip; vocab-mismatch refusal.
- **Unit — `credit()`:** hand-computed flows on the tiny graphs the kernel tests
  already use (chain, branch, multi-path); credits are L1-normalized (sum to 1.0);
  an unreachable target yields empty credits; plus the **forward/backward
  cross-check property**: `Σ_seeds seed_mass · B_maxdepth[(seed, query_relation)]`
  equals `propagate(...)[target]`, asserted on several shapes. That invariant ties
  the new backward walk to the already-tested forward kernel.
- **Unit — reverse CSR:** in-edges match a brute-force scan of out-edges.
- **Integration — `RgdbEngine`:** on the synthetic planted-path probes,
  `query → record_feedback → refresh` shifts ranking in the predicted direction; a
  `refresh()` with no feedback is a no-op; `record_feedback` on an evicted id errors.
- **Acceptance — MetaQA replay:** replay gold training chains through
  `record_feedback` as synthetic feedback events, then verify the **online-learned**
  matrix reproduces the **offline-trained** result (2-hop MRR ≈ 0.275, 1-hop ≈ 0.99).
  This validates the whole loop end-to-end against numbers already independently
  measured.

## Decisions made without the user (please review)

The user delegated the remaining approval. Two questions from design section 4–6 were
left open and are resolved here:

1. **`rebuild_every_n` defaults to `64`, not manual-only.** Rationale: the stated
   value was "query behavior changes only at *known points*," and every-N-events is a
   deterministic, known point. Defaulting to manual (`0`) would mean a caller wires up
   feedback, records thousands of events, and never observes learning — a surprising
   footgun. `0` remains available for full manual control.
2. **The `query_id` cache trade-offs are accepted as specified** (explicit
   `UnknownQuery` error, restart loses in-flight contexts, single-process). The
   stateless overload is *not* built — the user explicitly chose the cached `query_id`
   option over "both," and the seam is trivial to add later.

## Non-goals

- Option C (per-hop schedules / RotatE). Deferred, with a validation gate.
- Learning the intent→`query_relation` mapping. `query_relation` continues to come
  from the caller or the existing intent classifier.
- A REST/serving API. `RgdbEngine` is a library surface; a server can wrap it.
- Distributed / multi-process feedback. Single-process, per the cache decision.
