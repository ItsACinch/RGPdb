# RGDB Core Redesign — Design Spec

**Date:** 2026-07-09
**Status:** Approved for planning
**Scope:** Three sub-projects addressing the 5-point review of RGDB —
(1) hygiene, (2) principled propagation semantics, (3) sparse + parallel
propagation, (4) drop/fix PVS, (5) retrieval-quality evaluation harness.

---

## Context

A code review of RGDB found that the distinctive parts of the engine (angle
bins, refraction, rooms/PVS) were redundant, unsound, or unvalidated, and that
the propagation kernel computed an order-dependent, buggy variant of
personalized PageRank (PPR). There was no retrieval-quality evaluation anywhere
in the repo (benchmarks measured only speed), and the test suite did not
compile.

The five recommendations from that review are grouped here into three
sub-projects, sequenced **Hygiene → Eval → Core rewrite** so that every
behavioral change to the model is measured against a captured baseline.

### Decisions locked during brainstorming

| Topic | Decision |
|-------|----------|
| Sequencing | Hygiene → Eval → Core rewrite |
| Backward compat | Free rein: break Rust API + file format, bump version, update Python bindings |
| Influence semantics | Sum over paths (PPR / heat-kernel family) |
| Relation model | Path-internal refraction: `sim(prev_edge_relation, next_edge_relation)`; state = `(node, incoming_relation)` |
| Mass bounding | Random-walk normalization + per-node damping (PPR-proper) |
| Rooms / PVS | Drop PVS entirely; keep rooms / partitioning / portals |
| Eval dataset | MetaQA (external validity) + small synthetic probe suite (controlled ablations) |
| Eval harness language | Python, driving RGDB through `rgdb-python` bindings |
| CUDA | Feature-gate off, mark stale; defer sparse-GPU-PPR port to a later separate project |

---

## Sub-project A — Hygiene

**Goal:** a repo where `cargo test --workspace` is green and there is exactly
one source tree.

### Tasks
1. **Fix `rgdb/src/level_file.rs` tests.** ~8 call sites treat `Graph::new()`
   as returning `Graph`; it returns `Result<Graph, GraphError>`. Update the
   call sites, run the full suite, fix anything else that surfaces until green.
2. **Delete the stale root source tree.** Root `src/`, `main.rs`, `benches/`,
   and root `build.rs`. The root `Cargo.toml` is workspace-only (no `[package]`
   section), so these files are not compiled by the workspace. **Verification
   step before deletion:** confirm nothing references them (grep for paths;
   confirm the root `build.rs` is not wired to CUDA kernel compilation for the
   `rgdb` crate — the `rgdb` crate has its own `build.rs` if needed).
3. **Add minimal CI.** A GitHub Actions workflow running
   `cargo build --workspace` and `cargo test --workspace` on push, so a
   non-compiling test suite cannot silently recur.

### Explicitly out of scope for A
- No behavior changes.
- No dead-code cleanup in `pvs.rs` / `property_map.rs` — those files are
  deleted or replaced in sub-project B.
- No new propagation tests — the semantics change in B; the eval harness (C)
  captures the *current* model's behavior as the comparison baseline instead.

---

## Sub-project C — Evaluation harness

**Goal:** a repeatable benchmark that scores any ranking function on multi-hop
retrieval, so the core rewrite lands with a measured before/after.

Built **before** the core rewrite (per sequencing decision) so the current
model is captured as a baseline.

### Location & language
New `rgdb-eval/` Python package in the repo, driving RGDB through the
`rgdb-python` bindings (built with maturin). **Prerequisite:** a working
maturin build of `rgdb-python` on Windows.

### Datasets

**Dataset 1 — MetaQA** (external validity):
- Movie knowledge graph: ~43k entities, ~135k typed triples, 9 relation types.
- 1-hop / 2-hop / 3-hop question sets with gold answer entities.
- Sample ~1,000 questions per hop level from the test split for fast runs.
- Each question names its topic entity → **seeding is exact** (no embedding
  lookup needed to find the source node). This isolates graph-ranking quality
  from seed-retrieval quality.
- Question templates map directly to relations → eval **bypasses the keyword
  intent classifier** (deliberate isolation).

**Dataset 2 — synthetic probe suite** (controlled ablations):
- A generator that plants relation-typed paths in random graphs.
- Two probe families:
  - (a) relation-coherent paths lead to gold answers → refraction *should* win.
  - (b) incoherent paths lead to gold answers → refraction *should not* hurt
    much.
- Purpose: explain *why* results move, not just *whether*.

### Contenders

All behind one interface: `rank(seeds, query_relation, k) -> ranked node ids`.

1. Vector-only top-k (sentence-transformer embeddings of entity names).
2. Vector + uniform 2-hop expansion (the "standard stack" straw man).
3. Untyped PPR (scipy sparse, ~20 lines — the literature baseline).
4. **Current RGDB** (old propagation via existing bindings — the pre-rewrite
   baseline).
5. *(added after B)* **New RGDB** with refraction, plus its ablation
   (all-ones similarity matrix = typed PPR without refraction).

### Metrics
- Hits@k and recall@k for k ∈ {1, 5, 10, 20}, plus MRR.
- Broken down by hop count (1/2/3).
- One command produces a markdown results table committed under
  `rgdb-eval/results/`.

### Decision rule this enables
If contender 5 does not beat contender 3 (untyped PPR) **and** its own
all-ones ablation on 2/3-hop questions, the refraction layer is not paying its
complexity — and that verdict is a table, not a debate.

---

## Sub-project B — Core rewrite

**Goal:** replace the order-dependent, dense, angle-bin propagation with a
principled, sparse, calibrated typed-PPR kernel with path-internal refraction.

### B1 — Propagation kernel

**State:** `(node, incoming_relation)`. The incoming relation is the type of
the edge the walk arrived by.

**Query unification:** the query's intent relation is the *initial* incoming
relation at each seed. Thus the first hop's refraction is
`sim(query_relation, first_edge_relation)` and every later hop is
`sim(prev_edge_relation, next_edge_relation)`. Query bias and path coherence
are one mechanism.

**Transition** from state `(u, r_in)` with mass `m`, along edge `u→v` of
relation `r_out`:

```
contribution(v, r_out) += m · reflection(u) · p(u→v) · sim(r_in, r_out) ^ refraction_index(u)
```

where:
- `p(u→v) = base_weight(u→v) / Σ_w base_weight(u→w)`, and
  `base_weight = (1 − attenuation)`. Row-stochastic: `Σ_v p(u→v) = 1`
  (random-walk normalization).
- `reflection(u) ∈ [0,1]` is **per-node damping** (continuation probability);
  default **0.85**.
- `sim(r_in, r_out)` from the relation-similarity matrix.
- `refraction_index(u)` sharpens (`>1`) or softens (`<1`) the relation-turn
  penalty per node — the meaningful survival of the old `refraction_index`
  field.

**Boundedness:** outgoing mass ≤ `m · reflection(u) ≤ m`, so mass strictly
decays each hop regardless of branching factor. Total accumulated mass ≤
`m₀ / (1 − reflection_max)`, on the same scale for every query.

**Accumulation:** `I(v) = Σ over incoming relations, summed across hops` —
coherently sum-over-paths throughout. When two paths reach `(v, r_in)` at the
same depth, their masses **add** into one frontier entry, expanded once at the
next depth. This fixes both original bugs: (a) the old code mixed a *max*-update
into the frontier with a *sum* into the accumulator (incoherent,
order-dependent); (b) the old code pushed duplicate stale frontier states.

**Termination:** mass < `min_intensity` (keeps the frontier small → sparse) or
`max_depth` reached.

**Seeding (RAG path):** each vector-similarity seed injects initial mass =
its similarity; multi-source diffusion = sum of per-seed diffusions.

### B2 — Data structures

`NodeProps` collapses from 6 fields to 2:
```rust
struct NodeProps { reflection: f32, refraction_index: f32 }
```
Removed: `luminance`, `directional_luminance[16]`, `default_angle_bin`,
`relationship_property`. Nodes no longer self-emit; the query injects mass at
seeds (PPR personalization vector).

`EdgeProps`:
```rust
struct EdgeProps { attenuation: f32, relation: RelationId, is_portal: bool }
```
`angle_bin` → `relation` (`RelationId = u16`).

**New abstraction — `RelationVocab`** (replaces the hardcoded
`RelationshipProperty` enum + in-code matrix; `property_map.rs` → `relation.rs`):
```rust
struct RelationVocab {
    relations:  Vec<String>, // names; RelationId = index
    similarity: Vec<f32>,    // symmetric n×n matrix, flattened
}
```
Similarity populated three ways:
1. Explicit user-provided matrix (curated KBs).
2. Cosine of embedded relation names — the natural default; works for any
   vocabulary including MetaQA's `directed_by` / `starred_actors`.
3. All-ones off-diagonal — the **ablation** that disables refraction → pure
   typed PPR (a one-line matrix swap for the eval).

`intent.rs` stays but resolves to a relation *name* looked up in the vocab (no
longer hardwired to 10 relations); eval bypasses it via templates.

### B3 — Sparse + parallel

- **Sparse frontier:** `HashMap<(NodeId, RelationId), f32>` replaces
  `vec![0.0; n·num_bins]`. Cost ∝ reached ball, not graph size. This is why
  dropping PVS is safe: there is no O(n) pass left to prune.
- **Parallelism:** rayon across independent per-seed diffusions in the RAG
  engine, and across the ~1,000 questions in the eval harness.
- **Deferred (YAGNI):** parallelizing *within* a single diffusion (concurrent
  frontier). A depth-4 sparse walk on 43k nodes is fast serially; revisit only
  if profiling demands it.
- **Testing:** unit tests on tiny graphs with hand-computed path sums —
  closing the "core algorithm untested" finding.

### B4 — Format, bindings, CUDA, fusion

- **Level file v2:** bump version; node section 2×f32; edge section adds
  `relation:u16`; **new RelationVocab section**; **drop** PVS + angle-table +
  directional-luminance sections. Delete `pvs.rs`.
- **Rooms:** `rooms.rs`, `partitioning.rs`, portals **kept** (access control in
  `UserContext`, file layout). Only PVS is removed.
- **Python bindings:** updated to the new `NodeProps` / `EdgeProps` /
  `RelationVocab` and the new propagation signature (drops `pvs` and
  `initial_bin`; adds `query_relation`). Consumed by the eval harness.
- **CUDA:** feature-gate off, mark stale/unsupported. No kernel rewrite in this
  effort. Sparse-GPU-PPR port is a separate future spec after CPU semantics are
  validated.
- **Fusion cleanup** (`query_engine.rs`): remove max-normalization; graph
  scores are already calibrated (≈[0,1]), so `α·graph + β·vector +
  γ·personalization` is principled without it.

---

## Success criteria

1. `cargo test --workspace` green; single source tree; CI enforcing both.
2. `rgdb-eval` produces a committed markdown metrics table for all contenders,
   broken down by hop count.
3. New RGDB kernel is sparse (memory ∝ reached ball), deterministic
   (order-independent), and unit-tested against hand-computed path sums.
4. A data-supported verdict on whether path-internal refraction beats untyped
   PPR and its own no-refraction ablation on 2/3-hop retrieval.

## Non-goals

- GPU port of the new kernel (separate future project).
- REST API server (already listed as future work).
- Incremental graph updates (separate future project; current design remains
  build-once).
- Within-diffusion parallelism (deferred until profiling shows need).
