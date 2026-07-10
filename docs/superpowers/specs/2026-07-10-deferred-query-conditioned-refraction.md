# Deferred: Query-Conditioned Per-Hop Refraction ("Option C")

> **STATUS: STILL DEFERRED. The validation gate was run on 2026-07-10 and FAILED
> as written. Do not build C.** The gate also turned out to be mis-specified in
> the exact way open question #4 predicted, so the failure does not mean the idea
> is dead — it means the experiment could not test it. A replacement gate is at
> the bottom of the "Validation result" section. Read that section before doing
> anything else with this document.

**Date captured:** 2026-07-10
**Date validated:** 2026-07-10 — see "Validation result" below.
**Depends on:** the A+B design (self-learning loop) — see below, B is what makes C
*trainable*.

---

## Validation result (2026-07-10)

Scripts: `rgdb-eval/scripts/experiment_gold_schedule.py`,
`rgdb-eval/scripts/experiment_distance_artifact.py`.
Data: `rgdb-eval/results/metaqa-gold-schedule.md`,
`rgdb-eval/results/metaqa-distance-artifact.md`.
MetaQA, 1000 test questions per hop, gold chains from qtypes.

**Validity check first.** Walking the gold relation chain from the topic entity
reaches a gold answer for **100.0%** of questions at 1, 2, and 3 hops (recall
1.000). The qtype→relation mapping is correct, so the numbers below mean something.

### The gate: FAILED

| 3-hop | untyped-ppr | gold schedule (floor 0.05) | gold schedule (floor 0.0) |
|---|---:|---:|---:|
| MRR | **0.279** | 0.258 | 0.254 |

The gate said: *"if 3-hop MRR with a gold schedule does not clearly beat
untyped-PPR's 0.279, abandon C."* It does not. **C as specified is not built.**

### But the gate could not separate C from the kernel's accumulation rule

The same run shows the gold schedule *finding* the answer far better than PPR
while *ranking* it worse:

| 3-hop | untyped-ppr | gold schedule (hard) |
|---|---:|---:|
| Hits@20 | 0.611 | **0.793** |
| recall@20 | 0.262 | **0.578** |
| MRR | **0.279** | 0.254 |

`propagate()` accumulates intensity at **every visited node**, and every hop
multiplies by `reflection · p(u→v) < 1`. So a hop-1 neighbour *on the correct
chain* necessarily outranks the hop-3 answer *on that same chain*. A per-hop
schedule cannot fix this, because the nodes drowning the answer are the ones the
schedule itself is routing mass through. This is the "distance artifact" named in
open question #4, written before the data existed.

Controlling for it, with a mask that uses only `k` (shortest-path distance == 3,
relations ignored — deployable in principle) and **no** chain knowledge:

| 3-hop MRR | uniform | gold schedule |
|---|---:|---:|
| no mask | 0.264 | 0.260 |
| + graph-distance==3 mask | 0.313 | **0.927** |

Hits@1 under that mask: uniform 0.178 → gold schedule **0.926**, choosing
correctly out of a mean of 8373 candidates.

Crossing the depth axis against the relation model
(`rgdb-eval/scripts/experiment_depth_control.py`,
`rgdb-eval/results/metaqa-depth-control.md`) gives the full picture. 3-hop MRR:

| relation model | no depth control | + exactly-k mask |
|---|---:|---:|
| uniform (refraction off) | 0.264 | 0.313 |
| **trained matrix** (option A, ships today) | **0.235** | **0.419** |
| gold schedule (option C's ceiling) | 0.260 | 0.927 |

`trained` is seeded with the first-hop relation only — intent-classifiable, no chain
knowledge. The mask needs only `k`, no relation knowledge. So **`trained + mask` is
deployable today**; `gold` is not.

This is an **interaction**, not two independent effects:

- **Depth control alone buys little** (0.264 → 0.313). It narrows the field but
  cannot discriminate within it — the mask still leaves ~8373 candidates.
- **Relation modelling alone is actively HARMFUL at 3 hops** (0.264 → 0.235).
  Sharpening relation coherence concentrates mass along the correct chain, which
  makes the hop-1/hop-2 intermediates on that chain *stronger* competitors to the
  answer. Better relation modelling makes ranking worse until scoring is depth-aware.
- **Together they compound** (0.313 → 0.419 with a global matrix; → 0.927 with a
  perfect schedule).

The original gate held the depth axis at "uncontrolled", the one setting where the
relation axis has negative slope. It could not have passed, for any relation model.

> A first version of this diagnostic masked candidates to the **gold chain's
> terminal set** instead. That was wrong and nearly produced the opposite
> conclusion: the mask leaves only ~14 candidates that the gold chain already
> selected, so it hands the relation model to the "uniform" baseline for free
> (uniform scores 0.792 under it). Any future mask must not encode the answer's
> reasoning path.

### What this does NOT show

The 0.927 is an upper bound under **perfect chain prediction and perfect `k`**, and
it is dangerously close to a tautology: the chain-terminal mask reaches 100% of
gold answers with a mean of **14.3** candidates. If you truly know the chain and
`k`, you can execute it as a graph traversal and read off the answers — no
diffusion required. So this experiment establishes the ceiling of perfect schedule
knowledge; it says nothing about the two things C actually lives or dies on:

1. how accurately a schedule can be **predicted** (C2's RotatE + B's feedback), and
2. whether diffusion **degrades gracefully** when the predicted schedule is soft or
   wrong — the only regime where diffusion beats plain traversal.

Both are untested. A soft/incorrect schedule could easily land below PPR.

### Where the schedule already pays without any depth fix

2-hop, unmasked, today's kernel:

| 2-hop | untyped-ppr | gold schedule (hard) |
|---|---:|---:|
| MRR | 0.219 | **0.387** |
| Hits@5 | 0.491 | **0.881** |
| recall@20 | 0.798 | **0.948** |

At 2 hops one intermediate layer is thin enough that the schedule's precision wins
outright. (Hits@1 is ≈0 for *every* diffusion contender at 2 and 3 hops — the same
artifact, visible everywhere.)

### Verdict and replacement gate

C stays deferred. It was not built. The original gate is retired: it held the depth
axis at the one setting where the relation axis has negative slope, so no relation
model — gold included — could have passed it.

**The next piece of work is not C. It is depth-aware scoring**, and it is worth doing
on its own merits: it is what turns the already-shipped trained matrix from a 3-hop
*regression* (0.235) into a 3-hop *win* (0.419 vs untyped-PPR's 0.279, +50%).

Suggested shape — additive, no behavioural break by default. Let `propagate()`
accumulate intensity per depth and accept `depth_weights: &[f32]`:

- `[1,1,1,1]` reproduces today's behaviour exactly (the default).
- `[0,0,0,1]` is terminal-mass-only.
- a soft prior peaked at `k` degrades gracefully when `k` is uncertain.

Prefer this to the hard mask used in the experiment. The mask is brittle: only
**87.1%** of gold answers sit at shortest-distance exactly 3 (the rest are reachable
by shortcut edges), so it discards 13% of them outright and caps Hits@20 at ~0.93. A
soft depth prior keeps them. It also removes the need to know `k` exactly.

**If C is revisited afterwards, it must clear all three, in order:**

1. **Depth-aware scoring has landed and is measured separately.** Without it, C's
   own mechanism is self-defeating.
2. **A *predicted* schedule beats the trained matrix under the SAME depth control** —
   i.e. beat **0.419**, not untyped-PPR's 0.279. Gold schedules are settled: they
   reach 0.927. The only open question is prediction, and the honest baseline is the
   best deployable alternative, not the weakest one.
3. **Graceful degradation is measured.** Sweep schedule accuracy from 100% down to
   random and find where C crosses below 0.419. If the crossover sits at high
   accuracy, C is too brittle to ship regardless of its ceiling.

The 0.419 → 0.927 gap is the prize, and it is large. But note what the ceiling
implies: the chain-terminal mask reaches 100% of gold answers with a mean of **14.3**
candidates, so at perfect chain knowledge you could execute the chain as a traversal
and skip diffusion entirely. C only earns its keep in the middle of that range —
where the schedule is good but imperfect, and diffusion's soft accumulation beats a
brittle exact walk. Step 3 is therefore the real test of C, not step 2.

Until a predicted schedule exists (which needs B's feedback loop to generate training
data), there is nothing to test. **B (done) → depth-aware scoring → re-open C.**

---

## Why this exists: the measured 3-hop plateau

Path-internal refraction rewards relation-*coherent* paths. That is exactly right
for single-relation retrieval and exactly wrong for compositional multi-hop, which
*requires* changing relation type between hops. Training the relation-transition
matrix from data (option A) fixes most of this — but not all of it.

MetaQA, 1000 test questions per hop, query relation supplied from gold qtypes:

| hop | metric | name-embedded refraction | **trained matrix (A)** | untyped-ppr |
|-----|--------|-------------------------:|-----------------------:|------------:|
| 1   | MRR    | 0.999 | 0.992 | 0.956 |
| 2   | MRR    | 0.200 | **0.275** | 0.219 |
| 2   | Hits@5 | 0.441 | **0.682** | 0.491 |
| 3   | MRR    | 0.210 | 0.227 | **0.279** |
| 3   | Hits@20| 0.530 | **0.689** | 0.611 |

Read the 3-hop row carefully — it contains the whole diagnosis:

- The trained matrix has the **best deep recall** at 3 hops (Hits@20 0.689 vs
  PPR's 0.611). It *finds* the answer.
- The trained matrix has **worse MRR** at 3 hops (0.227 vs PPR's 0.279). It
  *ranks* the answer poorly.

It knows which transitions are *plausible in general*, but it cannot pin the exact
chain **this particular question** needs.

## The diagnosis

The learned matrix `T[r_prev][r_next]` is a **global marginal** — it averages over
every question type in the training set. But the correct next relation is a
property of the *question's reasoning chain*, not of the previous relation alone.

After `starred_actors`, one question wants `directed_by` and another wants
`has_genre`. A single global row must hedge between them. At 1 hop there is no
next relation to get wrong. At 2 hops one marginal is usually enough. At 3 hops the
ambiguity compounds and precision collapses — which is precisely the shape of the
data above.

**Corollary:** no amount of additional training data fixes this. It is a modeling
limit, not a sample-size limit. Conditioning is required.

## The proposal

Two coupled changes.

### C1. Per-hop expected-relation schedule (replaces "penalize any change")

Today the kernel scores a hop by `sim(r_in, r_out)` — the similarity between the
*previous* edge's relation and the *next* edge's relation. Instead, score it against
the relation the query **expects at that hop**:

```
current:  weight *= sim(r_prev_edge, r_next_edge)
proposed: weight *= sim(r_next_edge, expected[k])     # k = hop index
```

where `expected = [r1, r2, r3]` is the query's reasoning chain. This is the
straightforward generalization of the thing that already works: 1-hop hits MRR 0.999
*precisely because* we hand it `expected[0]` (the gold query relation) and it
rewards edges matching it. C1 simply extends that from hop 1 to every hop.

Note this makes refraction **query-relative** rather than path-internal — which is
the fork we consciously took the other side of during the original brainstorm. The
data now suggests path-internal was right for 1-hop and wrong for compositional QA.

### C2. Compositional relation embeddings (how the schedule gets predicted)

A schedule has to come from somewhere. `rgdb-embeddings` already ships a **RotatE +
GNN** training pipeline (`rgdb-embed train`). RotatE models each relation as a
rotation in complex space, so a path's *composed* relation is the product of its
rotations: `r1 ∘ r2 ∘ r3`. That gives a principled way to:

- score a candidate schedule against the query's target relation, and
- predict `expected[k]` given the query embedding and the partial path so far.

The global transition matrix from option A becomes the **prior** over schedules;
RotatE supplies the query-conditioned posterior.

## Why B (the self-learning loop) is a prerequisite

The A+B design's **backward credit pass** computes, for a correct answer, exactly
which relation transitions carried the mass that reached it — i.e. it *recovers the
reasoning chain from feedback*. That is precisely the supervision a schedule
predictor needs, and no deployment has it otherwise (MetaQA's gold `qtype` chains
are a research-dataset luxury).

So the dependency is real and load-bearing: **B generates C's training data.**
Building C first would mean hand-labeling reasoning chains.

## Validation experiment — run this BEFORE designing C

Do the cheap upper-bound test first. It can kill the idea in an afternoon.

1. Parse the **full gold relation sequence** from each MetaQA test qtype
   (`qtype_to_relation_sequence` already exists in `rgdb-eval/rgdb_eval/metaqa.py`
   and returns exactly this).
2. Feed that sequence to the kernel as a per-hop `expected[k]`.

   **No kernel change is required** (this doc originally claimed otherwise). The
   kernel's `r_in` *is* the relation traversed on the previous hop, so a per-hop
   schedule is exactly expressible in the existing `sim(r_in, r_out)` matrix, given
   one synthetic relation:

   - Add `__START__` as relation id `n` (the graph's edges never use it).
   - Seed the query with `query_relation = __START__`.
   - Build a per-question matrix: everything at `floor`, except
     `M[__START__][r1] = 1.0` and `M[r_i][r_{i+1}] = 1.0` for `i = 1..k-1`.

   Hop 1 then scores `sim(__START__, e)` — rewarding only `r1`. After traversing
   `r1`, `r_in = r1`, so hop 2 scores `sim(r1, e)` — rewarding only `r2`. And so on.
   Hops past the schedule find only `floor` and are suppressed. A per-question
   19x19 matrix is 361 floats; the experiment is pure Python over the existing
   bindings.
3. Re-run 1/2/3-hop and compare against the numbers in the table above.

**Pass/fail (RETIRED — this gate was run and is superseded):**
- ~~If 3-hop MRR with a *gold* schedule does not clearly beat untyped-PPR's 0.279,
  **abandon C.**~~ Run on 2026-07-10: gold schedule scored 0.254–0.260 vs 0.279, so
  this gate **failed**. C was not built.
- The gate's premise — "perfect chain knowledge is the best case" — was false as
  operationalized. It measured the schedule *and* the kernel's accumulate-at-every-node
  rule together, and the latter structurally penalizes exactly the deep paths the
  schedule routes mass along. See "Validation result" at the top of this document for
  the corrected experiment and the three-part replacement gate.

This mirrors how we tested 1-hop: hand the model the gold relation, establish the
ceiling, and only then ask whether it can be inferred. The 1-hop result (0.999) is
an upper bound under perfect intent classification, and C's result would be an
upper bound under perfect schedule prediction. Be honest about that in both cases.

## Seam left open by the A+B design

The A+B design keeps `propagate(graph, vocab, seeds, query_relation, params)`
unchanged and does **not** introduce a `RelationSimilarity` trait (YAGNI). But it was
shaped so C can slot in without a rewrite:

- The relation-similarity lookup is the only thing C needs to override. Introducing
  a provider trait (static matrix | learned matrix | per-hop schedule) is an
  additive change behind the existing call site.
- `TransitionStore` already produces a `RelationVocab`; a schedule provider would be
  a sibling implementation, not a replacement.
- The kernel already threads a per-state `r_in` through the frontier — a per-hop
  `expected[k]` needs the hop index `k`, which the depth loop already has.

Cost when it lands: dynamic dispatch (or a generic) in the propagation hot loop.
Measure it; the sparse walk touches ~789 nodes per query on MetaQA, so the overhead
is likely negligible relative to the win.

## Open questions

- **Schedule length vs. actual depth.** A predicted 3-step schedule applied to a
  path that terminates in 2 hops — pad, truncate, or score partial matches?
- **Soft schedules.** `expected[k]` as a *distribution* over relations rather than a
  single id would degrade more gracefully under an uncertain predictor, at the cost
  of a dot-product per hop instead of a table lookup.
- **Backwards compatibility.** Query-relative scoring changes results for every
  existing query. It is a behavioral break, not an additive feature.
- **Does the k-hop ranking bias survive?** — **ANSWERED: yes, and it decided the
  gate.** Diffusion ranks nearer nodes above the k-hop answer (Hits@1 ≈ 0 for every
  diffusion method at 2/3-hop). A perfect schedule did *not* fix Hits@1 on its own;
  it made 3-hop MRR slightly *worse* than PPR while nearly doubling recall@20. The
  orthogonal fix guessed at here (terminal-node bias / exactly-k-hop restriction) is
  a hard prerequisite, not an optional companion: with it, the gold schedule goes
  from 0.260 to 0.927 MRR. This question, filed as speculation, turned out to be the
  whole answer. See "Validation result".
