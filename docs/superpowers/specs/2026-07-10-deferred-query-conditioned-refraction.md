# Deferred: Query-Conditioned Per-Hop Refraction ("Option C")

> **STATUS: DEFERRED — an unvalidated hypothesis, not an approved design.**
> Explicitly out of scope for the data-weighted-transitions + self-learning-loop
> design (options A + B). Do not build this until the validation experiment below
> has been run and passes. Captured here so the reasoning isn't lost.

**Date captured:** 2026-07-10
**Depends on:** the A+B design (self-learning loop) — see below, B is what makes C
*trainable*.

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
2. Feed that sequence to the kernel as a per-hop `expected[k]`, scoring each hop as
   `sim(edge_relation, expected[k])` instead of `sim(prev_edge, edge)`.
   This requires a kernel variant that accepts a schedule.
3. Re-run 1/2/3-hop and compare against the numbers in the table above.

**Pass/fail:**
- If 3-hop MRR with a *gold* schedule does not clearly beat untyped-PPR's 0.279,
  **abandon C.** Perfect knowledge of the chain would be the best case, and if the
  best case doesn't win, predicting the chain imperfectly certainly won't.
- If it does beat it, the question becomes "can we predict the schedule?" — and
  only then is C2 (RotatE + the B feedback loop) worth designing.

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
- **Does the k-hop ranking bias survive?** Diffusion inherently ranks nearer nodes
  above the k-hop answer (Hits@1 ≈ 0 for every diffusion method at 2/3-hop). A
  perfect schedule may still not fix Hits@1, because that is a *distance* artifact,
  not a *relation* artifact. C may lift MRR without touching Hits@1 — worth
  measuring separately, and possibly needing an orthogonal fix (terminal-node bias,
  or restricting candidates to exactly-k-hop nodes).
