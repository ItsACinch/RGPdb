# Online-learning acceptance (MetaQA)

### hop1

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| untyped-ppr | 1000 | 0.930 | 0.992 | 0.995 | 0.997 | 0.687 | 0.920 | 0.961 | 0.989 | 0.956 |
| rgdb-new-uniform | 1000 | 0.958 | 0.994 | 0.996 | 0.997 | 0.689 | 0.926 | 0.967 | 0.990 | 0.973 |
| rgdb-new-online-learned | 1000 | 0.988 | 0.999 | 0.999 | 0.999 | 0.709 | 0.935 | 0.973 | 0.993 | 0.993 |

### hop2

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| untyped-ppr | 1000 | 0.010 | 0.491 | 0.773 | 0.934 | 0.003 | 0.385 | 0.621 | 0.798 | 0.219 |
| rgdb-new-uniform | 1000 | 0.003 | 0.454 | 0.756 | 0.943 | 0.000 | 0.342 | 0.602 | 0.801 | 0.211 |
| rgdb-new-online-learned | 1000 | 0.002 | 0.588 | 0.843 | 0.958 | 0.000 | 0.439 | 0.664 | 0.836 | 0.244 |

### hop3

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| untyped-ppr | 1000 | 0.172 | 0.404 | 0.497 | 0.611 | 0.055 | 0.113 | 0.166 | 0.262 | 0.279 |
| rgdb-new-uniform | 1000 | 0.159 | 0.362 | 0.442 | 0.563 | 0.048 | 0.107 | 0.138 | 0.215 | 0.255 |
| rgdb-new-online-learned | 1000 | 0.135 | 0.244 | 0.397 | 0.626 | 0.043 | 0.073 | 0.133 | 0.264 | 0.204 |

## Verdict: the feedback loop learns — and reveals its own limit

Replayed **4,000 feedback events** (query → `record_feedback(gold answer)`) through
`RgdbEngine`, starting from an exactly-uniform cold start. No gold reasoning chains
were used: credit was attributed purely by the backward pass over seed→answer paths.

| hop | uniform (cold start) | **online-learned** | offline-trained | untyped-ppr |
|-----|---------------------:|-------------------:|----------------:|------------:|
| 1 (MRR)  | 0.973 | **0.993** | 0.992 | 0.956 |
| 2 (MRR)  | 0.211 | **0.244** | 0.275 | 0.219 |
| 3 (MRR)  | 0.255 | 0.204 | 0.227 | **0.279** |
| 2 (Hits@5) | 0.454 | **0.588** | 0.682 | 0.491 |

**What worked.** Both acceptance assertions passed. From zero prior knowledge the loop
recovered essentially all of the 1-hop win (0.993, matching offline's 0.992 and beating
the uniform start's 0.973) and lifted 2-hop MRR from 0.211 to 0.244 — over half the way
to the offline-trained 0.275, from only 4,000 events versus offline's 347,372 gold
transitions. Hits@5 at 2-hop moved 0.454 → 0.588. Learning from *observed feedback
alone*, with no labeled reasoning chains, demonstrably works.

**What regressed — honestly.** 3-hop MRR fell from 0.255 (uniform) to 0.204, below both
the offline matrix (0.227) and plain PPR (0.279). Deep recall still improved
(Hits@20 0.626 vs PPR's 0.611; recall@20 0.264 vs 0.262), so it finds 3-hop answers and
ranks them worse.

**Why.** The learned matrix is *sharp*: off-diagonal mean 0.535, minimum pinned at the
`floor = 0.05`. A sharp matrix penalizes relation changes hard, which is exactly right
for 1-hop (stay on the query relation) and helpful at 2-hop, but actively harmful at
3-hop, where the correct chain *requires* two relation changes. Three effects compound
it: (1) credit rewards every seed→answer path, not only the gold chain, so the learned
distribution is noisier than offline's; (2) 4,000 events is thin evidence, so `κ=10`
pseudo-counts are overwhelmed unevenly across rows; (3) the documented pruning bias
(`min_intensity = 1e-4` during replay) under-credits relation-turning transitions.

**This is the same wall the offline experiment hit**, and it points at the same place:
a *global* transition matrix is a marginal and cannot encode a per-question reasoning
chain. See `2026-07-10-deferred-query-conditioned-refraction.md`. More feedback, a
softer `floor`, or a larger `κ` would likely recover some 3-hop ground; none of them
fix the modeling limit.
