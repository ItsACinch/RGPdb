# MetaQA retrieval results

### hop2

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 1000 | 0.002 | 0.010 | 0.016 | 0.024 | 0.001 | 0.003 | 0.006 | 0.011 | 0.006 |
| vector+2hop | 1000 | 0.238 | 0.631 | 0.750 | 0.804 | 0.148 | 0.471 | 0.619 | 0.700 | 0.405 |
| untyped-ppr | 1000 | 0.010 | 0.491 | 0.773 | 0.934 | 0.003 | 0.385 | 0.621 | 0.798 | 0.219 |
| rgdb-new-uniform | 1000 | 0.003 | 0.454 | 0.756 | 0.943 | 0.000 | 0.342 | 0.602 | 0.801 | 0.211 |
| rgdb-new-refraction | 1000 | 0.000 | 0.390 | 0.700 | 0.907 | 0.000 | 0.327 | 0.596 | 0.791 | 0.185 |

## Verdict (1000-question sample of MetaQA 2-hop)

**Path-internal refraction does not pay its complexity here — it actively hurts.**
`rgdb-new-refraction` is worse than its own no-refraction ablation
`rgdb-new-uniform` on *every* metric (MRR 0.185 vs 0.211; Hits@5 0.390 vs
0.454). This matches the mechanism: MetaQA 2-hop questions require a relation
*change* (`starred_actors_inv → directed_by`), and path-internal refraction
penalizes relation changes, down-weighting the correct reasoning path.

The whole graph-diffusion family (untyped-ppr ≈ typed-uniform ≥ refraction)
loses to a simple **vector+2-hop** baseline on precision (Hits@1 0.238,
MRR 0.405) while winning on deep recall (Hits@20 ≈ 0.93 vs 0.80). Pure
vector-only is near-useless (Hits@1 0.002) — confirming multi-hop needs graph
structure, just not this refraction model.

**Caveat:** 2-hop questions are mapped to `query_relation=None`, the setting
least favorable to refraction (no query signal; every relation change is
penalized blindly). 1-hop questions (single query relation) would be the fair
test of refraction's intended strength but were not in the provided data.
