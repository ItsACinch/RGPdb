# MetaQA retrieval results

### hop1

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 1000 | 0.003 | 0.006 | 0.007 | 0.008 | 0.003 | 0.004 | 0.005 | 0.006 | 0.004 |
| vector+2hop | 1000 | 0.062 | 0.440 | 0.735 | 0.909 | 0.049 | 0.358 | 0.622 | 0.780 | 0.235 |
| untyped-ppr | 1000 | 0.930 | 0.992 | 0.995 | 0.997 | 0.687 | 0.920 | 0.961 | 0.989 | 0.956 |
| rgdb-new-uniform | 1000 | 0.958 | 0.994 | 0.996 | 0.997 | 0.689 | 0.926 | 0.967 | 0.990 | 0.973 |
| rgdb-new-refraction | 1000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.710 | 0.936 | 0.974 | 0.994 | 0.999 |

### hop2

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 1000 | 0.002 | 0.010 | 0.016 | 0.024 | 0.001 | 0.003 | 0.006 | 0.011 | 0.006 |
| vector+2hop | 1000 | 0.238 | 0.631 | 0.750 | 0.804 | 0.148 | 0.471 | 0.619 | 0.700 | 0.405 |
| untyped-ppr | 1000 | 0.010 | 0.491 | 0.773 | 0.934 | 0.003 | 0.385 | 0.621 | 0.798 | 0.219 |
| rgdb-new-uniform | 1000 | 0.003 | 0.454 | 0.756 | 0.943 | 0.000 | 0.342 | 0.602 | 0.801 | 0.211 |
| rgdb-new-refraction | 1000 | 0.000 | 0.441 | 0.759 | 0.923 | 0.000 | 0.354 | 0.623 | 0.814 | 0.200 |

### hop3

| ranker | n | hits@1 | hits@5 | hits@10 | hits@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 1000 | 0.000 | 0.006 | 0.011 | 0.024 | 0.000 | 0.001 | 0.004 | 0.008 | 0.003 |
| vector+2hop | 1000 | 0.006 | 0.017 | 0.034 | 0.053 | 0.002 | 0.006 | 0.011 | 0.017 | 0.014 |
| untyped-ppr | 1000 | 0.172 | 0.404 | 0.497 | 0.611 | 0.055 | 0.113 | 0.166 | 0.262 | 0.279 |
| rgdb-new-uniform | 1000 | 0.159 | 0.362 | 0.442 | 0.563 | 0.048 | 0.107 | 0.138 | 0.215 | 0.255 |
| rgdb-new-refraction | 1000 | 0.129 | 0.307 | 0.417 | 0.530 | 0.042 | 0.094 | 0.130 | 0.205 | 0.210 |

## Verdict (1000 questions/hop, query relation from gold qtypes)

**No single method wins across hop counts — and refraction has a real, strong niche.**

| hop | winner (MRR) | refraction | typed-uniform | untyped-ppr | vector+2hop |
|-----|--------------|-----------:|--------------:|------------:|------------:|
| 1   | **refraction** | **0.999** | 0.973 | 0.956 | 0.235 |
| 2   | vector+2hop  | 0.200 | 0.211 | 0.219 | **0.405** |
| 3   | untyped-ppr  | 0.210 | 0.255 | **0.279** | 0.014 |

- **1-hop (single relation to follow): refraction WINS decisively** — Hits@1 0.999.
  Given the query relation, it isolates the correct-relation neighbors almost
  perfectly, beating plain PPR (0.930) and crushing vector search. This
  vindicates the refraction idea for *directional single-hop* retrieval.
- **2-hop / 3-hop (compositional): refraction mildly HURTS** and is the worst of
  the graph-diffusion methods. Compositional questions require *changing*
  relation type between hops, which path-internal refraction penalizes; the
  penalty compounds with depth.
- **vector+2hop** owns 2-hop (answers sit in the 2-hop ball) but **collapses at
  3-hop** (0.014 — it can't reach that far). Pure vector-only is useless at every
  depth: multi-hop genuinely needs graph structure.

**Mechanism:** path-internal refraction rewards relation-*coherent* paths — exactly
right for "give me the entities related to X by relation R," exactly wrong for
"compose across relation types." MetaQA mixes both, so the answer is
regime-dependent, not a single verdict.

**Caveat:** the query relation here is the GOLD qtype (perfect relation
inference). A real system must infer it from the question text (the intent
classifier), so the 1-hop 0.999 is an upper bound on refraction's benefit.
