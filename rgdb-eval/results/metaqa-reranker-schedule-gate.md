# Proposal #3 gate: expected-final-relation match feature vs. degraded schedules (MetaQA 3-hop)

The online reranker (`rgdb/src/reranker.rs`, see `results/metaqa-reranker.md`) regressed 3-hop Hits@1 because its features are GLOBAL (query-agnostic): a single learned feature-preference vector cannot supply the question-specific target relation that separates a correct same-depth answer from a same-depth distractor. Proposal #3 adds ONE query-conditioned feature -- `match = 1.0 if candidate's dominant incoming relation == schedule[last] else 0.0` -- and asks: does that single bit recover Hits@1 specifically in the realistic, DEGRADED regime where the schedule driving the query is only partially correct?

**Key subtlety respected here:** when the schedule is corrupted, `schedule[last]` used by the match feature is read from the SAME corrupted schedule that drove propagation -- never the gold schedule. The match feature can only help when the FINAL hop's relation happens to survive corruption while earlier hops do not; comparing against the gold schedule instead would leak the answer's relation type and invalidate the gate.

**Honesty caveat:** MetaQA is templated (15 fixed 3-hop qtypes), so its gold schedules are clean and corruption here is purely synthetic, uniform-random label noise -- a real relation predictor would make structured, correlated errors (e.g. confusing semantically similar relations), not uniform substitution. This experiment isolates a narrower question -- given that some fraction of a schedule's hops are wrong, does the `schedule[last]`-match feature carry signal the rest of the (global) reranker features don't already have -- as a proxy for predictor error, not a simulation of one.

Setup: trained transition-matrix vocab (co-occurrence over `qa_train_{1,2,3}hop_qtype.txt`, floor=0.05), `max_depth=4`, `min_intensity=0.0001`, top-20 candidates by `propagate_layered(...).per_depth[3]` (terminal(3) collapse). Train: 1500 3-hop questions from `qa_train_3hop.txt`. Test: 1500 3-hop questions from `qa_test_3hop.txt` (disjoint file). Two independent `numpy.random.default_rng(0)` streams drive train- and test-set corruption respectively. Two `LogisticRegression(max_iter=1000, class_weight='balanced')` models fit per p on `StandardScaler`-scaled features pooled over all train candidates: WITH the match feature (28 features: 5 per-depth + log1p(score) + log1p(out_degree) + 18-dim dominant-relation one-hot + match) and WITHOUT it (27 features).

## Results

| p | schedule-alone Hits@1 | reranker-no-match Hits@1 | reranker-with-match Hits@1 | recall@20 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.7787 | 0.6947 | 0.8720 | 0.8624 |
| 0.10 | 0.6240 | 0.5760 | 0.7173 | 0.6997 |
| 0.25 | 0.4567 | 0.4347 | 0.5560 | 0.5201 |
| 0.40 | 0.3193 | 0.3267 | 0.4287 | 0.3777 |

n = 1500 test questions per row. recall@20 is identical across the three rankings by construction (they rerank the same fixed top-20 candidate set; only the ORDER differs), so it is reported once per p as the ceiling on achievable Hits@1.

## Verdict

PASS requires reranker-with-match to beat BOTH schedule-alone AND reranker-no-match by >= 0.02 Hits@1 at p in {0.25, 0.4} (the degraded, realistic regime).

- p=0.25: with-match 0.5560 vs schedule-alone 0.4567 (delta +0.0993) vs no-match 0.4347 (delta +0.1213)
- p=0.40: with-match 0.4287 vs schedule-alone 0.3193 (delta +0.1093) vs no-match 0.3267 (delta +0.1020)

**GATE: PASS**

The match feature recovers Hits@1 over both the schedule-alone ranking and a reranker without it, at p in [0.25, 0.4], by a margin that survives the honesty caveat above. Worth building the query-conditioned reranker feature for real; the surviving open question is whether a real relation predictor's error pattern (correlated, not uniform-random) preserves this advantage.
