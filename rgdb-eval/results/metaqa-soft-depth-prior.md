# Follow-up #2 measure-first gate: soft depth-weight prior shapes vs. hard `terminal(k)` (MetaQA 3-hop)

Question: does a SOFT (spread-out) depth-weight shape, peaked at a hop-count guess `k_used`, degrade more gracefully than the hard `terminal(k)` readout when the hop-count predictor is wrong by one? `terminal(k)` reads out only the mass arriving at exactly depth `k`; if `k` is wrong, it reads out (near-)zero at the depth where the true answer's mass actually concentrates. A soft shape spreads some weight to neighboring depths, so it should retain more signal when `k` is off, at the cost of (possibly) diluting the signal when `k` is exactly right.

This is a different question from `experiment_soft_depth_weights.py` / `results/metaqa-soft-depth-weights.md`, which tested a *learned* per-depth weight vector (via the Engine's feedback loop) and found it a dead end -- the 3-hop answer's mass concentrates at the same depth as distractors, which is a per-node problem no depth-only reweighting can fix. Here the weight vector's SHAPE is fixed and hand-specified; only its peak position `k_used` varies, simulating a hop-count predictor that is off by one.

Setup: MetaQA KB, trained transition-matrix vocab (relation-pair co-occurrence from `qa_train_{1,2,3}hop_qtype.txt`), `core.build_graph`, `max_depth=4`, `min_intensity=1e-4`. 3-hop TEST questions (n=2000), seeded with the gold first-hop relation via `load_questions(..., qtype_path=...)`. True hop count k=3. Rankings strip the seed node, top-20.

## Weight shapes

Length `MAX_DEPTH+1=5`, index `d` = arrival depth, peaked at `k`:

- `terminal(k)`: 1.0 at `d=k`, else 0.0
- `geometric(k, r)`: `w[d] = r**|d-k|`, `r` in {0.3, 0.5}
- `triangular(k, W)`: `w[d] = max(0, 1 - |d-k|/W)`, `W=2`

## MRR by weight shape x k_used

True k = 3. k_used=2 and k_used=4 simulate a hop-count predictor off by one.

| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |
|---|---:|---:|---:|
| terminal | 0.0258 | 0.3781 | 0.0262 |
| geometric r=0.3 | 0.1876 | 0.2690 | 0.2181 |
| geometric r=0.5 | 0.2064 | 0.2432 | 0.2294 |
| triangular W=2 | 0.2079 | 0.2095 | 0.3228 |

## recall@20 by weight shape x k_used

| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |
|---|---:|---:|---:|
| terminal | 0.1393 | 0.5395 | 0.1164 |
| geometric r=0.3 | 0.3097 | 0.4143 | 0.3643 |
| geometric r=0.5 | 0.3274 | 0.3815 | 0.3616 |
| triangular W=2 | 0.3313 | 0.4057 | 0.4468 |

## Hits@1 by weight shape x k_used

| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |
|---|---:|---:|---:|
| terminal | 0.0005 | 0.2075 | 0.0010 |
| geometric r=0.3 | 0.1310 | 0.1430 | 0.1145 |
| geometric r=0.5 | 0.1385 | 0.1395 | 0.1370 |
| triangular W=2 | 0.1385 | 0.0895 | 0.1700 |

## Verdict

- At exact k=3, shapes matching `terminal(3)` (MRR 0.3781) within 0.01 MRR: none.
- No soft shape matched terminal(3) at exact k within tolerance.

**GATE: FAIL/marginal**

No soft shape both matched terminal(3) at exact k and clearly beat terminal at wrong k under the measured conditions. See the numbers above for exactly where it falls short.

## Hop-count predictor accuracy (question text -> 1/2/3 hops)

Entity-masked TF-IDF (word 1-2gram + char_wb 3-5gram) + LogisticRegression, trained on ALL train questions across 1/2/3-hop, tested on the corresponding test sets. This tells us how often `k_used` would be exactly right vs. off by one in the deployable path (predicted schedule, no gold qtype).

**Overall accuracy: 1.0000**

| true hop count | accuracy |
|---:|---:|
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |

**Honesty note:** MetaQA questions are machine-templated from a small, fixed set of qtypes, and hop count strongly correlates with surface-level cues (answer type, question length, template phrasing). Text -> hop-count prediction is consequently a much easier problem here than in real, open-ended questions, where paraphrase and compositional structure make it harder to tell from phrasing alone how many hops are needed. Treat this accuracy as an optimistic upper bound: it tells us that on templated MetaQA the k-error case is rare, not that it will be rare in a real deployment. The k-error ROBUSTNESS results above (the weight-shape comparison at k_used=2/3/4) are the transferable evidence -- they hold regardless of how often k is actually wrong; this accuracy number only tells us how much that robustness would matter on THIS dataset.

