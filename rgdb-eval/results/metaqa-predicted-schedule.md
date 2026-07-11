# Feature #3 measure-first gate: predicted schedule vs. baseline (MetaQA 3-hop)

Per the replacement gate in `docs/superpowers/specs/2026-07-10-deferred-query-conditioned-refraction.md`: depth-aware scoring (`depth_weights`) has already landed and is held constant here (`terminal(3)` on all three conditions). The open question is whether a schedule **predicted from question text** (no gold qtype) beats the deployed no-schedule baseline, and how gracefully it degrades as prediction accuracy drops.

**Honesty note:** MetaQA questions are machine-templated (only 15 distinct 3-hop qtypes), so text -> qtype prediction is an EASY, near-saturated problem here. The classifier accuracy below is an optimistic upper bound relative to real, open-ended questions, where paraphrase and compositional phrasing make schedule prediction substantially harder. The degradation sweep -- not the accuracy number -- is what actually speaks to deployability, since it measures what happens as prediction quality falls short of this optimistic ceiling.

## 1. Classifier

TF-IDF (word 1-2gram + char_wb 3-5gram) + LogisticRegression, entity span masked to a constant ` ENT ` token before featurizing. Trained on `qa_train_3hop.txt` (114196 questions, 15 qtype classes), evaluated on `qa_test_3hop.txt`.

**qtype accuracy: 1.0000** (n=14274)

## 2. Three conditions, full 3-hop test set

All three use `max_depth=4`, `min_intensity=1e-4`, `depth_weights=terminal(3)` = `[0,0,0,1,0]`. n = 14274.

| condition | MRR | Hits@1 | recall@20 |
|---|---:|---:|---:|
| baseline (trained matrix, gold 1st-hop relation) | 0.3806 | 0.2088 | 0.5310 |
| gold-schedule (`__START__` matrix, GOLD qtype) | 0.8777 | 0.7907 | 0.8597 |
| predicted-schedule (`__START__` matrix, PREDICTED qtype) | 0.8777 | 0.7907 | 0.8597 |

Reference: `results/metaqa-depth-weights.md` full-set baseline MRR = 0.381 (this run's baseline: 0.3806).

## 3. Graceful-degradation sweep

Starting from the GOLD schedule, each of the 3 relations is independently replaced with a uniformly random *different* relation with probability p (`numpy.random.default_rng(0)`). Scored on a fixed 2000-question subset (all questions with a resolved gold schedule, first 2000 in file order) for runtime; same subset used for every p and for the baseline reference line below.

| p (per-relation corruption) | 3-hop MRR |
|---:|---:|
| 0.00 | 0.8661 |
| 0.10 | 0.6671 |
| 0.25 | 0.4600 |
| 0.50 | 0.2349 |
| 1.00 | 0.1395 |
| — baseline (same subset) | 0.3781 |

Crossover: MRR first drops below the same-subset baseline (0.3781) at p = 0.5. The classifier's actual error rate on this test set is 0.0000 (14274/14274 correct); this is not directly the same quantity as per-relation corruption probability p (a wrong qtype prediction can differ from gold in one, two, or three of the three relations), but it gives the order of magnitude of real prediction error to compare against the crossover.

## Verdict

- classifier qtype accuracy: **1.0000**
- 3-hop MRR: baseline **0.3806**, predicted-schedule **0.8777**, gold-schedule **0.8777**
- degradation crossover: p = 0.5

**GATE: PASS**

Predicted-schedule 3-hop MRR clearly beats the deployed baseline, and the degradation sweep shows it stays above baseline down to a corruption rate at or below what the classifier's real error rate implies. #3 is worth designing for real (i.e. non-templated) deployments IF a comparably accurate predictor can be built there -- which this experiment, run on templated MetaQA text, cannot establish on its own (see honesty note above).
