# Query-conditioned schedule via the production API (MetaQA 3-hop)

Predicted schedule fed through `core.propagate(..., schedule=...)` with `depth_weights=terminal(3)`.

| condition | MRR | Hits@1 |
|---|---|---|
| baseline terminal(3), no schedule | 0.3806 | - |
| predicted-schedule (reference predictor) | 0.8776 | 0.7905 |

CAVEAT: MetaQA's 15 fixed 3-hop templates make the reference predictor near-perfect; this is NOT evidence that open-ended question->schedule prediction is easy. The mechanism's robustness (degradation sweep in experiment_predicted_schedule.py) is the transferable evidence.
