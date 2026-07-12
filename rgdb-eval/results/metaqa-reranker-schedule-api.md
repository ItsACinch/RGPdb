# Query-conditioned reranker via the production engine (MetaQA 3-hop)

Degraded schedules (each relation corrupted with prob p). Reranker trained online through record_feedback under the corrupted schedule's expected final relation; both variants evaluated on the same corrupted test schedules.

| p | schedule-alone Hits@1 | reranker+match Hits@1 |
|---|---|---|
| 0.0 | 0.7695 | 0.8025 |
| 0.25 | 0.4455 | 0.5140 |
| 0.4 | 0.2905 | 0.3920 |

CAVEAT: corruption is synthetic uniform relation substitution, a proxy for a real predictor's correlated errors; MetaQA is templated. Follow-up #1 (non-templated eval) is where a real predictor's error pattern is measured. The lean feature-set pruning (drop net-negative degree/incoming-one-hot) remains a follow-up.
