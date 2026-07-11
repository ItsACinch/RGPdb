# Soft depth weights (#2): a measured dead end (MetaQA 3-hop)

Learned per-depth weight vectors cannot beat hard `terminal(k)`, because the 3-hop answer's arrival mass concentrates at depth 1 (~76%) alongside the distractors. Separating same-depth answer from distractor needs a per-node model (feature #4, the reranker).

| 3-hop derivation | MRR | recall@20 |
|---|---|---|
| terminal(3) | 0.3806 | 0.5310 |
| discriminative | 0.2824 | 0.4889 |
| generative | 0.2817 | 0.4886 |
