# Distance artifact vs relation model (MetaQA 3-hop)


The two masks differ in how much gold knowledge they use:


- **graph-distance==3**: nodes whose shortest-path distance from the topic is
  exactly 3, relations ignored. Needs only `k` -- a deployable fix could use it.
  Mean candidates 8373.3; covers 0.871 of gold answers.

- **gold-chain-terminal**: nodes reached by walking the exact gold relation chain.
  Uses the answer's reasoning path; a strict upper bound, not deployable.
  Mean candidates 14.3; covers 1.000 of gold answers.


Recall@20 is bounded by each mask's gold coverage.


| variant | hits@1 | hits@20 | recall@20 | mrr |
|---|---|---|---|---|
| uniform, no mask | 0.159 | 0.563 | 0.220 | 0.264 |
| gold-schedule, no mask | 0.151 | 0.793 | 0.577 | 0.260 |
| uniform + graph-distance==3 mask | 0.178 | 0.765 | 0.351 | 0.313 |
| gold-schedule + graph-distance==3 mask | 0.926 | 0.932 | 0.744 | 0.927 |
| uniform + gold-chain-terminal mask | 0.626 | 1.000 | 0.902 | 0.792 |
| gold-schedule + gold-chain-terminal mask | 0.809 | 1.000 | 0.903 | 0.895 |
