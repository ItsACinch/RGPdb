# What fixes 3-hop? depth control x relation model (MetaQA, 1000 q)


`trained` uses only the first-hop relation (intent-classifiable).
`gold` uses the full reasoning chain (not deployable).
`exactly-k mask` uses only k (shortest-path distance), no relation info.


| variant | hits@1 | hits@20 | recall@20 | mrr |
|---|---|---|---|---|
| uniform, no depth control | 0.159 | 0.563 | 0.220 | 0.264 |
| uniform + exactly-k mask | 0.178 | 0.765 | 0.351 | 0.313 |
| trained matrix, no depth control | 0.136 | 0.689 | 0.329 | 0.235 |
| trained matrix + exactly-k mask | 0.246 | 0.844 | 0.472 | 0.419 |
| gold schedule, no depth control | 0.151 | 0.793 | 0.577 | 0.260 |
| gold schedule + exactly-k mask | 0.926 | 0.932 | 0.744 | 0.927 |
