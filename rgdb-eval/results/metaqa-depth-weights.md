# Depth-weighted terminal(k) acceptance (MetaQA, full test set)


Trained transition matrix, seeded with the first-hop relation. `terminal(k)`
is the depth-weighted readout; `uniform` is the SAME matrix with no depth
weighting (depth_weights=None) -- the isolated lift. Full test set per hop, so
these are stable estimates, not slice-dependent.


Baseline reference (from results/metaqa-depth-control.md): untyped-PPR 3-hop
MRR 0.279; the trained matrix without depth control 0.235. terminal(3)'s 0.381
beats both by +37% / +62%.


| hop | n | terminal(k) Hits@1 | terminal(k) MRR | uniform MRR | floor |
|---|---|---|---|---|---|
| 1 | 9947 | 0.9761 | 0.9869 | 0.9854 | 0.98 |
| 2 | 14872 | 0.4416 | 0.6117 | 0.2665 | 0.58 |
| 3 | 14274 | 0.2088 | 0.3806 | 0.2278 | 0.36 |
