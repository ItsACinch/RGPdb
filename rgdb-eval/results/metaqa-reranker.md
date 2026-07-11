# Reranker gate (MetaQA 3-hop)

`EngineConfig.reranker_enabled` defaults to **false** (rgdb/src/engine.rs): the
reranker is neither applied in `query()` nor trained in `record_feedback()` unless
explicitly turned on. This mirrors `depth_profile_learning`.

| variant | Hits@1 | recall@20 | source |
|---|---|---|---|
| cold / terminal(3), reranker off (this script, default `core.Engine`) | 0.2093 | 0.5249 | measured here |
| + online reranker enabled (Rust-only; not reachable from `core.Engine`) | 0.176 | (+10% recall@20) | design spec Component #4 MEASURED OUTCOME |

The reranker-enabled row is **not reproduced by this script** -- the Python
`core.Engine` binding (rgdb-python/src/lib.rs) does not expose a `reranker_enabled`
kwarg, so every Python-constructed engine gets the gated-off (default) `EngineConfig`.
Enabling the reranker requires building an `RgdbEngine` from Rust with
`EngineConfig { reranker_enabled: true, .. }`. The regression when it IS enabled
(3-hop Hits@1 0.208 -> 0.176, while recall@20 improves ~10%) is documented in
`docs/superpowers/specs/2026-07-11-feedback-learned-ranking-design.md` (Component #4
MEASURED OUTCOME) and in the raw acceptance-gate run captured in
`.superpowers/sdd/task-8-report.md`. Root cause: the reranker learns a single global
feature-preference vector shared across all queries, so it cannot supply the
question-specific target relation that separates a correct same-depth answer from a
distractor -- the same wall the learned soft depth weights (#2) hit. Disposition: keep
the reranker as an inert seam for a future query-conditioned extension (#3); do not
enable it by default.
