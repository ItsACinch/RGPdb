//! `RgdbEngine`: graph + live vocab + learned transitions + feedback loop.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use lru::LruCache;
use thiserror::Error;

use crate::credit::credit;
use crate::depth_profile::{DepthProfileConfig, DepthProfileStore};
use crate::depth_weights::DepthWeights;
use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate_layered, LayeredResult, PropagationParams};
use crate::relation::RelationVocab;
use crate::transitions::{TransitionConfig, TransitionError, TransitionStore};

pub type QueryId = u64;

#[derive(Debug, Error)]
pub enum FeedbackError {
    #[error("unknown or expired query id {0}")]
    UnknownQuery(QueryId),
    #[error("target node {0} is out of bounds")]
    InvalidTarget(NodeId),
    #[error("target node {0} is unreachable from the query's seeds within max_depth")]
    TargetUnreachable(NodeId),
    #[error(transparent)]
    Transition(#[from] TransitionError),
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub ranked: Vec<(NodeId, f32)>,
    pub query_id: QueryId,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub cache_capacity: usize,
    pub cache_ttl: Duration,
    /// Applied to queries whose caller omits `depth_weights`, but only when its
    /// length matches the query's `max_depth + 1`; otherwise the query runs uniform.
    /// Defaults to `uniform` (all-ones), so out-of-the-box behavior is unchanged.
    pub default_depth_weights: DepthWeights,
    /// When false (default), feedback does NOT update the depth profile, so `hop_hint`
    /// resolves to a stable `terminal(k)`. The learned derivation is a measured dead end
    /// at 3 hops (see docs/superpowers/specs/2026-07-11-feedback-learned-ranking-design.md);
    /// this seam stays off until a per-node-aware derivation exists.
    pub depth_profile_learning: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            cache_capacity: 4096,
            cache_ttl: Duration::from_secs(3600),
            default_depth_weights: DepthWeights::uniform(4),
            depth_profile_learning: false,
        }
    }
}

/// Everything `credit()` needs to attribute a later feedback event, including the
/// vocab that actually produced the ranking (a `refresh()` may swap the live one).
#[derive(Clone)]
struct QueryContext {
    seeds: Vec<(NodeId, f32)>,
    query_relation: Option<RelationId>,
    params: PropagationParams,
    vocab: Arc<RelationVocab>,
    hop_hint: Option<usize>,
    layered: LayeredResult,
    created: Instant,
}

pub struct RgdbEngine {
    graph: Graph,
    vocab: ArcSwap<RelationVocab>,
    store: Mutex<TransitionStore>,
    cache: Mutex<LruCache<QueryId, QueryContext>>,
    next_id: AtomicU64,
    engine_cfg: EngineConfig,
    depth_profile: Mutex<DepthProfileStore>,
}

impl RgdbEngine {
    /// `vocab_prior` supplies both the relation names and the prior matrix.
    /// Use `RelationVocab::with_names_uniform(names)` for the do-no-harm default.
    pub fn new(graph: Graph, vocab_prior: RelationVocab, cfg: TransitionConfig) -> Self {
        Self::with_engine_config(graph, vocab_prior, cfg, EngineConfig::default())
    }

    pub fn with_engine_config(
        graph: Graph,
        vocab_prior: RelationVocab,
        cfg: TransitionConfig,
        engine_cfg: EngineConfig,
    ) -> Self {
        let store = TransitionStore::new(
            vocab_prior.names().to_vec(),
            Some(vocab_prior.matrix().to_vec()),
            cfg,
        )
        .expect("prior came from a RelationVocab, so its shape is n*n");
        let initial = store.derive_vocab();
        Self::assemble(graph, store, initial, engine_cfg)
    }

    fn assemble(graph: Graph, store: TransitionStore, vocab: RelationVocab, engine_cfg: EngineConfig) -> Self {
        let cap = NonZeroUsize::new(engine_cfg.cache_capacity.max(1)).unwrap();
        Self {
            graph,
            vocab: ArcSwap::from_pointee(vocab),
            store: Mutex::new(store),
            cache: Mutex::new(LruCache::new(cap)),
            next_id: AtomicU64::new(1),
            engine_cfg,
            depth_profile: Mutex::new(DepthProfileStore::new(4, DepthProfileConfig::default())),
        }
    }

    /// Restore learned state from a sidecar, validating relation coverage.
    pub fn load(graph: Graph, path: &str) -> Result<Self, TransitionError> {
        let store = TransitionStore::load(path)?;
        // Only enforce coverage when the graph actually uses relations. An edgeless
        // graph vacuously satisfies any vocabulary, including an empty one.
        if let Some(graph_max) = graph.edge_props().iter().map(|e| e.relation as usize).max() {
            if store.len() <= graph_max {
                return Err(TransitionError::RelationCoverage {
                    store_len: store.len(),
                    graph_max,
                });
            }
        }
        let vocab = store.derive_vocab();
        Ok(Self::assemble(graph, store, vocab, EngineConfig::default()))
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    pub fn vocab(&self) -> Arc<RelationVocab> { self.vocab.load_full() }
    pub fn matrix(&self) -> Vec<f32> { self.vocab().matrix().to_vec() }
    pub fn counts_snapshot(&self) -> Vec<f32> { self.store.lock().unwrap().counts().to_vec() }
    pub fn events_since_rebuild(&self) -> u32 { self.store.lock().unwrap().events_since_rebuild() }

    pub fn query(
        &self,
        seeds: &[(NodeId, f32)],
        query_relation: Option<RelationId>,
        hop_hint: Option<usize>,
        params: &PropagationParams,
    ) -> QueryResult {
        // Resolve depth weights: caller override wins; else learned soft weights for the
        // hop hint (when its length fits); else the config default; else uniform.
        let resolved = if params.depth_weights.is_some() {
            params.clone()
        } else if let Some(k) = hop_hint {
            let w = self.depth_profile.lock().unwrap().weights_for(k);
            if w.as_slice().len() == params.max_depth + 1 {
                PropagationParams { depth_weights: Some(w), ..params.clone() }
            } else {
                params.clone()
            }
        } else if self.engine_cfg.default_depth_weights.as_slice().len() == params.max_depth + 1 {
            PropagationParams {
                depth_weights: Some(self.engine_cfg.default_depth_weights.clone()),
                ..params.clone()
            }
        } else {
            params.clone()
        };

        let vocab = self.vocab.load_full();
        let layered = propagate_layered(&self.graph, &vocab, seeds, query_relation, &resolved);
        let c = resolved.depth_weights.clone();
        // Collapse layered -> scalar score with the resolved weights (uniform if None).
        let mut ranked: Vec<(NodeId, f32)> = layered
            .per_depth
            .iter()
            .map(|(&node, prof)| {
                let s = match &c {
                    Some(w) => prof.iter().zip(w.as_slice()).map(|(x, ww)| x * ww).sum(),
                    None => prof.iter().sum(),
                };
                (node, s)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache.lock().unwrap().put(
            query_id,
            QueryContext {
                seeds: seeds.to_vec(),
                query_relation,
                params: resolved,
                vocab,
                hop_hint,
                layered,
                created: Instant::now(),
            },
        );
        QueryResult { ranked, query_id }
    }

    /// Attribute `target` back to the relation transitions that carried mass to it.
    /// `signal` is signed: negative reports a wrong answer.
    ///
    /// Not idempotent: the query context is retained on success so a caller may
    /// report several good answers for one ranking. Calling this twice with the same
    /// `(query_id, target)` credits that target twice. Retrying callers must
    /// de-duplicate.
    pub fn record_feedback(
        &self,
        query_id: QueryId,
        target: NodeId,
        signal: f32,
    ) -> Result<(), FeedbackError> {
        let ctx = {
            let mut cache = self.cache.lock().unwrap();
            match cache.get(&query_id) {
                Some(c) if c.created.elapsed() <= self.engine_cfg.cache_ttl => c.clone(),
                Some(_) => {
                    cache.pop(&query_id);
                    return Err(FeedbackError::UnknownQuery(query_id));
                }
                None => return Err(FeedbackError::UnknownQuery(query_id)),
            }
        };

        if (target as usize) >= self.graph.num_nodes() {
            return Err(FeedbackError::InvalidTarget(target));
        }

        // Reachability is decided by the layered result (did the target receive any
        // mass?), NOT by transition-credit emptiness: under a narrow depth_weights the
        // transition credit can be empty for a target that is still reachable at another
        // depth, and the depth-profile learner must still see that signal.
        let reachable = ctx
            .layered
            .per_depth
            .get(&target)
            .map(|p| p.iter().any(|&x| x > 0.0))
            .unwrap_or(false);
        if !reachable {
            return Err(FeedbackError::TargetUnreachable(target));
        }

        // Depth-profile learner: gated off by default (measured dead end at 3 hops).
        if self.engine_cfg.depth_profile_learning {
            if let Some(k) = ctx.hop_hint {
                let width = ctx.params.max_depth + 1;
                let answer = ctx.layered.per_depth.get(&target).cloned()
                    .unwrap_or_else(|| vec![0.0; width]);
                let mut background = vec![0.0f32; width];
                for prof in ctx.layered.per_depth.values() {
                    for (d, &x) in prof.iter().enumerate() {
                        if d < width {
                            background[d] += x;
                        }
                    }
                }
                let mut store = self.depth_profile.lock().unwrap();
                store.record(k, &answer, &background, signal);
                let n = store.config().rebuild_every_n;
                if n > 0 && store.events_since_rebuild() >= n {
                    store.rebuild_marker(); // weights_for derives lazily on read; just reset the counter
                }
            }
        }

        // Transition learner: credit under the query-time vocab + params. May be empty
        // under a narrow depth_weights — recording only when there is credit preserves
        // the pre-feature transition-event behavior exactly.
        let credits = credit(
            &self.graph,
            &ctx.vocab,
            &ctx.seeds,
            ctx.query_relation,
            target,
            &ctx.params,
        );
        if !credits.is_empty() {
            let new_vocab = {
                let mut store = self.store.lock().unwrap();
                store.record(&credits, signal);
                let n = store.config().rebuild_every_n;
                if n > 0 && store.events_since_rebuild() >= n {
                    Some(store.rebuild())
                } else {
                    None
                }
            };
            if let Some(vocab) = new_vocab {
                self.vocab.store(Arc::new(vocab));
            }
        }
        Ok(())
    }

    /// Reset the depth-profile event counter (weights are derived lazily on read).
    pub fn refresh_profiles(&self) {
        self.depth_profile.lock().unwrap().rebuild_marker();
    }

    /// Rebuild the similarity matrix from counts and atomically swap it in.
    pub fn refresh(&self) {
        let vocab = { self.store.lock().unwrap().rebuild() };
        self.vocab.store(Arc::new(vocab));
    }

    pub fn save(&self, path: &str) -> Result<(), TransitionError> {
        self.store.lock().unwrap().save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::PropagationParams;
    use crate::relation::RelationVocab;
    use crate::transitions::TransitionConfig;

    /// 0 -(A=0)-> 1 -(B=1)-> 2
    fn engine(rebuild_every_n: u32) -> RgdbEngine {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let cfg = TransitionConfig { rebuild_every_n, ..TransitionConfig::default() };
        RgdbEngine::new(g, prior, cfg)
    }

    fn params() -> PropagationParams { PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None } }

    #[test]
    fn cold_start_matrix_is_uniform() {
        let e = engine(0);
        for a in 0..2u16 {
            for b in 0..2u16 {
                assert_eq!(e.vocab().similarity(a, b), 1.0);
            }
        }
    }

    #[test]
    fn query_returns_ranked_results_and_an_id() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), None, &params());
        assert!(r.query_id > 0);
        assert!(!r.ranked.is_empty());
        // sorted descending by score
        for w in r.ranked.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn feedback_on_unknown_query_errors() {
        let e = engine(0);
        assert!(matches!(e.record_feedback(9999, 2, 1.0), Err(FeedbackError::UnknownQuery(_))));
    }

    #[test]
    fn feedback_on_out_of_bounds_target_errors() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), None, &params());
        assert!(matches!(e.record_feedback(r.query_id, 99, 1.0), Err(FeedbackError::InvalidTarget(_))));
    }

    #[test]
    fn feedback_on_unreachable_target_errors() {
        // seed at node 2 (a sink): nothing is reachable
        let e = engine(0);
        let r = e.query(&[(2, 1.0)], Some(0), None, &params());
        assert!(matches!(e.record_feedback(r.query_id, 0, 1.0), Err(FeedbackError::TargetUnreachable(_))));
    }

    #[test]
    fn feedback_then_refresh_changes_the_matrix() {
        // Three relations, so row A has TWO off-diagonals. With only two relations the
        // single off-diagonal always normalizes to 1.0 and this test could not
        // discriminate. Graph: 0 -(A=0)-> 1 -(B=1)-> 2, and 0 -(C=2)-> 3.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let ec = (3u32, EdgeProps { attenuation: 0.0, relation: 2, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(
            4,
            vec![vec![ea, ec], vec![eb], vec![], vec![]],
            NodeProps::default(),
        )
        .unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into(), "C".into()]);
        let e = RgdbEngine::new(
            g,
            prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
        );

        // Cold start: do no harm, everything is exactly 1.0.
        assert_eq!(e.vocab().similarity(0, 1), 1.0);
        assert_eq!(e.vocab().similarity(0, 2), 1.0);

        let p = PropagationParams { max_depth: 2, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        e.record_feedback(r.query_id, 2, 100.0).unwrap();

        // Recorded, but not rebuilt yet: the live matrix must not have moved.
        assert_eq!(e.vocab().similarity(0, 2), 1.0, "no refresh yet");

        e.refresh();

        // Credit for target 2 splits 0.5 on (A,A) and 0.5 on (A,B); signal 100 puts
        // 50 into counts[A][B] and 50 into counts[A][A] (diagonal, ignored by derive).
        // Row A off-diagonals: blended(A,B) = 50 + 10 = 60, blended(A,C) = 0 + 10 = 10,
        // off-diagonal rowmax = 60.
        assert_eq!(e.vocab().similarity(0, 0), 1.0, "diagonal stays pinned");
        assert!(
            (e.vocab().similarity(0, 1) - 1.0).abs() < 1e-5,
            "credited A->B becomes the row max, got {}",
            e.vocab().similarity(0, 1)
        );
        assert!(
            (e.vocab().similarity(0, 2) - (10.0 / 60.0)).abs() < 1e-4,
            "uncredited A->C must drop below 1.0, got {}",
            e.vocab().similarity(0, 2)
        );
    }

    #[test]
    fn auto_refresh_fires_at_rebuild_every_n() {
        let e = engine(1); // refresh after every event
        let r = e.query(&[(0, 1.0)], Some(0), None, &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        assert_eq!(e.events_since_rebuild(), 0, "auto-refresh reset the counter");
    }

    #[test]
    fn save_load_roundtrip_preserves_learning() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), None, &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        let path = "test_engine_roundtrip.transitions";
        e.save(path).unwrap();

        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let e2 = RgdbEngine::load(g, path).unwrap();
        assert_eq!(e2.counts_snapshot(), e.counts_snapshot());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn expired_query_context_errors_and_is_evicted() {
        use std::time::Duration;
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let e = RgdbEngine::with_engine_config(
            g,
            prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
            EngineConfig {
                cache_capacity: 8,
                cache_ttl: Duration::from_millis(1),
                default_depth_weights: DepthWeights::uniform(4),
                depth_profile_learning: false,
            },
        );
        let r = e.query(&[(0, 1.0)], Some(0), None, &params());
        std::thread::sleep(Duration::from_millis(10));

        // Expired: must error, never silently use the stale context.
        assert!(matches!(
            e.record_feedback(r.query_id, 2, 1.0),
            Err(FeedbackError::UnknownQuery(_))
        ));
        // And it was popped, so a second attempt errors too.
        assert!(matches!(
            e.record_feedback(r.query_id, 2, 1.0),
            Err(FeedbackError::UnknownQuery(_))
        ));
    }

    #[test]
    fn credit_uses_the_vocab_captured_at_query_time() {
        // Graph: 0 -(A)-> 1 -(B)-> 3, 0 -(A)-> 2 -(C)-> 3, and 1 -(B)-> 4.
        // Under a UNIFORM matrix, crediting target 3 favours the C path 2:1 over the
        // B path (node 1 splits its mass across two out-edges; node 2 does not).
        // If credit ran under a LIVE matrix sharpened to punish A->C, that ratio
        // inverts. So `delta(A,C) > delta(A,B)` proves the captured vocab was used.
        let a1 = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let a2 = (2u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let b3 = (3u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let b4 = (4u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let c3 = (3u32, EdgeProps { attenuation: 0.0, relation: 2, is_portal: false });
        let g = Graph::from_adjacency(
            5,
            vec![vec![a1, a2], vec![b3, b4], vec![c3], vec![], vec![]],
            NodeProps::default(),
        )
        .unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into(), "C".into()]);
        let e = RgdbEngine::new(
            g,
            prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
        );
        let p = PropagationParams { max_depth: 2, min_intensity: 0.0, depth_weights: None };

        // 1) Issue the query first: it captures the uniform vocab.
        let q1 = e.query(&[(0, 1.0)], Some(0), None, &p);

        // 2) Sharpen the LIVE matrix to punish A->C, by crediting target 4 (reachable
        //    only via A then B) and rebuilding.
        let q2 = e.query(&[(0, 1.0)], Some(0), None, &p);
        e.record_feedback(q2.query_id, 4, 1000.0).unwrap();
        e.refresh();
        assert!(e.vocab().similarity(0, 2) < 0.2, "live matrix should now punish A->C");

        // 3) Feed back the OLD query. It must be credited under its captured vocab.
        let before = e.counts_snapshot();
        e.record_feedback(q1.query_id, 3, 1.0).unwrap();
        let after = e.counts_snapshot();

        let d_ab = after[1] - before[1]; // counts[A][B]
        let d_ac = after[2] - before[2]; // counts[A][C]
        assert!(
            d_ac > d_ab,
            "credit must use the query-time (uniform) vocab: dAC={d_ac} should exceed dAB={d_ab}"
        );
    }

    #[test]
    fn engine_default_depth_weights_are_applied_when_caller_omits_them() {
        use crate::depth_weights::DepthWeights;
        // Graph 0 -A-> 1 -B-> 2. terminal(2) scores ONLY depth-2 arrivals, so node 2
        // (depth 2) ranks above node 1 (depth 1); under uniform, node 1 outranks it.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let e = RgdbEngine::with_engine_config(
            g,
            prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
            EngineConfig {
                cache_capacity: 8,
                cache_ttl: Duration::from_secs(3600),
                default_depth_weights: DepthWeights::terminal(4, 2).unwrap(),
                depth_profile_learning: false,
            },
        );
        // Caller passes None -> engine applies its terminal(2) default (max_depth 4 matches).
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(2), "terminal(2) default ranks node 2 first");
    }

    #[test]
    fn caller_depth_weights_override_the_engine_default() {
        use crate::depth_weights::DepthWeights;
        let e = engine(0); // 0 -A-> 1 -B-> 2, default config = uniform
        // Caller forces terminal(1): node 1 (depth 1) must rank first.
        let p = PropagationParams {
            max_depth: 4,
            min_intensity: 0.0,
            depth_weights: Some(DepthWeights::terminal(4, 1).unwrap()),
        };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(1), "caller terminal(1) ranks node 1 first");
    }

    #[test]
    fn query_with_hop_hint_uses_learned_soft_weights_after_feedback() {
        // Graph 0 -A-> 1 -B-> 2. hop_hint=2 with a cold DepthProfileStore behaves like
        // terminal(2): node 2 (depth 2) outranks node 1 (depth 1).
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let e = RgdbEngine::with_engine_config(
            g, prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
            EngineConfig { depth_profile_learning: true, ..EngineConfig::default() },
        );
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
        assert_eq!(r.ranked.first().map(|x| x.0), Some(2), "cold hop_hint=2 == terminal(2)");

        // Feed back that node 1 (a depth-1 answer) is correct, repeatedly, then refresh
        // the profile. The learned weight at depth 1 rises, so node 1 can now score.
        for _ in 0..40 {
            let q = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
            let _ = e.record_feedback(q.query_id, 1, 1.0);
        }
        e.refresh_profiles();
        let after = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
        // depth-1 mass is now weighted, so node 1's score is nonzero (was 0 under terminal(2)).
        let n1 = after.ranked.iter().find(|x| x.0 == 1).map(|x| x.1).unwrap_or(0.0);
        assert!(n1 > 0.0, "learned soft weights should score the depth-1 node, got {n1}");
    }

    #[test]
    fn query_without_hop_hint_is_unchanged() {
        // hop_hint = None must reproduce the pre-feature behavior (config default path).
        let e = engine(0);
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert!(!r.ranked.is_empty());
        assert!(r.query_id > 0);
    }

    #[test]
    fn hop_hint_default_does_not_drift_without_learning() {
        // Default engine (depth_profile_learning = false): feedback must NOT move the
        // hop_hint weights off terminal(k). Node 1 (depth 1) stays scored 0 under hop_hint=2.
        let e = engine(0); // default config -> learning off
        let p = PropagationParams { max_depth: 4, min_intensity: 0.0, depth_weights: None };
        for _ in 0..40 {
            let q = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
            let _ = e.record_feedback(q.query_id, 1, 1.0);
        }
        let after = e.query(&[(0, 1.0)], Some(0), Some(2), &p);
        let n1 = after.ranked.iter().find(|x| x.0 == 1).map(|x| x.1).unwrap_or(0.0);
        assert_eq!(n1, 0.0, "learning off: hop_hint=2 stays terminal(2), depth-1 node scores 0");
    }
}
