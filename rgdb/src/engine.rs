//! `RgdbEngine`: graph + live vocab + learned transitions + feedback loop.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use lru::LruCache;
use thiserror::Error;

use crate::credit::credit;
use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate, PropagationParams};
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

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub cache_capacity: usize,
    pub cache_ttl: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { cache_capacity: 4096, cache_ttl: Duration::from_secs(3600) }
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
    created: Instant,
}

pub struct RgdbEngine {
    graph: Graph,
    vocab: ArcSwap<RelationVocab>,
    store: Mutex<TransitionStore>,
    cache: Mutex<LruCache<QueryId, QueryContext>>,
    next_id: AtomicU64,
    engine_cfg: EngineConfig,
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
        params: &PropagationParams,
    ) -> QueryResult {
        let vocab = self.vocab.load_full();
        let totals = propagate(&self.graph, &vocab, seeds, query_relation, params);
        let mut ranked: Vec<(NodeId, f32)> = totals.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache.lock().unwrap().put(
            query_id,
            QueryContext {
                seeds: seeds.to_vec(),
                query_relation,
                params: *params,
                vocab,
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

        // Credit under the vocab that produced the ranking, not the live one.
        let credits = credit(
            &self.graph,
            &ctx.vocab,
            &ctx.seeds,
            ctx.query_relation,
            target,
            &ctx.params,
        );
        if credits.is_empty() {
            return Err(FeedbackError::TargetUnreachable(target));
        }

        // Record and (if the threshold is crossed) rebuild inside ONE critical
        // section. Splitting them lets two concurrent feedback events each observe
        // `>= rebuild_every_n` and both call rebuild(), applying decay twice for a
        // single threshold crossing.
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
        Ok(())
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

    fn params() -> PropagationParams { PropagationParams { max_depth: 4, min_intensity: 0.0 } }

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
        let r = e.query(&[(0, 1.0)], Some(0), &params());
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
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        assert!(matches!(e.record_feedback(r.query_id, 99, 1.0), Err(FeedbackError::InvalidTarget(_))));
    }

    #[test]
    fn feedback_on_unreachable_target_errors() {
        // seed at node 2 (a sink): nothing is reachable
        let e = engine(0);
        let r = e.query(&[(2, 1.0)], Some(0), &params());
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

        let p = PropagationParams { max_depth: 2, min_intensity: 0.0 };
        let r = e.query(&[(0, 1.0)], Some(0), &p);
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
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        assert_eq!(e.events_since_rebuild(), 0, "auto-refresh reset the counter");
    }

    #[test]
    fn save_load_roundtrip_preserves_learning() {
        let e = engine(0);
        let r = e.query(&[(0, 1.0)], Some(0), &params());
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
            EngineConfig { cache_capacity: 8, cache_ttl: Duration::from_millis(1) },
        );
        let r = e.query(&[(0, 1.0)], Some(0), &params());
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
        let p = PropagationParams { max_depth: 2, min_intensity: 0.0 };

        // 1) Issue the query first: it captures the uniform vocab.
        let q1 = e.query(&[(0, 1.0)], Some(0), &p);

        // 2) Sharpen the LIVE matrix to punish A->C, by crediting target 4 (reachable
        //    only via A then B) and rebuilding.
        let q2 = e.query(&[(0, 1.0)], Some(0), &p);
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
}
