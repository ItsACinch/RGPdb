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
        let graph_max = graph.edge_props().iter().map(|e| e.relation as usize).max().unwrap_or(0);
        if store.len() <= graph_max {
            return Err(TransitionError::RelationCoverage { store_len: store.len(), graph_max });
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

        let should_refresh = {
            let mut store = self.store.lock().unwrap();
            store.record(&credits, signal);
            let n = store.config().rebuild_every_n;
            n > 0 && store.events_since_rebuild() >= n
        };
        if should_refresh {
            self.refresh();
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
        let e = engine(0); // manual refresh only
        let r = e.query(&[(0, 1.0)], Some(0), &params());
        e.record_feedback(r.query_id, 2, 100.0).unwrap();
        // not refreshed yet -> still uniform
        assert_eq!(e.vocab().similarity(0, 1), 1.0);
        e.refresh();
        // A->B was credited; A->A (diagonal) stays pinned
        assert_eq!(e.vocab().similarity(0, 0), 1.0);
        assert_eq!(e.vocab().similarity(0, 1), 1.0, "credited transition becomes the row max");
        // B has no outgoing evidence, so its row stays uniform
        assert_eq!(e.vocab().similarity(1, 0), 1.0);
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
}
