//! RAG query engine: seeds from vector search, typed-PPR diffusion, calibrated fusion.

use crate::graph::{Graph, NodeId};
use crate::propagation::{propagate, intensity_to_distance, PropagationParams};
use crate::relation::RelationVocab;
use crate::queries::InfluenceResult;

use super::embedding_store::EmbeddingStore;
use super::intent::IntentClassifier;
use super::personalization::UserContext;

use ndarray::Array1;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct QueryConfig {
    pub top_k: usize,
    pub max_hops: usize,
    pub alpha: f32,          // graph weight
    pub beta: f32,           // vector weight
    pub gamma: f32,          // personalization weight
    pub min_relevance: f32,
    pub num_sources: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self { top_k: 10, max_hops: 4, alpha: 0.5, beta: 0.4, gamma: 0.1,
               min_relevance: 1e-3, num_sources: 5 }
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub node_id: NodeId,
    pub score: f32,
    pub graph_intensity: f32,
    pub vector_similarity: f32,
    pub personalization_boost: f32,
    pub distance: f32,
}

pub struct RAGQueryEngine {
    graph: Graph,
    vocab: RelationVocab,
    embedding_store: EmbeddingStore,
    intent_classifier: IntentClassifier,
}

impl RAGQueryEngine {
    pub fn new(graph: Graph, vocab: RelationVocab, embedding_store: EmbeddingStore) -> Self {
        Self { graph, vocab, embedding_store, intent_classifier: IntentClassifier::new() }
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    pub fn embedding_store(&self) -> &EmbeddingStore { &self.embedding_store }

    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &Array1<f32>,
        user_context: Option<&UserContext>,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        // 1. Intent -> relation id (None if the vocab lacks it -> untyped query).
        let intent = self.intent_classifier.classify(query_text);
        let query_relation = self.vocab.id_of(intent.relation_name());

        // 2. Vector seeds, weights normalized to sum to 1 (calibrated graph mass).
        let cands = self.embedding_store.top_k_similar(query_embedding, config.num_sources);
        if cands.is_empty() {
            return Vec::new();
        }
        let sum: f32 = cands.iter().map(|c| c.similarity.max(0.0)).sum();
        let seeds: Vec<(NodeId, f32)> = if sum > 0.0 {
            cands.iter().map(|c| (c.node_id, c.similarity.max(0.0) / sum)).collect()
        } else {
            cands.iter().map(|c| (c.node_id, 1.0 / cands.len() as f32)).collect()
        };

        // 3. Diffuse.
        let params = PropagationParams { max_depth: config.max_hops, min_intensity: config.min_relevance };
        let totals = propagate(&self.graph, &self.vocab, &seeds, query_relation, &params);

        // 4. Vector similarities for all nodes (dense; ANN is future work).
        let sims = self.embedding_store.all_similarities(query_embedding);

        // 5. Fuse (no max-normalization; graph mass is already calibrated).
        let mut results: Vec<QueryResult> = Vec::new();
        for node in 0..self.graph.num_nodes() {
            let g = totals.get(&(node as NodeId)).copied().unwrap_or(0.0);
            let v = sims.get(node).copied().unwrap_or(0.0);
            if g < config.min_relevance && v < config.min_relevance {
                continue;
            }
            let p_boost = match user_context {
                Some(ctx) => ctx.compute_boost(node as NodeId, self.graph.room_map().get(node).copied()),
                None => 1.0,
            };
            if p_boost <= 0.0 {
                continue;
            }
            let base = config.alpha * g + config.beta * v;
            let score = base * (1.0 + config.gamma * (p_boost - 1.0));
            results.push(QueryResult {
                node_id: node as NodeId,
                score,
                graph_intensity: g,
                vector_similarity: v,
                personalization_boost: p_boost,
                distance: intensity_to_distance(&[score], 1e-6)[0],
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(config.top_k);
        results
    }

    pub fn to_influence_results(results: Vec<QueryResult>) -> Vec<InfluenceResult> {
        results.into_iter().map(|r| InfluenceResult {
            node: r.node_id, intensity: r.score, distance: r.distance,
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::RelationVocab;
    use ndarray::Array2;

    fn engine() -> RAGQueryEngine {
        let graph = Graph::new(5).unwrap();
        let emb = Array2::from_shape_vec(
            (5, 4),
            vec![1.0,0.0,0.0,0.0, 0.8,0.2,0.0,0.0, 0.0,1.0,0.0,0.0,
                 0.0,0.0,1.0,0.0, 0.5,0.5,0.0,0.0],
        ).unwrap();
        RAGQueryEngine::new(graph, RelationVocab::uniform(1), EmbeddingStore::new(emb))
    }

    #[test]
    fn engine_builds_and_queries() {
        let e = engine();
        assert_eq!(e.graph().num_nodes(), 5);
        let q = ndarray::arr1(&[1.0, 0.0, 0.0, 0.0]);
        let res = e.query("what is x?", &q, None, &QueryConfig::default());
        assert!(res.len() <= QueryConfig::default().top_k);
    }
}
