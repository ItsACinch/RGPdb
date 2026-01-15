//! RAG Query Engine combining graph reasoning with vector similarity

use crate::graph::{Graph, NodeId, AngleBin};
use crate::propagation::{propagate_light, propagate_light_with_pvs, LightParams, intensity_to_distance};
use crate::pvs::PVS;
use crate::queries::InfluenceResult;

use super::embedding_store::EmbeddingStore;
use super::intent::{IntentClassifier, QueryIntent};
use super::personalization::UserContext;

use ndarray::Array1;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for RAG queries
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Maximum propagation depth
    pub max_hops: usize,
    /// Weight for graph influence (0.0-1.0)
    /// Higher = more graph, lower = more vector similarity
    pub alpha: f32,
    /// Weight for vector similarity (0.0-1.0)
    pub beta: f32,
    /// Weight for personalization boost (0.0-1.0)
    pub gamma: f32,
    /// Minimum relevance threshold
    pub min_relevance: f32,
    /// Refraction sharpness (higher = stricter direction matching)
    pub refraction_sharpness: f32,
    /// Number of source nodes for multi-source propagation
    pub num_sources: usize,
    /// Whether to include reasoning paths in results
    pub include_paths: bool,
    /// Whether to use multiple angle bins based on intent
    pub use_related_bins: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            max_hops: 4,
            alpha: 0.5,
            beta: 0.4,
            gamma: 0.1,
            min_relevance: 1e-3,
            refraction_sharpness: 5.0,
            num_sources: 5,
            include_paths: false,
            use_related_bins: true,
        }
    }
}

/// Extended query result with additional metadata
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Node ID
    pub node_id: NodeId,
    /// Combined relevance score
    pub score: f32,
    /// Graph propagation intensity
    pub graph_intensity: f32,
    /// Vector similarity score
    pub vector_similarity: f32,
    /// Personalization boost applied
    pub personalization_boost: f32,
    /// Distance from source (log of inverse intensity)
    pub distance: f32,
    /// Reasoning path (if include_paths is true)
    pub reasoning_path: Vec<NodeId>,
}

impl PartialEq for QueryResult {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for QueryResult {}

impl PartialOrd for QueryResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap: smaller score = higher priority (we want to keep largest)
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for QueryResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// RAG Query Engine combining graph reasoning with vector similarity
pub struct RAGQueryEngine {
    /// The graph database
    graph: Graph,
    /// Embedding store for vector similarity
    embedding_store: EmbeddingStore,
    /// PVS for query optimization
    pvs: Option<PVS>,
    /// Intent classifier
    intent_classifier: IntentClassifier,
}

impl RAGQueryEngine {
    /// Create a new RAG query engine
    pub fn new(graph: Graph, embedding_store: EmbeddingStore, pvs: Option<PVS>) -> Self {
        Self {
            graph,
            embedding_store,
            pvs,
            intent_classifier: IntentClassifier::new(),
        }
    }

    /// Get a reference to the graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get a reference to the embedding store
    pub fn embedding_store(&self) -> &EmbeddingStore {
        &self.embedding_store
    }

    /// Execute a RAG query with natural language
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &Array1<f32>,
        user_context: Option<&UserContext>,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        // 1. Classify query intent
        let intent = self.intent_classifier.classify(query_text);
        let primary_bin = intent.to_angle_bin();

        // 2. Find source nodes via vector similarity
        let source_candidates = self.embedding_store.top_k_similar(query_embedding, config.num_sources);
        if source_candidates.is_empty() {
            return Vec::new();
        }

        // 3. Build light params
        let params = LightParams {
            k: config.refraction_sharpness,
            min_intensity: config.min_relevance,
            max_depth: config.max_hops,
            num_angle_bins: crate::graph::N_ANGLE_BINS,
        };

        // 4. Determine angle bins to use
        let angle_bins = if config.use_related_bins {
            intent.related_bins()
        } else {
            vec![primary_bin]
        };

        // 5. Multi-source, multi-bin propagation
        let mut combined_intensity = vec![0.0f32; self.graph.num_nodes()];

        for source_result in &source_candidates {
            let source = source_result.node_id;

            for &angle_bin in &angle_bins {
                let intensities = if let Some(ref pvs) = self.pvs {
                    propagate_light_with_pvs(&self.graph, source, angle_bin, params, Some(pvs))
                } else {
                    propagate_light(&self.graph, source, angle_bin, params)
                };

                // Weight by source similarity and accumulate
                let source_weight = source_result.similarity.max(0.0);
                for (node, &intensity) in intensities.iter().enumerate() {
                    combined_intensity[node] += intensity * source_weight;
                }
            }
        }

        // 6. Normalize graph intensities
        let max_intensity = combined_intensity.iter().copied().fold(0.0f32, f32::max);
        let normalized_intensity: Vec<f32> = if max_intensity > 1e-10 {
            combined_intensity.iter().map(|&i| i / max_intensity).collect()
        } else {
            vec![0.0; combined_intensity.len()]
        };

        // 7. Compute vector similarities for all nodes
        let vector_similarities = self.embedding_store.all_similarities(query_embedding);

        // 8. Compute hybrid scores with personalization
        self.compute_hybrid_scores(
            &normalized_intensity,
            &vector_similarities,
            user_context,
            config,
        )
    }

    /// Execute a query starting from specific source nodes
    pub fn query_from_sources(
        &self,
        sources: &[NodeId],
        angle_bin: AngleBin,
        query_embedding: Option<&Array1<f32>>,
        user_context: Option<&UserContext>,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        let params = LightParams {
            k: config.refraction_sharpness,
            min_intensity: config.min_relevance,
            max_depth: config.max_hops,
            num_angle_bins: crate::graph::N_ANGLE_BINS,
        };

        // Multi-source propagation
        let mut combined_intensity = vec![0.0f32; self.graph.num_nodes()];

        for &source in sources {
            let intensities = if let Some(ref pvs) = self.pvs {
                propagate_light_with_pvs(&self.graph, source, angle_bin, params, Some(pvs))
            } else {
                propagate_light(&self.graph, source, angle_bin, params)
            };

            for (node, &intensity) in intensities.iter().enumerate() {
                combined_intensity[node] += intensity;
            }
        }

        // Normalize
        let max_intensity = combined_intensity.iter().copied().fold(0.0f32, f32::max);
        let normalized_intensity: Vec<f32> = if max_intensity > 1e-10 {
            combined_intensity.iter().map(|&i| i / max_intensity).collect()
        } else {
            vec![0.0; combined_intensity.len()]
        };

        // Vector similarities (if query embedding provided)
        let vector_similarities = if let Some(emb) = query_embedding {
            self.embedding_store.all_similarities(emb)
        } else {
            vec![0.0; self.graph.num_nodes()]
        };

        self.compute_hybrid_scores(
            &normalized_intensity,
            &vector_similarities,
            user_context,
            config,
        )
    }

    /// Compute hybrid scores combining graph, vector, and personalization
    fn compute_hybrid_scores(
        &self,
        graph_intensity: &[f32],
        vector_similarity: &[f32],
        user_context: Option<&UserContext>,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        let num_nodes = graph_intensity.len().min(vector_similarity.len());

        // Use min-heap for top-K selection
        let mut heap: BinaryHeap<QueryResult> = BinaryHeap::with_capacity(config.top_k + 1);

        for node_id in 0..num_nodes {
            let g_score = graph_intensity[node_id];
            let v_score = vector_similarity.get(node_id).copied().unwrap_or(0.0);

            // Skip nodes with no signal
            if g_score < config.min_relevance && v_score < config.min_relevance {
                continue;
            }

            // Compute personalization boost
            let p_boost = if let Some(ctx) = user_context {
                let room = self.graph.room_map().get(node_id).copied();
                ctx.compute_boost(node_id as NodeId, room)
            } else {
                1.0
            };

            // Skip inaccessible nodes
            if p_boost <= 0.0 {
                continue;
            }

            // Hybrid score: alpha*graph + beta*vector + gamma*personalization
            let base_score = config.alpha * g_score + config.beta * v_score;
            let final_score = base_score * (1.0 + config.gamma * (p_boost - 1.0));

            let result = QueryResult {
                node_id: node_id as NodeId,
                score: final_score,
                graph_intensity: g_score,
                vector_similarity: v_score,
                personalization_boost: p_boost,
                distance: intensity_to_distance(&[final_score], 1e-6)[0],
                reasoning_path: Vec::new(), // TODO: Track paths if config.include_paths
            };

            heap.push(result);

            if heap.len() > config.top_k {
                heap.pop();
            }
        }

        // Convert to sorted vector (highest score first)
        let mut results: Vec<_> = heap.into_vec();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Generate activity feed based on user context
    pub fn generate_feed(
        &self,
        user_context: &UserContext,
        config: &QueryConfig,
    ) -> Vec<QueryResult> {
        // Get seed nodes from user's recent activity
        let seeds = user_context.get_feed_seeds(config.num_sources);
        if seeds.is_empty() {
            return Vec::new();
        }

        let params = LightParams {
            k: config.refraction_sharpness,
            min_intensity: config.min_relevance,
            max_depth: config.max_hops.min(3), // Limit depth for feed
            num_angle_bins: crate::graph::N_ANGLE_BINS,
        };

        // Propagate from seeds in user's preferred directions
        let preferred_bins = user_context.preferred_bins();
        let mut combined_intensity = vec![0.0f32; self.graph.num_nodes()];

        for &seed in &seeds {
            // Use top 3 preferred bins
            for &(bin_idx, affinity) in preferred_bins.iter().take(3) {
                if affinity < 0.1 {
                    continue;
                }

                let intensities = if let Some(ref pvs) = self.pvs {
                    propagate_light_with_pvs(&self.graph, seed, bin_idx as AngleBin, params, Some(pvs))
                } else {
                    propagate_light(&self.graph, seed, bin_idx as AngleBin, params)
                };

                // Weight by affinity
                for (node, &intensity) in intensities.iter().enumerate() {
                    combined_intensity[node] += intensity * affinity;
                }
            }
        }

        // Normalize
        let max_intensity = combined_intensity.iter().copied().fold(0.0f32, f32::max);
        let normalized: Vec<f32> = if max_intensity > 1e-10 {
            combined_intensity.iter().map(|&i| i / max_intensity).collect()
        } else {
            vec![0.0; combined_intensity.len()]
        };

        // No vector similarity for feed (purely graph-based)
        let zero_similarity = vec![0.0; self.graph.num_nodes()];

        // Apply personalization with higher weight
        let mut feed_config = config.clone();
        feed_config.alpha = 0.8;
        feed_config.beta = 0.0;
        feed_config.gamma = 0.2;

        self.compute_hybrid_scores(&normalized, &zero_similarity, Some(user_context), &feed_config)
    }

    /// Find reasoning paths between two concepts
    pub fn find_reasoning_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_depth: usize,
    ) -> Option<Vec<NodeId>> {
        // Simple BFS to find path
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parents: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

        queue.push_back(source);
        visited.insert(source);

        while let Some(current) = queue.pop_front() {
            if current == target {
                // Reconstruct path
                let mut path = vec![target];
                let mut node = target;
                while node != source {
                    if let Some(&parent) = parents.get(&node) {
                        path.push(parent);
                        node = parent;
                    } else {
                        break;
                    }
                }
                path.reverse();
                return Some(path);
            }

            if path_length(&parents, source, current) >= max_depth {
                continue;
            }

            for (neighbor, _) in self.graph.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parents.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Convert to basic InfluenceResult for compatibility
    pub fn to_influence_results(results: Vec<QueryResult>) -> Vec<InfluenceResult> {
        results
            .into_iter()
            .map(|r| InfluenceResult {
                node: r.node_id,
                intensity: r.score,
                distance: r.distance,
            })
            .collect()
    }
}

/// Helper to compute path length
fn path_length(
    parents: &std::collections::HashMap<NodeId, NodeId>,
    source: NodeId,
    current: NodeId,
) -> usize {
    let mut length = 0;
    let mut node = current;
    while node != source {
        if let Some(&parent) = parents.get(&node) {
            length += 1;
            node = parent;
        } else {
            break;
        }
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_engine() -> RAGQueryEngine {
        // Create a simple test graph
        let graph = Graph::new(5).unwrap();

        // Create test embeddings
        let embeddings = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.8, 0.2, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.0, 0.0,
            ],
        ).unwrap();

        let embedding_store = EmbeddingStore::new(embeddings);

        RAGQueryEngine::new(graph, embedding_store, None)
    }

    #[test]
    fn test_query_engine_creation() {
        let engine = create_test_engine();
        assert_eq!(engine.graph().num_nodes(), 5);
        assert_eq!(engine.embedding_store().num_embeddings(), 5);
    }

    #[test]
    fn test_default_config() {
        let config = QueryConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.max_hops, 4);
        assert!(config.alpha > 0.0);
    }

    #[test]
    fn test_query_result_ordering() {
        let r1 = QueryResult {
            node_id: 0,
            score: 0.8,
            graph_intensity: 0.8,
            vector_similarity: 0.0,
            personalization_boost: 1.0,
            distance: 0.0,
            reasoning_path: vec![],
        };
        let r2 = QueryResult {
            node_id: 1,
            score: 0.5,
            graph_intensity: 0.5,
            vector_similarity: 0.0,
            personalization_boost: 1.0,
            distance: 0.0,
            reasoning_path: vec![],
        };

        // For min-heap: r2 should be "greater" since we want to keep largest
        assert!(r1 < r2);
    }
}
