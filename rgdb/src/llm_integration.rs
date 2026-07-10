compile_error!(
    "The llm-chain integration targets the pre-rewrite angle-bin propagation API \
     and has not been ported to the sparse typed-PPR model. It is intentionally \
     disabled. Track the port in a separate effort; do not build with --features llm."
);

/// LLM integration using llm-chain for query understanding

use crate::graph::{Graph, NodeId, AngleBin};
use std::collections::HashMap;

/// Query parser that converts natural language to graph queries
pub struct QueryParser {
    /// Mapping from keywords/phrases to node IDs (simplified)
    node_keywords: HashMap<String, NodeId>,
    /// Mapping from direction phrases to angle bins
    direction_keywords: HashMap<String, AngleBin>,
}

impl QueryParser {
    pub fn new() -> Self {
        let mut node_keywords = HashMap::new();
        // Example mappings (would be learned from graph or provided)
        node_keywords.insert("start".to_string(), 0);
        node_keywords.insert("beginning".to_string(), 0);
        
        let mut direction_keywords = HashMap::new();
        for i in 0..crate::graph::N_ANGLE_BINS {
            direction_keywords.insert(format!("direction_{}", i), i as AngleBin);
        }
        
        Self {
            node_keywords,
            direction_keywords,
        }
    }
    
    /// Parse natural language query to (source_node, angle_bin)
    pub fn parse_query(&self, query: &str) -> Option<(NodeId, AngleBin)> {
        // Simplified parsing - in real implementation would use LLM
        let query_lower = query.to_lowercase();
        
        // Try to find source node
        let mut source = None;
        for (keyword, &node_id) in &self.node_keywords {
            if query_lower.contains(keyword) {
                source = Some(node_id);
                break;
            }
        }
        
        // Default angle bin
        let angle_bin = 0;
        
        source.map(|s| (s, angle_bin))
    }
    
    /// Use LLM to understand query and extract parameters
    pub async fn parse_with_llm<S: llm_chain::traits::Step>(
        &self,
        query: &str,
        _llm_chain: &llm_chain::chains::sequential::Chain<S>,
    ) -> Option<(NodeId, AngleBin)> {
        // TODO: Integrate with llm-chain to parse queries
        // For now, fall back to simple parsing
        self.parse_query(query)
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

/// RAG integration combining vector search with graph reasoning
pub struct RAGEngine {
    graph: Graph,
    embeddings: ndarray::Array2<f32>,
    query_parser: QueryParser,
}

impl RAGEngine {
    pub fn new(graph: Graph, embeddings: ndarray::Array2<f32>) -> Self {
        Self {
            graph,
            embeddings,
            query_parser: QueryParser::new(),
        }
    }
    
    /// Perform RAG query combining retrieval and reasoning
    pub fn query(
        &self,
        query: &str,
        k: usize,
    ) -> Vec<crate::queries::InfluenceResult> {
        // Parse query
        if let Some((source, angle_bin)) = self.query_parser.parse_query(query) {
            // Use graph propagation for reasoning
            let params = crate::propagation::LightParams::default();
            crate::queries::query_top_k_influence(
                &self.graph,
                source,
                angle_bin,
                k,
                params,
                None,
            )
        } else {
            Vec::new()
        }
    }
}

