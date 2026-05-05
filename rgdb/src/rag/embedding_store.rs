//! Embedding storage and similarity search for RAG queries

use crate::graph::NodeId;
use ndarray::{Array1, Array2};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Minimum vector norm to avoid division by zero
const MIN_NORM: f32 = 1e-10;

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub node_id: NodeId,
    pub similarity: f32,
}

impl PartialEq for SimilarityResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for SimilarityResult {}

impl PartialOrd for SimilarityResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimilarityResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (we want to keep highest similarities)
        other.similarity.partial_cmp(&self.similarity).unwrap_or(Ordering::Equal)
    }
}

/// Storage for node embeddings with similarity search
pub struct EmbeddingStore {
    /// Dense embeddings: [num_nodes, embedding_dim]
    embeddings: Array2<f32>,
    /// Embedding dimension
    dim: usize,
    /// Precomputed norms for each embedding
    norms: Vec<f32>,
}

impl EmbeddingStore {
    /// Create a new embedding store from a 2D array
    pub fn new(embeddings: Array2<f32>) -> Self {
        let dim = embeddings.ncols();
        let num_nodes = embeddings.nrows();

        // Precompute norms for efficient cosine similarity
        let mut norms = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let row = embeddings.row(i);
            let norm = row.dot(&row).sqrt();
            norms.push(norm);
        }

        Self {
            embeddings,
            dim,
            norms,
        }
    }

    /// Create an empty embedding store with given dimensions
    pub fn empty(num_nodes: usize, dim: usize) -> Self {
        let embeddings = Array2::zeros((num_nodes, dim));
        let norms = vec![0.0; num_nodes];

        Self {
            embeddings,
            dim,
            norms,
        }
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of stored embeddings
    pub fn num_embeddings(&self) -> usize {
        self.embeddings.nrows()
    }

    /// Get embedding for a specific node
    pub fn get_embedding(&self, node_id: NodeId) -> Option<Array1<f32>> {
        let idx = node_id as usize;
        if idx < self.embeddings.nrows() {
            Some(self.embeddings.row(idx).to_owned())
        } else {
            None
        }
    }

    /// Set embedding for a specific node
    pub fn set_embedding(&mut self, node_id: NodeId, embedding: &Array1<f32>) {
        let idx = node_id as usize;
        if idx < self.embeddings.nrows() && embedding.len() == self.dim {
            for (j, &val) in embedding.iter().enumerate() {
                self.embeddings[[idx, j]] = val;
            }
            // Update precomputed norm
            self.norms[idx] = embedding.dot(embedding).sqrt();
        }
    }

    /// Compute cosine similarity between query and a node embedding
    pub fn cosine_similarity(&self, query: &Array1<f32>, node_id: NodeId) -> f32 {
        let idx = node_id as usize;
        if idx >= self.embeddings.nrows() {
            return 0.0;
        }

        let embedding = self.embeddings.row(idx);
        let dot = query.dot(&embedding);
        let query_norm = query.dot(query).sqrt();
        let emb_norm = self.norms[idx];

        if query_norm > MIN_NORM && emb_norm > MIN_NORM {
            dot / (query_norm * emb_norm)
        } else {
            0.0
        }
    }

    /// Find top-K most similar nodes to a query embedding
    pub fn top_k_similar(&self, query: &Array1<f32>, k: usize) -> Vec<SimilarityResult> {
        let query_norm = query.dot(query).sqrt();
        if query_norm < MIN_NORM {
            return Vec::new();
        }

        // Use a min-heap to maintain top-K
        let mut heap: BinaryHeap<SimilarityResult> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.embeddings.nrows() {
            let emb_norm = self.norms[idx];
            if emb_norm < MIN_NORM {
                continue;
            }

            let embedding = self.embeddings.row(idx);
            let dot = query.dot(&embedding);
            let similarity = dot / (query_norm * emb_norm);

            heap.push(SimilarityResult {
                node_id: idx as NodeId,
                similarity,
            });

            if heap.len() > k {
                heap.pop();
            }
        }

        // Convert to sorted vector (highest similarity first)
        let mut results: Vec<_> = heap.into_vec();
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));
        results
    }

    /// Compute similarities for all nodes (useful for hybrid scoring)
    pub fn all_similarities(&self, query: &Array1<f32>) -> Vec<f32> {
        let query_norm = query.dot(query).sqrt();
        let num_nodes = self.embeddings.nrows();

        if query_norm < MIN_NORM {
            return vec![0.0; num_nodes];
        }

        let mut similarities = Vec::with_capacity(num_nodes);
        for idx in 0..num_nodes {
            let emb_norm = self.norms[idx];
            if emb_norm < MIN_NORM {
                similarities.push(0.0);
                continue;
            }

            let embedding = self.embeddings.row(idx);
            let dot = query.dot(&embedding);
            similarities.push(dot / (query_norm * emb_norm));
        }

        similarities
    }

    /// Save embeddings to a binary file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write header: num_nodes (u32), dim (u32)
        let num_nodes = self.embeddings.nrows() as u32;
        let dim = self.dim as u32;
        file.write_all(&num_nodes.to_le_bytes())?;
        file.write_all(&dim.to_le_bytes())?;

        // Write embeddings as flat f32 array
        for val in self.embeddings.iter() {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load embeddings from a binary file
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Read header
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        let num_nodes = u32::from_le_bytes(buf) as usize;
        file.read_exact(&mut buf)?;
        let dim = u32::from_le_bytes(buf) as usize;

        // Read embeddings
        let mut data = vec![0.0f32; num_nodes * dim];
        for val in data.iter_mut() {
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        let embeddings = Array2::from_shape_vec((num_nodes, dim), data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(Self::new(embeddings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_embedding_store_basic() {
        let embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0,  // Node 0
                0.0, 1.0, 0.0, 0.0,  // Node 1
                1.0, 1.0, 0.0, 0.0,  // Node 2 (similar to both)
            ],
        ).unwrap();

        let store = EmbeddingStore::new(embeddings);
        assert_eq!(store.num_embeddings(), 3);
        assert_eq!(store.dim(), 4);
    }

    #[test]
    fn test_cosine_similarity() {
        let embeddings = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0,  // Node 0
                1.0, 0.0, 0.0,  // Node 1 (identical)
            ],
        ).unwrap();

        let store = EmbeddingStore::new(embeddings);
        let query = array![1.0, 0.0, 0.0];

        let sim = store.cosine_similarity(&query, 0);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_similar() {
        let embeddings = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.0,   // Node 0
                0.9, 0.1,   // Node 1 (very similar to 0)
                0.0, 1.0,   // Node 2 (orthogonal)
                -1.0, 0.0,  // Node 3 (opposite)
            ],
        ).unwrap();

        let store = EmbeddingStore::new(embeddings);
        let query = array![1.0, 0.0];

        let results = store.top_k_similar(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node_id, 0);  // Most similar
        assert_eq!(results[1].node_id, 1);  // Second most similar
    }
}
