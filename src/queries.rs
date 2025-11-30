/// Query engine for RGDB

use crate::graph::{Graph, NodeId, AngleBin};
use crate::propagation::{propagate_light, propagate_light_with_pvs, LightParams, intensity_to_distance};
use crate::pvs::PVS;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Epsilon value for intensity-to-distance conversion
const INTENSITY_EPSILON: f32 = 1e-6;
/// Minimum value for safe division (prevents division by zero)
const MIN_NORM: f32 = 1e-10;

/// Top-K influence query result
#[derive(Debug, Clone)]
pub struct InfluenceResult {
    pub node: NodeId,
    pub intensity: f32,
    pub distance: f32,
}

impl PartialEq for InfluenceResult {
    fn eq(&self, other: &Self) -> bool {
        self.intensity == other.intensity
    }
}

impl Eq for InfluenceResult {}

impl PartialOrd for InfluenceResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap: smaller intensity = higher priority
        other.intensity.partial_cmp(&self.intensity)
    }
}

impl Ord for InfluenceResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Query top-K nodes by influence from a source
pub fn query_top_k_influence(
    graph: &Graph,
    source: NodeId,
    initial_bin: AngleBin,
    k: usize,
    params: LightParams,
    pvs: Option<&PVS>,
) -> Vec<InfluenceResult> {
    let intensities = if let Some(pvs) = pvs {
        propagate_light_with_pvs(graph, source, initial_bin, params, Some(pvs))
    } else {
        propagate_light(graph, source, initial_bin, params)
    };
    
    let distances = intensity_to_distance(&intensities, INTENSITY_EPSILON);
    
    // Use min-heap to keep top-K
    let mut heap = BinaryHeap::with_capacity(k + 1);
    
    for (node_id, &intensity) in intensities.iter().enumerate() {
        if intensity > params.min_intensity {
            heap.push(InfluenceResult {
                node: node_id as NodeId,
                intensity,
                distance: distances[node_id],
            });
            
            if heap.len() > k {
                heap.pop(); // Remove smallest
            }
        }
    }
    
    // Convert to sorted vector (highest intensity first)
    let mut results: Vec<_> = heap.into_vec();
    results.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(Ordering::Equal));
    results
}

/// Query distance between two nodes
pub fn query_distance(
    graph: &Graph,
    source: NodeId,
    target: NodeId,
    initial_bin: AngleBin,
    params: LightParams,
    pvs: Option<&PVS>,
) -> Option<f32> {
    let intensities = if let Some(pvs) = pvs {
        propagate_light_with_pvs(graph, source, initial_bin, params, Some(pvs))
    } else {
        propagate_light(graph, source, initial_bin, params)
    };
    
    let target_idx = target as usize;
    if target_idx >= intensities.len() {
        return None;
    }
    
    let intensity = intensities[target_idx];
    if intensity <= params.min_intensity {
        return None;
    }
    
    let distance = intensity_to_distance(&[intensity], INTENSITY_EPSILON)[0];
    Some(distance)
}

/// Hybrid query combining graph influence with vector similarity
pub fn query_hybrid(
    graph: &Graph,
    source: NodeId,
    initial_bin: AngleBin,
    embeddings: &[ndarray::Array1<f32>],
    query_embedding: &ndarray::Array1<f32>,
    alpha: f32, // Weight for graph influence (1-alpha for vector similarity)
    k: usize,
    params: LightParams,
    pvs: Option<&PVS>,
) -> Vec<InfluenceResult> {
    // Get graph influence
    let intensities = if let Some(pvs) = pvs {
        propagate_light_with_pvs(graph, source, initial_bin, params, Some(pvs))
    } else {
        propagate_light(graph, source, initial_bin, params)
    };
    
    // Normalize intensities to [0, 1] with safe division
    let max_intensity = intensities.iter().copied().fold(0.0f32, f32::max);
    let normalized_intensities: Vec<f32> = if max_intensity > MIN_NORM {
        intensities.iter().map(|&i| i / max_intensity).collect()
    } else {
        // All intensities are zero or near-zero - return zeros
        vec![0.0; intensities.len()]
    };
    
    // Compute vector similarities
    let mut similarities = Vec::new();
    for (i, emb) in embeddings.iter().enumerate() {
        if i < normalized_intensities.len() {
            // Cosine similarity with safe division
            let dot = query_embedding.dot(emb);
            let norm_q = query_embedding.dot(query_embedding).sqrt();
            let norm_e = emb.dot(emb).sqrt();
            let similarity = if norm_q > MIN_NORM && norm_e > MIN_NORM {
                dot / (norm_q * norm_e)
            } else {
                0.0 // Vectors are zero or near-zero
            };
            
            // Hybrid score
            let hybrid_score = alpha * normalized_intensities[i] + (1.0 - alpha) * similarity;
            similarities.push((i, hybrid_score));
        }
    }
    
    // Sort and return top-K
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarities
        .into_iter()
        .take(k)
        .map(|(node_id, score)| InfluenceResult {
            node: node_id as NodeId,
            intensity: score,
            distance: intensity_to_distance(&[score], 1e-6)[0],
        })
        .collect()
}

