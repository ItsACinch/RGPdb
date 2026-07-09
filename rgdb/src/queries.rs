//! Query helpers over the sparse propagation kernel.

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate, intensity_to_distance, PropagationParams};
use crate::relation::RelationVocab;
use std::cmp::Ordering;

const INTENSITY_EPSILON: f32 = 1e-6;
const MIN_NORM: f32 = 1e-10;

#[derive(Debug, Clone)]
pub struct InfluenceResult {
    pub node: NodeId,
    pub intensity: f32,
    pub distance: f32,
}

/// Top-K nodes by influence, excluding the seed nodes themselves.
pub fn query_top_k_influence(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    k: usize,
    params: &PropagationParams,
) -> Vec<InfluenceResult> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let seed_set: std::collections::HashSet<NodeId> = seeds.iter().map(|&(n, _)| n).collect();

    let mut results: Vec<InfluenceResult> = totals
        .into_iter()
        .filter(|(node, _)| !seed_set.contains(node))
        .map(|(node, intensity)| InfluenceResult {
            node,
            intensity,
            distance: intensity_to_distance(&[intensity], INTENSITY_EPSILON)[0],
        })
        .collect();

    results.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(Ordering::Equal));
    results.truncate(k);
    results
}

/// Contextual distance from the seeds to a target (None if unreached).
pub fn query_distance(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    target: NodeId,
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> Option<f32> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let intensity = totals.get(&target).copied()?;
    if intensity <= params.min_intensity {
        return None;
    }
    Some(intensity_to_distance(&[intensity], INTENSITY_EPSILON)[0])
}

/// Hybrid query: fuse graph influence with cosine similarity over dense embeddings.
pub fn query_hybrid(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    embeddings: &[ndarray::Array1<f32>],
    query_embedding: &ndarray::Array1<f32>,
    alpha: f32,
    query_relation: Option<RelationId>,
    k: usize,
    params: &PropagationParams,
) -> Vec<InfluenceResult> {
    let totals = propagate(graph, vocab, seeds, query_relation, params);
    let norm_q = query_embedding.dot(query_embedding).sqrt();

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        let g = totals.get(&(i as NodeId)).copied().unwrap_or(0.0);
        let norm_e = emb.dot(emb).sqrt();
        let sim = if norm_q > MIN_NORM && norm_e > MIN_NORM {
            query_embedding.dot(emb) / (norm_q * norm_e)
        } else {
            0.0
        };
        scored.push((i, alpha * g + (1.0 - alpha) * sim));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(node, score)| InfluenceResult {
            node: node as NodeId,
            intensity: score,
            distance: intensity_to_distance(&[score], INTENSITY_EPSILON)[0],
        })
        .collect()
}
