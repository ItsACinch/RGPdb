//! Multi-source contextual embedding generation via propagation.

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::{propagate_single, PropagationParams};
use crate::relation::RelationVocab;
use ndarray::Array2;

/// One embedding column per (source, query_relation) pair, normalized to [0,1].
pub fn generate_embeddings(
    graph: &Graph,
    vocab: &RelationVocab,
    sources: &[NodeId],
    query_relations: &[Option<RelationId>],
    params: &PropagationParams,
) -> Array2<f32> {
    let num_nodes = graph.num_nodes();
    let num_sources = sources.len();
    let mut m = Array2::zeros((num_nodes, num_sources));

    for (i, (&src, &rel)) in sources.iter().zip(query_relations.iter()).enumerate() {
        let totals = propagate_single(graph, vocab, src, 1.0, rel, params);
        let max = totals.values().copied().fold(0.0f32, f32::max);
        if max > 0.0 {
            for (node, intensity) in &totals {
                m[[*node as usize, i]] = intensity / max;
            }
        }
    }
    m
}

/// Export a dense embedding matrix as CSV.
pub fn export_embeddings_csv(embeddings: &Array2<f32>, path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(path)?;
    let (num_nodes, dim) = embeddings.dim();
    for i in 0..num_nodes {
        for j in 0..dim {
            write!(file, "{:.6}", embeddings[[i, j]])?;
            if j < dim - 1 {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }
    Ok(())
}
