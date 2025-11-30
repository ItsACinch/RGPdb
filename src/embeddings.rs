/// Multi-hop contextual embedding generation

use crate::graph::{Graph, NodeId, AngleBin};
use crate::propagation::{propagate_light, propagate_light_with_pvs, LightParams};
use crate::pvs::PVS;
use ndarray::Array2;

/// Generate embeddings from multiple source nodes
pub fn generate_embeddings(
    graph: &Graph,
    sources: &[NodeId],
    initial_bins: &[AngleBin],
    params: LightParams,
    pvs: Option<&PVS>,
) -> Array2<f32> {
    let num_nodes = graph.num_nodes();
    let num_sources = sources.len();
    
    let mut embedding_matrix = Array2::zeros((num_nodes, num_sources));
    
    for (i, (&source, &initial_bin)) in sources.iter().zip(initial_bins.iter()).enumerate() {
        let intensities = if let Some(pvs) = pvs {
            propagate_light_with_pvs(graph, source, initial_bin, params, Some(pvs))
        } else {
            propagate_light(graph, source, initial_bin, params)
        };
        
        // Normalize intensities
        let max_intensity = intensities.iter().copied().fold(0.0f32, f32::max);
        if max_intensity > 0.0 {
            for (node_id, &intensity) in intensities.iter().enumerate() {
                embedding_matrix[[node_id, i]] = intensity / max_intensity;
            }
        }
    }
    
    embedding_matrix
}

/// Generate embeddings with dimension reduction (PCA-like)
pub fn generate_reduced_embeddings(
    graph: &Graph,
    sources: &[NodeId],
    initial_bins: &[AngleBin],
    target_dim: usize,
    params: LightParams,
    pvs: Option<&PVS>,
) -> Array2<f32> {
    // Generate full embeddings
    let full_embeddings = generate_embeddings(graph, sources, initial_bins, params, pvs);
    
    // Simple dimension reduction: use first target_dim principal components
    // For now, just take first target_dim columns (simplified)
    let num_nodes = full_embeddings.nrows();
    let num_cols = full_embeddings.ncols().min(target_dim);
    
    let mut reduced = Array2::zeros((num_nodes, target_dim));
    for i in 0..num_nodes {
        for j in 0..num_cols {
            reduced[[i, j]] = full_embeddings[[i, j]];
        }
    }
    
    reduced
}

/// Export embeddings to numpy-compatible format (CSV for now)
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

