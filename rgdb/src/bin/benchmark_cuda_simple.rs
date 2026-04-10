/// Simple CUDA vs CPU performance benchmark (single run to avoid cleanup issues)

use rgdb::*;
use rgdb::property_map::{RelationshipProperty, create_directional_luminance};
use std::time::Instant;

fn create_test_graph(num_nodes: usize) -> graph::Graph {
    let mut adj = Vec::new();
    
    for i in 0..num_nodes {
        let mut neighbors = Vec::new();
        for j in 1..=8 {
            let target = (i + j) % num_nodes;
            if target != i {
                neighbors.push((
                    target as graph::NodeId,
                    graph::EdgeProps {
                        attenuation: 0.1 + (j as f32) * 0.02,
                        angle_bin: ((i + j) % graph::N_ANGLE_BINS) as graph::AngleBin,
                        is_portal: false,
                    },
                ));
            }
        }
        adj.push(neighbors);
    }
    
    let node_props = graph::NodeProps {
        relationship_property: Some(RelationshipProperty::IsA),
        directional_luminance: create_directional_luminance(RelationshipProperty::IsA, 2.0, 0.8),
        ..graph::NodeProps::default()
    };
    
    graph::Graph::from_adjacency(num_nodes, adj, node_props)
        .expect("Failed to create graph")
}

fn main() {
    println!("=== CUDA vs CPU Performance Benchmark (Single Run) ===\n");
    
    // Check CUDA availability
    let cuda_available = cuda::CudaContext::is_available();
    println!("CUDA Available: {}", cuda_available);
    
    if !cuda_available {
        println!("CUDA not available - CPU only");
        return;
    }
    
    // Create CUDA context
    let mut cuda_ctx = match cuda::CudaContext::new() {
        Ok(ctx) => {
            if ctx.kernels_loaded() {
                println!("CUDA kernels loaded: ✓");
                Some(ctx)
            } else {
                println!("CUDA kernels not loaded - using CPU fallback");
                None
            }
        }
        Err(e) => {
            println!("Failed to initialize CUDA: {}", e);
            None
        }
    };
    
    if cuda_ctx.is_none() {
        return;
    }
    
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: graph::N_ANGLE_BINS,
    };
    
    let test_sizes = vec![100, 500, 1000, 5000];
    let source: graph::NodeId = 0;
    let initial_bin: graph::AngleBin = 2;
    
    println!("\n{:<10} {:<15} {:<15} {:<15}", "Nodes", "CPU (ms)", "CUDA (ms)", "Speedup");
    println!("{}", "-".repeat(55));
    
    for &num_nodes in &test_sizes {
        print!("Creating graph with {} nodes... ", num_nodes);
        let graph = create_test_graph(num_nodes);
        println!("done");
        
        // Benchmark CPU
        let cpu_start = Instant::now();
        let cpu_result = propagation::propagate_light(&graph, source, initial_bin, params);
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;
        
        // Benchmark CUDA (single run to avoid cleanup issues)
        if let Some(ref mut ctx) = cuda_ctx {
            match cuda::propagate_light_cuda(ctx, &graph, source, initial_bin, params, None) {
                Ok(cuda_result) => {
                    let cuda_start = Instant::now();
                    let _ = cuda::propagate_light_cuda(ctx, &graph, source, initial_bin, params, None);
                    let cuda_time = cuda_start.elapsed().as_secs_f64() * 1000.0;
                    
                    let speedup = cpu_time / cuda_time;
                    let match_result = cpu_result.iter()
                        .zip(cuda_result.iter())
                        .all(|(c, g)| (c - g).abs() < 0.01);
                    
                    println!(
                        "{:<10} {:<15.3} {:<15.3} {:<15.2}x {}",
                        num_nodes,
                        cpu_time,
                        cuda_time,
                        speedup,
                        if match_result { "✓" } else { "✗" }
                    );
                }
                Err(e) => {
                    println!(
                        "{:<10} {:<15.3} {:<15} {:<15}",
                        num_nodes, cpu_time, format!("Error: {}", e), "N/A"
                    );
                }
            }
        } else {
            println!("{:<10} {:<15.3} {:<15} {:<15}", num_nodes, cpu_time, "N/A", "N/A");
        }
    }
    
    println!("\n=== Benchmark Complete ===");
    println!("Note: CUDA cleanup issues prevent multiple iterations.");
    println!("Results shown are single-run timings.");
}

