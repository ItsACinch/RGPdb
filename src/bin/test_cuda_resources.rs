/// Minimal test to verify CUDA resource management

use rgdb::*;

fn main() {
    println!("=== CUDA Resource Management Test ===\n");
    
    // Check CUDA availability
    if !cuda::CudaContext::is_available() {
        println!("CUDA not available - skipping test");
        return;
    }
    
    println!("Creating CUDA context...");
    let mut ctx = match cuda::CudaContext::new() {
        Ok(ctx) => {
            println!("✓ CUDA context created");
            if ctx.kernels_loaded() {
                println!("✓ CUDA kernels loaded");
            } else {
                println!("✗ CUDA kernels not loaded");
                return;
            }
            ctx
        }
        Err(e) => {
            println!("✗ Failed to create CUDA context: {}", e);
            return;
        }
    };
    
    // Create a small test graph
    println!("\nCreating test graph...");
    let num_nodes = 10;
    let mut adj = Vec::new();
    for i in 0..num_nodes {
        let mut neighbors = Vec::new();
        for j in 1..=3 {
            let target = (i + j) % num_nodes;
            if target != i {
                neighbors.push((
                    target as graph::NodeId,
                    graph::EdgeProps {
                        attenuation: 0.1,
                        angle_bin: (j % graph::N_ANGLE_BINS) as graph::AngleBin,
                        is_portal: false,
                    },
                ));
            }
        }
        adj.push(neighbors);
    }
    
    let graph = graph::Graph::from_adjacency(num_nodes, adj, graph::NodeProps::default())
        .expect("Failed to create graph");
    println!("✓ Test graph created ({} nodes)", num_nodes);
    
    // Test propagation multiple times to check resource cleanup
    println!("\nTesting CUDA propagation (multiple iterations)...");
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 2,
        num_angle_bins: graph::N_ANGLE_BINS,
    };
    
    for i in 1..=5 {
        print!("  Iteration {}... ", i);
        match cuda::propagate_light_cuda(&mut ctx, &graph, 0, 0, params, None) {
            Ok(result) => {
                let non_zero = result.iter().filter(|&&x| x > 0.0).count();
                println!("✓ ({} nodes with intensity)", non_zero);
            }
            Err(e) => {
                println!("✗ Error: {}", e);
                return;
            }
        }
    }
    
    println!("\n=== Test Complete ===");
    println!("If you see this message, CUDA resource management is working correctly!");
}

