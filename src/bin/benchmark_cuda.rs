/// Performance benchmark comparing CPU vs CUDA propagation

use rgdb::*;
use rgdb::property_map::{RelationshipProperty, create_directional_luminance};
use std::time::Instant;

fn create_large_test_graph(num_nodes: usize) -> graph::Graph {
    let mut adj = Vec::new();
    
    // Create a graph with nodes connected in a chain with branches
    // This creates a realistic graph structure
    for i in 0..num_nodes {
        let mut neighbors = Vec::new();
        
        // Each node connects to next 3-8 nodes (creating a dense graph)
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
    
    // Create node properties with directional luminance
    let node_props = graph::NodeProps {
        relationship_property: Some(RelationshipProperty::IsA),
        directional_luminance: create_directional_luminance(RelationshipProperty::IsA, 2.0, 0.8),
        ..graph::NodeProps::default()
    };
    
    graph::Graph::from_adjacency(num_nodes, adj, node_props)
        .expect("Failed to create graph")
}

fn benchmark_cpu(graph: &graph::Graph, source: graph::NodeId, iterations: usize) -> (f64, Vec<f32>) {
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: graph::N_ANGLE_BINS,
    };
    
    let start = Instant::now();
    let mut last_result = Vec::new();
    
    for _ in 0..iterations {
        last_result = propagation::propagate_light(graph, source, 2, params);
    }
    
    let duration = start.elapsed();
    let avg_time_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
    
    (avg_time_ms, last_result)
}

fn benchmark_cuda(
    ctx: &mut cuda::CudaContext,
    graph: &graph::Graph,
    source: graph::NodeId,
    iterations: usize,
) -> Result<(f64, Vec<f32>), cuda::CudaError> {
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: graph::N_ANGLE_BINS,
    };
    
    let start = Instant::now();
    let mut last_result = Vec::new();
    
    // CRITICAL: Ensure context stays alive for all iterations
    // Each iteration creates and cleans up CUDA resources, so we need
    // the context to remain active throughout
    for _ in 0..iterations {
        // Each call creates streams and buffers that are cleaned up when it returns
        // The context must remain active during this cleanup
        last_result = cuda::propagate_light_cuda(ctx, graph, source, 2, params, None)?;
    }
    
    let duration = start.elapsed();
    let avg_time_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
    
    Ok((avg_time_ms, last_result))
}

fn verify_results_match(cpu_result: &[f32], cuda_result: &[f32], tolerance: f32) -> bool {
    if cpu_result.len() != cuda_result.len() {
        return false;
    }
    
    for (cpu_val, cuda_val) in cpu_result.iter().zip(cuda_result.iter()) {
        let diff = (cpu_val - cuda_val).abs();
        if diff > tolerance {
            return false;
        }
    }
    
    true
}

fn main() {
    println!("=== CUDA vs CPU Performance Benchmark ===\n");
    
    // Check CUDA availability
    let cuda_available = cuda::CudaContext::is_available();
    println!("CUDA Available: {}", cuda_available);
    
    // Create CUDA context once and reuse it (mutable for graph caching)
    let mut cuda_ctx = if cuda_available {
        match cuda::CudaContext::new() {
            Ok(ctx) => {
                if ctx.kernels_loaded() {
                    match ctx.device_name() {
                        Ok(name) => println!("CUDA Device: {}\n", name),
                        Err(_) => println!("CUDA Device: Unknown\n"),
                    }
                    Some(ctx)
                } else {
                    println!("CUDA kernels not loaded - using CPU fallback\n");
                    None
                }
            }
            Err(e) => {
                println!("Failed to initialize CUDA: {}\n", e);
                None
            }
        }
    } else {
        println!("CUDA not available - CPU only benchmark\n");
        None
    };
    
    // Test different graph sizes - focus on finding crossover point
    // Start with small sizes, then test larger ones to find where CUDA becomes faster
    let test_sizes = vec![100, 500, 1000, 5000, 10000, 25000, 50000];
    let iterations = 50;  // More iterations for better accuracy
    let warmup_iterations = 5;  // More warmup to ensure stable timings
    
    println!("Running benchmarks ({} iterations per test, {} warmup):\n", iterations, warmup_iterations);
    println!("{:<10} {:<15} {:<15} {:<15} {:<10}", "Nodes", "CPU (ms)", "CUDA (ms)", "Speedup", "Match");
    println!("{}", "-".repeat(70));
    
    for &num_nodes in &test_sizes {
        println!("Creating graph with {} nodes...", num_nodes);
        let graph = create_large_test_graph(num_nodes);
        let source: graph::NodeId = 0;
        
        // Warmup
        for _ in 0..warmup_iterations {
            let _ = benchmark_cpu(&graph, source, 1);
        }
        
        // Benchmark CPU
        let (cpu_time, cpu_result) = benchmark_cpu(&graph, source, iterations);
        
        // Benchmark CUDA if available
        let (cuda_time, cuda_result, results_match) = if let Some(ref mut ctx) = cuda_ctx {
            // Warmup
            for _ in 0..warmup_iterations {
                let _ = benchmark_cuda(ctx, &graph, source, 1).ok();
            }
            
            match benchmark_cuda(ctx, &graph, source, iterations) {
                Ok((cuda_time, cuda_result)) => {
                    let match_result = verify_results_match(&cpu_result, &cuda_result, 0.01);
                    (Some(cuda_time), Some(cuda_result), match_result)
                }
                Err(e) => {
                    eprintln!("CUDA error (using CPU fallback): {}", e);
                    (None, None, false)
                }
            }
        } else {
            (None, None, false)
        };
        
        // Print results
        if let Some(cuda_time) = cuda_time {
            let speedup = cpu_time / cuda_time;
            let match_str = if results_match { "✓" } else { "✗" };
            println!(
                "{:<10} {:<15.3} {:<15.3} {:<15.2}x {:<10}",
                num_nodes, cpu_time, cuda_time, speedup, match_str
            );
        } else {
            println!(
                "{:<10} {:<15.3} {:<15} {:<15} {:<10}",
                num_nodes, cpu_time, "N/A", "N/A", "-"
            );
        }
    }
    
    println!("\n=== Detailed Analysis (Finding Crossover Point) ===\n");
    
    // Run detailed benchmark on each test size to find crossover point
    let mut crossover_found = false;
    for &num_nodes in &test_sizes {
        println!("Detailed benchmark: {} nodes, {} iterations\n", num_nodes, iterations);
    
        let graph = create_large_test_graph(num_nodes);
        let source: graph::NodeId = 0;
        
        // CPU benchmark
        println!("CPU Benchmark:");
        let (cpu_time, cpu_result) = benchmark_cpu(&graph, source, iterations);
        println!("  Average time: {:.3} ms", cpu_time);
        println!("  Total time: {:.3} ms", cpu_time * iterations as f64);
        println!("  Nodes with intensity > 0: {}", cpu_result.iter().filter(|&&x| x > 0.0).count());
        println!("  Max intensity: {:.6}", cpu_result.iter().copied().fold(0.0, f32::max));
        
    // CUDA benchmark
    if let Some(ref mut ctx) = cuda_ctx {
            println!("\nCUDA Benchmark:");
            match benchmark_cuda(ctx, &graph, source, iterations) {
                Ok((cuda_time, cuda_result)) => {
                    println!("  Average time: {:.3} ms", cuda_time);
                    println!("  Total time: {:.3} ms", cuda_time * iterations as f64);
                    println!("  Nodes with intensity > 0: {}", cuda_result.iter().filter(|&&x| x > 0.0).count());
                    println!("  Max intensity: {:.6}", cuda_result.iter().copied().fold(0.0, f32::max));
                    
                    let speedup = cpu_time / cuda_time;
                    println!("\nPerformance Comparison:");
                    if speedup > 1.0 {
                        println!("  ✅ CUDA is {:.2}x FASTER (speedup: {:.2}x)", speedup, speedup);
                        if !crossover_found {
                            println!("  🎯 CROSSOVER POINT FOUND at {} nodes!", num_nodes);
                            crossover_found = true;
                        }
                    } else {
                        println!("  ⚠️  CUDA is {:.2}x SLOWER (CPU is {:.2}x faster)", 1.0 / speedup, 1.0 / speedup);
                    }
                    println!("  CPU time: {:.3} ms", cpu_time);
                    println!("  CUDA time: {:.3} ms", cuda_time);
                    println!("  Time difference: {:.3} ms", (cpu_time - cuda_time).abs());
                    
                    // Verify results match
                    if verify_results_match(&cpu_result, &cuda_result, 0.01) {
                        println!("  Results match: ✓ (within 0.01 tolerance)");
                    } else {
                        println!("  Results match: ✗ (differences detected)");
                        // Find max difference
                        let max_diff = cpu_result.iter()
                            .zip(cuda_result.iter())
                            .map(|(c, g)| (c - g).abs())
                            .fold(0.0, f32::max);
                        println!("  Max difference: {:.6}", max_diff);
                    }
                }
                Err(e) => {
                    println!("  CUDA error (falling back to CPU): {}", e);
                }
            }
        } else {
            println!("\nCUDA not available - CPU only");
        }
        
        println!("\n{}", "-".repeat(70));
        println!();
    }
    
    println!("\n=== Benchmark Complete ===");
}

