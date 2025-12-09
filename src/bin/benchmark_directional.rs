/// Benchmark and verification script for directional luminance

use rgdb::*;
use rgdb::property_map::{RelationshipProperty, create_directional_luminance};
use std::time::Instant;

fn create_test_graph(num_nodes: usize, use_directional: bool) -> graph::Graph {
    let mut adj = Vec::new();
    
    // Create a graph with nodes connected in a chain with some branches
    for i in 0..num_nodes {
        let mut neighbors = Vec::new();
        
        // Each node connects to next 3-5 nodes (creating a dense graph)
        for j in 1..=5 {
            let target = (i + j) % num_nodes;
            if target != i {
                neighbors.push((
                    target as graph::NodeId,
                    graph::EdgeProps {
                        attenuation: 0.1 + (j as f32) * 0.05,
                        angle_bin: ((i + j) % graph::N_ANGLE_BINS) as graph::AngleBin,
                        is_portal: false,
                    },
                ));
            }
        }
        
        adj.push(neighbors);
    }
    
    // Create node properties with different relationship properties
    let mut node_props_vec = Vec::new();
    for i in 0..num_nodes {
        let props = if use_directional {
            // Use directional luminance with different properties
            let property = match i % 10 {
                0..=2 => RelationshipProperty::IsA,
                3..=4 => RelationshipProperty::RelatedTo,
                5..=6 => RelationshipProperty::Causes,
                7..=8 => RelationshipProperty::Contains,
                _ => RelationshipProperty::SimilarTo,
            };
            graph::NodeProps::from_directional_luminance(
                property,
                create_directional_luminance(property, 2.0, 0.8),
            )
        } else {
            // Use uniform luminance
            graph::NodeProps::from_uniform_luminance(2.0)
        };
        node_props_vec.push(props);
    }
    
    // Build graph from adjacency
    let mut graph = graph::Graph::from_adjacency(num_nodes, adj, node_props_vec[0])
        .expect("Failed to create graph");
    
    // Set individual node properties
    for (i, props) in node_props_vec.iter().enumerate() {
        graph.set_node_props(i as graph::NodeId, *props)
            .expect("Failed to set node props");
    }
    
    graph
}

fn run_benchmark(num_nodes: usize, use_directional: bool, name: &str) -> (f64, Vec<f32>) {
    let graph = create_test_graph(num_nodes, use_directional);
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: graph::N_ANGLE_BINS,
    };
    
    let start = Instant::now();
    let intensities = propagation::propagate_light(&graph, 0, 2, params);
    let duration = start.elapsed();
    
    let elapsed_ms = duration.as_secs_f64() * 1000.0;
    println!("{}: {:.3} ms", name, elapsed_ms);
    
    (elapsed_ms, intensities)
}

fn verify_functionality() {
    println!("=== Verifying Directional Luminance Functionality ===\n");
    
    // Create a small test graph
    let num_nodes = 10;
    let mut adj = Vec::new();
    
    // Simple chain: 0 -> 1 -> 2 -> 3
    for i in 0..num_nodes {
        let mut neighbors = Vec::new();
        if i < num_nodes - 1 {
            neighbors.push((
                (i + 1) as graph::NodeId,
                graph::EdgeProps {
                    attenuation: 0.1,
                    angle_bin: 2,
                    is_portal: false,
                },
            ));
        }
        adj.push(neighbors);
    }
    
    // Create graph with directional luminance
    let mut graph = graph::Graph::from_adjacency(
        num_nodes,
        adj,
        graph::NodeProps::from_directional_luminance(
            RelationshipProperty::IsA,
            create_directional_luminance(RelationshipProperty::IsA, 2.0, 0.8),
        ),
    ).expect("Failed to create graph");
    
    // Set node 0 with IsA property
    graph.set_node_props(0, graph::NodeProps::from_directional_luminance(
        RelationshipProperty::IsA,
        create_directional_luminance(RelationshipProperty::IsA, 2.0, 0.8),
    )).unwrap();
    
    // Set node 1 with RelatedTo property (compatible)
    graph.set_node_props(1, graph::NodeProps::from_directional_luminance(
        RelationshipProperty::RelatedTo,
        create_directional_luminance(RelationshipProperty::RelatedTo, 1.5, 0.8),
    )).unwrap();
    
    // Set node 2 with OppositeOf property (incompatible)
    graph.set_node_props(2, graph::NodeProps::from_directional_luminance(
        RelationshipProperty::OppositeOf,
        create_directional_luminance(RelationshipProperty::OppositeOf, 1.5, 0.8),
    )).unwrap();
    
    let params = propagation::LightParams::default();
    let intensities = propagation::propagate_light(&graph, 0, 2, params);
    
    println!("Intensities after propagation:");
    for (i, &intensity) in intensities.iter().enumerate() {
        println!("  Node {}: {:.6}", i, intensity);
    }
    
    // Verify that node 1 (compatible) receives more intensity than node 2 (incompatible)
    if intensities.len() > 2 {
        let compatible_intensity = intensities[1];
        let incompatible_intensity = intensities[2];
        
        println!("\nVerification:");
        println!("  Node 1 (RelatedTo, compatible): {:.6}", compatible_intensity);
        println!("  Node 2 (OppositeOf, incompatible): {:.6}", incompatible_intensity);
        
        if compatible_intensity > incompatible_intensity {
            println!("  ✓ PASS: Compatible node receives more intensity");
        } else {
            println!("  ✗ FAIL: Incompatible node should receive less intensity");
        }
    }
    
    println!("\n=== Functionality Verification Complete ===\n");
}

fn main() {
    verify_functionality();
    
    println!("=== Performance Benchmark (1000 nodes) ===\n");
    
    let num_nodes = 1000;
    let iterations = 10;
    
    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let _ = run_benchmark(num_nodes, false, "warmup");
    }
    
    println!("\nRunning benchmarks ({} iterations each):\n", iterations);
    
    // Benchmark uniform (old approach)
    let mut uniform_times = Vec::new();
    for i in 0..iterations {
        let (time, _) = run_benchmark(num_nodes, false, &format!("Uniform #{}", i + 1));
        uniform_times.push(time);
    }
    
    // Benchmark directional (new approach)
    let mut directional_times = Vec::new();
    for i in 0..iterations {
        let (time, _) = run_benchmark(num_nodes, true, &format!("Directional #{}", i + 1));
        directional_times.push(time);
    }
    
    // Calculate statistics
    let uniform_avg = uniform_times.iter().sum::<f64>() / uniform_times.len() as f64;
    let directional_avg = directional_times.iter().sum::<f64>() / directional_times.len() as f64;
    
    let uniform_min = uniform_times.iter().copied().fold(f64::INFINITY, f64::min);
    let uniform_max = uniform_times.iter().copied().fold(0.0, f64::max);
    
    let directional_min = directional_times.iter().copied().fold(f64::INFINITY, f64::min);
    let directional_max = directional_times.iter().copied().fold(0.0, f64::max);
    
    println!("\n=== Results ===");
    println!("Uniform Luminance (old):");
    println!("  Average: {:.3} ms", uniform_avg);
    println!("  Min: {:.3} ms", uniform_min);
    println!("  Max: {:.3} ms", uniform_max);
    
    println!("\nDirectional Luminance (new):");
    println!("  Average: {:.3} ms", directional_avg);
    println!("  Min: {:.3} ms", directional_min);
    println!("  Max: {:.3} ms", directional_max);
    
    let speedup = uniform_avg / directional_avg;
    println!("\nSpeedup: {:.2}x", speedup);
    
    if speedup > 1.0 {
        println!("  ✓ Directional luminance is {:.1}% faster", (speedup - 1.0) * 100.0);
    } else {
        println!("  ⚠ Directional luminance is {:.1}% slower", (1.0 - speedup) * 100.0);
    }
}

