use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rgdb::*;
use rgdb::property_map::{RelationshipProperty, create_directional_luminance, create_uniform_luminance};

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
    
    // Create node properties
    let default_props = if use_directional {
        // Use directional luminance with different properties
        let property = match num_nodes % 10 {
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
    
    graph::Graph::from_adjacency(num_nodes, adj, default_props)
        .expect("Failed to create graph")
}

fn benchmark_propagation(c: &mut Criterion) {
    let num_nodes = 1000;
    
    // Benchmark uniform luminance (old approach)
    let graph_uniform = create_test_graph(num_nodes, false);
    c.bench_function("propagation_uniform_1000", |b| {
        b.iter(|| {
            let params = propagation::LightParams::default();
            let intensities = propagation::propagate_light(
                black_box(&graph_uniform),
                black_box(0),
                black_box(2),
                params,
            );
            black_box(intensities)
        })
    });
    
    // Benchmark directional luminance (new approach)
    let graph_directional = create_test_graph(num_nodes, true);
    c.bench_function("propagation_directional_1000", |b| {
        b.iter(|| {
            let params = propagation::LightParams::default();
            let intensities = propagation::propagate_light(
                black_box(&graph_directional),
                black_box(0),
                black_box(2),
                params,
            );
            black_box(intensities)
        })
    });
}

criterion_group!(benches, benchmark_propagation);
criterion_main!(benches);

