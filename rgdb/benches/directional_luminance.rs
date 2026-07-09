use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rgdb::graph::{EdgeProps, Graph, NodeProps, NodeId};
use rgdb::propagation::{propagate_single, PropagationParams};
use rgdb::relation::RelationVocab;

fn build(num_nodes: usize) -> Graph {
    let mut adj = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let mut ns = Vec::new();
        for j in 1..=5 {
            let dst = ((i + j) % num_nodes) as NodeId;
            if dst as usize != i {
                ns.push((dst, EdgeProps {
                    attenuation: 0.1,
                    relation: ((i + j) % 4) as u16,
                    is_portal: false,
                }));
            }
        }
        adj.push(ns);
    }
    Graph::from_adjacency(num_nodes, adj, NodeProps::default()).unwrap()
}

fn bench_propagation(c: &mut Criterion) {
    let graph = build(1000);
    let vocab = RelationVocab::uniform(4);
    let params = PropagationParams::default();
    c.bench_function("propagate_single_1000", |b| {
        b.iter(|| {
            let t = propagate_single(black_box(&graph), &vocab, 0, 1.0, Some(0), &params);
            black_box(t.len());
        });
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
