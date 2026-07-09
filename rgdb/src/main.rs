use rgdb::graph::{EdgeProps, Graph, NodeProps};
use rgdb::propagation::{propagate_single, PropagationParams};
use rgdb::relation::RelationVocab;

fn main() {
    // 0 -(causes)-> 1 -(causes)-> 2, with a relation-similarity matrix.
    let causes = 0u16;
    let edge = |dst, rel| (dst, EdgeProps { attenuation: 0.1, relation: rel, is_portal: false });
    let adj = vec![vec![edge(1, causes)], vec![edge(2, causes)], vec![]];
    let graph = Graph::from_adjacency(3, adj, NodeProps::default()).expect("graph");

    let vocab = RelationVocab::with_names_uniform(vec!["causes".into(), "isa".into()]);
    let params = PropagationParams::default();

    let totals = propagate_single(&graph, &vocab, 0, 1.0, Some(causes), &params);
    println!("Influence from node 0 (relation=causes):");
    for node in 0..graph.num_nodes() as u32 {
        println!("  node {node}: {:.6}", totals.get(&node).copied().unwrap_or(0.0));
    }
}
