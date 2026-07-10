//! Sparse typed-PPR light propagation with path-internal refraction.

use crate::graph::{Graph, NodeId, RelationId};
use crate::relation::RelationVocab;
use crate::depth_weights::DepthWeights;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Propagation parameters.
#[derive(Debug, Clone)]
pub struct PropagationParams {
    /// Maximum number of hops.
    pub max_depth: usize,
    /// Minimum mass to keep propagating (also prunes tiny contributions).
    pub min_intensity: f32,
    /// Per-hop scoring coefficients (index = arrival depth, 0 = seed). `None` =
    /// uniform (all-ones). When `Some`, length MUST equal `max_depth + 1`.
    pub depth_weights: Option<DepthWeights>,
}

impl Default for PropagationParams {
    fn default() -> Self {
        Self { max_depth: 4, min_intensity: 1e-3, depth_weights: None }
    }
}

/// Frontier key: a node reached via an incoming relation (None on the first
/// hop of an untyped query).
type FrontierKey = (NodeId, Option<RelationId>);

/// Multi-seed diffusion. Equivalent to summing single-seed diffusions (linear).
pub fn propagate(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> HashMap<NodeId, f32> {
    seeds
        .par_iter()
        .map(|&(seed, mass)| propagate_single(graph, vocab, seed, mass, query_relation, params))
        .reduce(HashMap::new, |mut acc, m| {
            for (k, v) in m {
                *acc.entry(k).or_insert(0.0) += v;
            }
            acc
        })
}

/// Single-seed sparse diffusion. Returns total intensity per reached node
/// (including the seed itself).
pub fn propagate_single(
    graph: &Graph,
    vocab: &RelationVocab,
    seed: NodeId,
    initial_mass: f32,
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> HashMap<NodeId, f32> {
    let mut totals: HashMap<NodeId, f32> = HashMap::new();
    let n = graph.num_nodes();
    if (seed as usize) >= n || initial_mass < params.min_intensity {
        return totals;
    }

    let dw = params.depth_weights.as_ref().map(|w| w.as_slice());
    debug_assert!(
        dw.map_or(true, |c| c.len() == params.max_depth + 1),
        "depth_weights length must equal max_depth + 1"
    );
    let weight_at = |depth: usize| -> f32 { dw.map_or(1.0, |c| c[depth]) };

    *totals.entry(seed).or_insert(0.0) += weight_at(0) * initial_mass;

    let mut frontier: HashMap<FrontierKey, f32> = HashMap::new();
    frontier.insert((seed, query_relation), initial_mass);

    // Lazily cache each node's out-weight sum (keeps cost proportional to the
    // reached ball rather than the whole graph).
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();

    for depth in 0..params.max_depth {
        if frontier.is_empty() {
            break;
        }
        let mut next: HashMap<FrontierKey, f32> = HashMap::new();

        for (&(u, r_in), &mass) in frontier.iter() {
            if mass < params.min_intensity {
                continue;
            }
            let u_idx = u as usize;
            let props = graph.node_props()[u_idx];
            let refl = props.reflection;
            if refl <= 0.0 {
                continue;
            }
            let rix = props.refraction_index;

            let denom = *denom_cache.entry(u).or_insert_with(|| {
                let mut s = 0.0f32;
                for (_v, ep) in graph.neighbors(u) {
                    s += (1.0 - ep.attenuation).max(0.0);
                }
                s
            });
            if denom <= 0.0 {
                continue;
            }

            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                if base <= 0.0 {
                    continue;
                }
                let p = base / denom;
                let sim = match r_in {
                    Some(a) => vocab.similarity(a, ep.relation),
                    None => 1.0,
                };
                let sim_term = if rix == 1.0 { sim } else { sim.powf(rix) };
                let transmitted = mass * refl * p * sim_term;
                if transmitted < params.min_intensity {
                    continue;
                }
                // READOUT is weighted by arrival depth; FLOW stays raw so mass keeps
                // propagating through depths that score zero.
                *totals.entry(v).or_insert(0.0) += weight_at(depth + 1) * transmitted;
                *next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;
            }
        }
        frontier = next;
    }

    totals
}

/// Convert intensity to a "light distance": d = -log(I + eps).
pub fn intensity_to_distance(intensities: &[f32], eps: f32) -> Vec<f32> {
    intensities
        .iter()
        .map(|&i| {
            let value = i + eps;
            if value > 0.0 { -value.ln() } else { f32::INFINITY }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::relation::RelationVocab;

    fn chain() -> Graph {
        // 0 -> 1 -> 2 -> 3, all relation 0, no attenuation.
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1)], vec![e(2)], vec![e(3)], vec![]];
        Graph::from_adjacency(4, adj, NodeProps::default()).unwrap()
    }

    #[test]
    fn chain_decays_by_reflection() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // reflection 0.85, p=1, sim=1 => geometric decay.
        assert!((t[&0] - 1.0).abs() < 1e-6);
        assert!((t[&1] - 0.85).abs() < 1e-5);
        assert!((t[&2] - 0.7225).abs() < 1e-5);
        assert!((t[&3] - 0.614125).abs() < 1e-5);
    }

    #[test]
    fn refraction_penalizes_relation_turn() {
        // 0 -(relA)-> 1 -(relB)-> 2 ; query relation = A.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let adj = vec![vec![ea], vec![eb], vec![]];
        let g = Graph::from_adjacency(3, adj, NodeProps::default()).unwrap();
        // sim(A,B) = 0.5
        let vocab = RelationVocab::new(
            vec!["A".into(), "B".into()],
            vec![1.0, 0.5, 0.5, 1.0],
        ).unwrap();
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None };
        let refr = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // 1: 1*0.85*1*sim(A,A)=0.85 ; 2: 0.85*0.85*1*sim(A,B)=0.36125
        assert!((refr[&1] - 0.85).abs() < 1e-5);
        assert!((refr[&2] - 0.36125).abs() < 1e-5);

        // With uniform vocab (no refraction), node 2 gets the full 0.7225.
        let uni = RelationVocab::uniform(2);
        let plain = propagate_single(&g, &uni, 0, 1.0, Some(0), &params);
        assert!((plain[&2] - 0.7225).abs() < 1e-5);
        assert!(refr[&2] < plain[&2]);
    }

    #[test]
    fn sums_over_multiple_paths() {
        // 0 -> 1 -> 2 and 0 -> 2 (two paths to node 2).
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1), e(2)], vec![e(2)], vec![]];
        let g = Graph::from_adjacency(3, adj, NodeProps::default()).unwrap();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        // node 0 has 2 out-edges => p=0.5 each.
        // direct 0->2: 1*0.85*0.5 = 0.425
        // via 1:   (0.425) then 1->2: 0.425*0.85*1 = 0.36125
        // total node 2 = 0.425 + 0.36125 = 0.78625
        assert!((t[&2] - 0.78625).abs() < 1e-5);
    }

    #[test]
    fn multi_seed_equals_sum_of_singles() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None };
        let combined = propagate(&g, &vocab, &[(0, 1.0), (1, 1.0)], Some(0), &params);
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &params);
        let b = propagate_single(&g, &vocab, 1, 1.0, Some(0), &params);
        for node in 0..4u32 {
            let expected = a.get(&node).copied().unwrap_or(0.0)
                + b.get(&node).copied().unwrap_or(0.0);
            let got = combined.get(&node).copied().unwrap_or(0.0);
            assert!((got - expected).abs() < 1e-5, "node {node}");
        }
    }

    #[test]
    fn uniform_weights_are_bit_identical_to_none() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let none = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None };
        let uni = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::uniform(3)),
        };
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &none);
        let b = propagate_single(&g, &vocab, 0, 1.0, Some(0), &uni);
        // EXACT equality, not approximate: 1.0 * x == x.
        for node in 0..4u32 {
            assert_eq!(a.get(&node), b.get(&node), "node {node}");
        }
    }

    #[test]
    fn terminal_weights_score_only_the_arrival_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(3, 1).unwrap()),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // Only node 1 (arrives at depth 1) is scored; node 3 arrives at depth 3.
        assert!((t[&1] - 0.85).abs() < 1e-6, "node 1 = {}", t[&1]);
        assert_eq!(t.get(&3).copied().unwrap_or(0.0), 0.0, "node 3 scored 0 under terminal(1)");
    }

    #[test]
    fn flow_is_not_weighted_so_terminal_still_reaches_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(3, 3).unwrap()),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // Node 3 is only reachable by flowing through depth-1 and depth-2 nodes that
        // score ZERO. If flow were weighted, node 3 would be unreachable.
        assert!((t[&3] - 0.614125).abs() < 1e-5, "node 3 = {}", t[&3]);
        assert_eq!(t.get(&1).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn pruning_tests_raw_flow_not_weighted_score() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        // c_3 = 0.001 makes the weighted score of node 3 tiny (0.000614), but its raw
        // flow (0.614) is well above min_intensity, so it must NOT be pruned.
        let p = PropagationParams {
            max_depth: 3,
            min_intensity: 0.1,
            depth_weights: Some(
                crate::depth_weights::DepthWeights::from_vec(vec![1.0, 1.0, 1.0, 0.001], 3).unwrap(),
            ),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        assert!((t[&3] - 0.001 * 0.614125).abs() < 1e-7, "node 3 = {}", t[&3]);
    }
}
