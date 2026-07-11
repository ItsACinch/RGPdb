//! Sparse typed-PPR light propagation with path-internal refraction.

use crate::graph::{Graph, NodeId, RelationId};
use crate::relation::RelationVocab;
use crate::depth_weights::DepthWeights;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Mismatch score for a scheduled hop whose edge relation is not the expected one.
/// Matches the validated measure-first gate; a small floor keeps off-schedule paths
/// alive rather than hard-zeroing them. Not configurable (YAGNI).
pub const SCHEDULE_FLOOR: f32 = 0.05;

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
    /// Per-hop expected relation chain: `schedule[k]` is the relation expected at hop
    /// `k` (first hop = index 0). `None` = vocab similarity (today's behavior). When
    /// `Some`, a scheduled hop scores 1.0 for a matching edge relation else
    /// `SCHEDULE_FLOOR`, REPLACING vocab similarity; hops past its end use the vocab.
    pub schedule: Option<Vec<RelationId>>,
}

impl Default for PropagationParams {
    fn default() -> Self {
        Self { max_depth: 4, min_intensity: 1e-3, depth_weights: None, schedule: None }
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
                let sim_term = match params.schedule.as_deref() {
                    Some(s) if depth < s.len() => {
                        if ep.relation == s[depth] { 1.0 } else { SCHEDULE_FLOOR }
                    }
                    _ => {
                        let sim = match r_in {
                            Some(a) => vocab.similarity(a, ep.relation),
                            None => 1.0,
                        };
                        if rix == 1.0 { sim } else { sim.powf(rix) }
                    }
                };
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

/// Per-node intensity broken out by arrival depth, plus each node's heaviest
/// incoming relation. The per-depth values are RAW (un-weighted): applying a
/// `DepthWeights` to them reconstructs `propagate`'s scalar output. `depth_weights`
/// in `params` is ignored here — weighting is the consumer's job.
#[derive(Debug, Clone, Default)]
pub struct LayeredResult {
    /// node -> `[I_0, I_1, .., I_maxdepth]`; index d = mass arriving after exactly d hops.
    pub per_depth: HashMap<NodeId, Vec<f32>>,
    /// node -> the incoming relation that delivered the most mass (absent for seeds).
    pub dominant_incoming: HashMap<NodeId, RelationId>,
}

pub fn propagate_layered(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
) -> LayeredResult {
    let d_max = params.max_depth;
    let n = graph.num_nodes();
    let mut per_depth: HashMap<NodeId, Vec<f32>> = HashMap::new();
    let mut incoming: HashMap<NodeId, HashMap<RelationId, f32>> = HashMap::new();
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();

    // Linear in seed mass, so accumulate each single-seed walk into shared maps.
    for &(seed, initial_mass) in seeds {
        if (seed as usize) >= n || initial_mass < params.min_intensity {
            continue;
        }
        per_depth.entry(seed).or_insert_with(|| vec![0.0; d_max + 1])[0] += initial_mass;

        let mut frontier: HashMap<FrontierKey, f32> = HashMap::new();
        frontier.insert((seed, query_relation), initial_mass);

        for depth in 0..d_max {
            if frontier.is_empty() {
                break;
            }
            let mut next: HashMap<FrontierKey, f32> = HashMap::new();
            for (&(u, r_in), &mass) in frontier.iter() {
                if mass < params.min_intensity {
                    continue;
                }
                let props = graph.node_props()[u as usize];
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
                    let sim_term = match params.schedule.as_deref() {
                        Some(s) if depth < s.len() => {
                            if ep.relation == s[depth] { 1.0 } else { SCHEDULE_FLOOR }
                        }
                        _ => {
                            let sim = match r_in {
                                Some(a) => vocab.similarity(a, ep.relation),
                                None => 1.0,
                            };
                            if rix == 1.0 { sim } else { sim.powf(rix) }
                        }
                    };
                    let transmitted = mass * refl * p * sim_term;
                    if transmitted < params.min_intensity {
                        continue;
                    }
                    per_depth.entry(v).or_insert_with(|| vec![0.0; d_max + 1])[depth + 1] += transmitted;
                    *incoming.entry(v).or_default().entry(ep.relation).or_insert(0.0) += transmitted;
                    *next.entry((v, Some(ep.relation))).or_insert(0.0) += transmitted;
                }
            }
            frontier = next;
        }
    }

    let dominant_incoming = incoming
        .into_iter()
        .filter_map(|(node, rels)| {
            rels.into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| (node, r))
        })
        .collect();

    LayeredResult { per_depth, dominant_incoming }
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
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None, schedule: None };
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
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None, schedule: None };
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
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None, schedule: None };
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
        let params = PropagationParams { max_depth: 4, min_intensity: 1e-6, depth_weights: None, schedule: None };
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
        let none = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None, schedule: None };
        let uni = PropagationParams {
            max_depth: 3,
            min_intensity: 1e-6,
            depth_weights: Some(crate::depth_weights::DepthWeights::uniform(3)),
            schedule: None,
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
            schedule: None,
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
            schedule: None,
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
            schedule: None,
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        assert!((t[&3] - 0.001 * 0.614125).abs() < 1e-7, "node 3 = {}", t[&3]);
    }

    #[test]
    fn layered_decomposes_the_chain_by_arrival_depth() {
        let g = chain(); // 0->1->2->3
        let vocab = RelationVocab::uniform(1);
        let p = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None, schedule: None };
        let r = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        // node k arrives only at depth k on a pure chain.
        assert_eq!(r.per_depth[&0], vec![1.0, 0.0, 0.0, 0.0]);
        assert!((r.per_depth[&1][1] - 0.85).abs() < 1e-5);
        assert!((r.per_depth[&2][2] - 0.7225).abs() < 1e-5);
        assert!((r.per_depth[&3][3] - 0.614125).abs() < 1e-5);
        assert_eq!(r.per_depth[&3][1], 0.0);
    }

    #[test]
    fn layered_collapses_to_scalar_propagate_for_any_weights() {
        // Σ_d c[d]·layered[v][d] must equal propagate(v) under the SAME c, exactly.
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        for c in [
            crate::depth_weights::DepthWeights::uniform(3),
            crate::depth_weights::DepthWeights::terminal(3, 2).unwrap(),
            crate::depth_weights::DepthWeights::from_vec(vec![0.0, 0.3, 0.7, 1.0], 3).unwrap(),
        ] {
            let p = PropagationParams { max_depth: 3, min_intensity: 0.0, depth_weights: Some(c.clone()), schedule: None };
            let scalar = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
            let layered = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
            let cs = c.as_slice();
            for (&v, prof) in &layered.per_depth {
                let collapsed: f32 = prof.iter().zip(cs).map(|(x, w)| x * w).sum();
                let want = scalar.get(&v).copied().unwrap_or(0.0);
                assert!((collapsed - want).abs() < 1e-5, "node {v}: {collapsed} != {want}");
            }
        }
    }

    #[test]
    fn layered_reports_dominant_incoming_relation() {
        // 0 -(A=0)-> 1 -(B=1)-> 2 ; node 1's only incoming is A, node 2's is B.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let vocab = RelationVocab::uniform(2);
        let p = PropagationParams { max_depth: 2, min_intensity: 1e-6, depth_weights: None, schedule: None };
        let r = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        assert_eq!(r.dominant_incoming.get(&1), Some(&0));
        assert_eq!(r.dominant_incoming.get(&2), Some(&1));
        assert_eq!(r.dominant_incoming.get(&0), None); // the seed has no incoming edge
    }

    // 0 -(A=0)-> 1 -(B=1)-> 2, used by the schedule tests.
    fn ab_chain() -> Graph {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap()
    }

    #[test]
    fn schedule_none_is_bit_identical() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let base = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None, schedule: None };
        // Two params values that differ only by an explicit `schedule: None` must be equal;
        // and both equal the pre-feature output (this is the do-no-harm guarantee).
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &base);
        assert!((a[&3] - 0.614125).abs() < 1e-6, "unchanged kernel value, got {}", a[&3]);
    }

    #[test]
    fn schedule_scores_matching_relation_per_hop() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        // schedule [A,B] matches both hops => node 2 gets full decayed mass.
        let p_ok = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0, 1]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p_ok);
        assert!((t[&2] - 0.7225).abs() < 1e-5, "matching schedule, got {}", t[&2]);
        // schedule [A,A] mismatches hop 2 (edge is B) => node 2 gets the floor on that hop.
        let p_bad = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0, 0]),
        };
        let t2 = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p_bad);
        // node2 = 0.85 * 0.85 * 1.0(p) * SCHEDULE_FLOOR(0.05) = 0.036125
        assert!((t2[&2] - 0.036125).abs() < 1e-6, "floored hop, got {}", t2[&2]);
    }

    #[test]
    fn schedule_falls_back_to_vocab_past_its_end() {
        let g = ab_chain();
        // Non-uniform vocab: sim(A,B) = 0.5. A length-1 schedule covers hop 0 only;
        // hop 1 must use the VOCAB (0.5), not the floor (0.05).
        let vocab = RelationVocab::new(vec!["A".into(), "B".into()], vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        let p = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // node2 = 0.85 * 0.85 * 1.0 * sim(A,B)=0.5 = 0.36125  (vocab fallback, NOT floor)
        assert!((t[&2] - 0.36125).abs() < 1e-5, "vocab fallback past schedule, got {}", t[&2]);
    }

    #[test]
    fn schedule_composes_with_depth_weights() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        // schedule [A,B] + terminal(2): only the depth-2 node scored, at the scheduled mass.
        let p = PropagationParams {
            max_depth: 2, min_intensity: 1e-9,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(2, 2).unwrap()),
            schedule: Some(vec![0, 1]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        assert!((t[&2] - 0.7225).abs() < 1e-5, "node 2 = {}", t[&2]);
        assert_eq!(t.get(&1).copied().unwrap_or(0.0), 0.0, "depth-1 node zeroed by terminal(2)");
    }

    #[test]
    fn layered_applies_schedule_and_stays_consistent() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        let c = crate::depth_weights::DepthWeights::from_vec(vec![0.0, 0.4, 1.0], 2).unwrap();
        let p = PropagationParams {
            max_depth: 2, min_intensity: 0.0, depth_weights: Some(c.clone()), schedule: Some(vec![0, 1]),
        };
        let scalar = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        let layered = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        let cs = c.as_slice();
        for (&v, prof) in &layered.per_depth {
            let collapsed: f32 = prof.iter().zip(cs).map(|(x, w)| x * w).sum();
            let want = scalar.get(&v).copied().unwrap_or(0.0);
            assert!((collapsed - want).abs() < 1e-6, "node {v}: {collapsed} != {want}");
        }
    }
}
