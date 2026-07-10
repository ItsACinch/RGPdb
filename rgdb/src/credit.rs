//! Backward credit pass: attribute a rewarded answer to relation transitions.

use hashbrown::{HashMap, HashSet};

use crate::graph::{Graph, NodeId, RelationId};
use crate::propagation::PropagationParams;
use crate::relation::RelationVocab;

/// A propagation state: a node, plus the relation of the edge we arrived by.
/// `None` only on the seed of an untyped query.
type State = (NodeId, Option<RelationId>);

/// Σ over `u`'s out-edges of `(1 - attenuation)`, memoized.
fn out_weight_sum(graph: &Graph, u: NodeId, cache: &mut HashMap<NodeId, f32>) -> f32 {
    if let Some(&d) = cache.get(&u) {
        return d;
    }
    let mut s = 0.0f32;
    for (_v, ep) in graph.neighbors(u) {
        s += (1.0 - ep.attenuation).max(0.0);
    }
    cache.insert(u, s);
    s
}

/// Weight of the hop `(u, r_in) --[r_out]--> v`, identical to the forward kernel's.
fn hop_weight(
    graph: &Graph,
    vocab: &RelationVocab,
    u: NodeId,
    r_in: Option<RelationId>,
    r_out: RelationId,
    base: f32,
    denom: f32,
) -> f32 {
    if base <= 0.0 || denom <= 0.0 {
        return 0.0;
    }
    let props = graph.node_props()[u as usize];
    if props.reflection <= 0.0 {
        return 0.0;
    }
    let p = base / denom;
    let sim = match r_in {
        Some(a) => vocab.similarity(a, r_out),
        None => 1.0,
    };
    let sim_term = if props.refraction_index == 1.0 { sim } else { sim.powf(props.refraction_index) };
    props.reflection * p * sim_term
}

/// `F[k]` = mass at each state after exactly `k` hops. Mirrors the forward kernel.
fn forward(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    params: &PropagationParams,
    denom_cache: &mut HashMap<NodeId, f32>,
) -> Vec<HashMap<State, f32>> {
    let d = params.max_depth;
    let n = graph.num_nodes();
    let mut f: Vec<HashMap<State, f32>> = vec![HashMap::new(); d + 1];

    for &(s, m) in seeds {
        if (s as usize) < n && m >= params.min_intensity && m > 0.0 {
            *f[0].entry((s, query_relation)).or_insert(0.0) += m;
        }
    }

    for k in 0..d {
        let current: Vec<(State, f32)> = f[k].iter().map(|(&s, &m)| (s, m)).collect();
        for ((u, r_in), mass) in current {
            if mass <= 0.0 || mass < params.min_intensity {
                continue;
            }
            let denom = out_weight_sum(graph, u, denom_cache);
            if denom <= 0.0 {
                continue;
            }
            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                let w = hop_weight(graph, vocab, u, r_in, ep.relation, base, denom);
                let t = mass * w;
                if t <= 0.0 || t < params.min_intensity {
                    continue;
                }
                *f[k + 1].entry((v, Some(ep.relation))).or_insert(0.0) += t;
            }
        }
    }
    f
}

/// `B[j][(v,r)]` = total weight of continuations from `(v,r)` that arrive at `target`
/// within `j` more hops. Backward in *depth budget*, forward in *graph direction* —
/// so this needs only out-adjacency, never a reverse index.
fn backward(
    graph: &Graph,
    vocab: &RelationVocab,
    target: NodeId,
    ball: &[State],
    params: &PropagationParams,
    denom_cache: &mut HashMap<NodeId, f32>,
) -> Vec<HashMap<State, f32>> {
    let d = params.max_depth;
    let mut b: Vec<HashMap<State, f32>> = vec![HashMap::new(); d + 1];

    for &st in ball {
        b[0].insert(st, if st.0 == target { 1.0 } else { 0.0 });
    }

    for j in 1..=d {
        let mut cur: HashMap<State, f32> = HashMap::with_capacity(ball.len());
        for &(v, r) in ball {
            let mut acc = if v == target { 1.0 } else { 0.0 };
            let denom = out_weight_sum(graph, v, denom_cache);
            if denom > 0.0 {
                for (x, ep) in graph.neighbors(v) {
                    let base = (1.0 - ep.attenuation).max(0.0);
                    let w = hop_weight(graph, vocab, v, r, ep.relation, base, denom);
                    if w <= 0.0 {
                        continue;
                    }
                    if let Some(&bx) = b[j - 1].get(&(x, Some(ep.relation))) {
                        acc += w * bx;
                    }
                }
            }
            cur.insert((v, r), acc);
        }
        b[j] = cur;
    }
    b
}

fn ball_of(f: &[HashMap<State, f32>]) -> Vec<State> {
    let mut set: HashSet<State> = HashSet::new();
    for m in f {
        for k in m.keys() {
            set.insert(*k);
        }
    }
    set.into_iter().collect()
}

/// L1-normalized credit per `(r_in, r_out)` transition for a rewarded `target`.
/// Empty when the target is unreachable within `max_depth`.
///
/// # Pruning bias (production)
///
/// The forward pass prunes states below `params.min_intensity`; the backward DP
/// does not. With pruning on (the production default), transitions that the
/// current matrix penalizes carry less mass, decay toward the threshold sooner,
/// and are therefore more likely to be dropped from the ball entirely — so they
/// receive systematically *less* credit than their true unpruned share. Credit is
/// flow-proportional by design, so some of this is intended; the pruning amplifies
/// it, and it can be self-reinforcing (a penalized transition earns less evidence,
/// so it stays penalized).
///
/// The do-no-harm uniform cold start mitigates this: when learning begins every
/// similarity is 1.0, so there is no differential pruning at the point the first
/// evidence is gathered. Set `min_intensity = 0.0` for an exact, unbiased credit.
pub fn credit(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    target: NodeId,
    params: &PropagationParams,
) -> Vec<(RelationId, RelationId, f32)> {
    let d = params.max_depth;
    if (target as usize) >= graph.num_nodes() || d == 0 {
        return Vec::new();
    }

    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();
    let f = forward(graph, vocab, seeds, query_relation, params, &mut denom_cache);
    let ball = ball_of(&f);
    let b = backward(graph, vocab, target, &ball, params, &mut denom_cache);

    let mut flow: HashMap<(RelationId, RelationId), f32> = HashMap::new();
    for k in 0..d {
        let budget = d - k - 1;
        let states: Vec<(State, f32)> = f[k].iter().map(|(&s, &m)| (s, m)).collect();
        for ((u, r_in), mass) in states {
            if mass <= 0.0 {
                continue;
            }
            // An untyped first hop has no source relation, so it credits nothing.
            let r_from = match r_in {
                Some(a) => a,
                None => continue,
            };
            let denom = out_weight_sum(graph, u, &mut denom_cache);
            if denom <= 0.0 {
                continue;
            }
            for (v, ep) in graph.neighbors(u) {
                let base = (1.0 - ep.attenuation).max(0.0);
                let w = hop_weight(graph, vocab, u, r_in, ep.relation, base, denom);
                if w <= 0.0 {
                    continue;
                }
                let bv = b[budget].get(&(v, Some(ep.relation))).copied().unwrap_or(0.0);
                if bv <= 0.0 {
                    continue;
                }
                *flow.entry((r_from, ep.relation)).or_insert(0.0) += mass * w * bv;
            }
        }
    }

    let total: f32 = flow.values().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    flow.into_iter().map(|((a, bb), v)| (a, bb, v / total)).collect()
}

/// Diagnostic: `Σ_seeds seed_mass · B_maxdepth[(seed, query_relation)]`.
/// Equals `propagate(...)[target]` exactly when `params.min_intensity == 0.0`.
pub fn backward_mass_at_target(
    graph: &Graph,
    vocab: &RelationVocab,
    seeds: &[(NodeId, f32)],
    query_relation: Option<RelationId>,
    target: NodeId,
    params: &PropagationParams,
) -> f32 {
    let d = params.max_depth;
    if (target as usize) >= graph.num_nodes() {
        return 0.0;
    }
    let mut denom_cache: HashMap<NodeId, f32> = HashMap::new();
    let f = forward(graph, vocab, seeds, query_relation, params, &mut denom_cache);
    let ball = ball_of(&f);
    let b = backward(graph, vocab, target, &ball, params, &mut denom_cache);

    seeds
        .iter()
        .map(|&(s, m)| m * b[d].get(&(s, query_relation)).copied().unwrap_or(0.0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::{propagate, PropagationParams};
    use crate::relation::RelationVocab;

    fn exact() -> PropagationParams {
        // The forward/backward invariant is exact only without pruning.
        PropagationParams { max_depth: 4, min_intensity: 0.0 }
    }

    fn chain() -> Graph {
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let adj = vec![vec![e(1)], vec![e(2)], vec![e(3)], vec![]];
        Graph::from_adjacency(4, adj, NodeProps::default()).unwrap()
    }

    /// 0 -(A)-> 1 -(B)-> 2, with sim(A,B) = 0.5
    fn two_relation() -> (Graph, RelationVocab) {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::new(vec!["A".into(), "B".into()], vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        (g, v)
    }

    #[test]
    fn credits_are_l1_normalized() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let c = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact());
        let total: f32 = c.iter().map(|&(_, _, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn chain_credits_only_the_diagonal() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let c = credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact());
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].0, c[0].1), (0, 0));
        assert!((c[0].2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn two_relation_splits_credit_evenly() {
        let (g, v) = two_relation();
        let mut c = credit(&g, &v, &[(0, 1.0)], Some(0), 2, &exact());
        c.sort_by_key(|&(a, b, _)| (a, b));
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].0, c[0].1), (0, 0));
        assert!((c[0].2 - 0.5).abs() < 1e-5, "diag got {}", c[0].2);
        assert_eq!((c[1].0, c[1].1), (0, 1));
        assert!((c[1].2 - 0.5).abs() < 1e-5, "off-diag got {}", c[1].2);
    }

    #[test]
    fn unreachable_target_yields_no_credit() {
        // node 3 has no path from node 2 in this 4-node graph with only 0->1
        let e = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(4, vec![vec![e], vec![], vec![], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        assert!(credit(&g, &v, &[(0, 1.0)], Some(0), 3, &exact()).is_empty());
    }

    #[test]
    fn untyped_query_skips_the_first_hop_transition() {
        // query_relation = None => the single hop has r_in = None => no transition
        let e = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(2, vec![vec![e], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        assert!(credit(&g, &v, &[(0, 1.0)], None, 1, &exact()).is_empty());
    }

    // THE cross-check: backward value at the seed == forward mass at the target.
    #[test]
    fn backward_matches_forward_on_chain() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&3).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 3, &p);
        assert!((fwd - 0.614125).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_with_refraction() {
        let (g, v) = two_relation();
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&2).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 2, &p);
        assert!((fwd - 0.36125).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_on_multipath() {
        // 0 -> 1 -> 2 and 0 -> 2 (two paths to node 2)
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![e(1), e(2)], vec![e(2)], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        let p = exact();
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&2).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 2, &p);
        assert!((fwd - 0.78625).abs() < 1e-5, "kernel value drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_when_target_has_out_edges() {
        // 0 -> 1 and 1 -> 1 (self-loop). The target is NOT a sink, so the
        // "+[v == target] at every budget j" branch runs with a real traversal out
        // of the target. The forward kernel counts EVERY arrival: 0.85 + 0.7225.
        let e = |dst| (dst, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(2, vec![vec![e(1)], vec![e(1)]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        let p = PropagationParams { max_depth: 2, min_intensity: 0.0 };
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&1).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 1, &p);
        assert!((fwd - 1.5725).abs() < 1e-5, "forward drifted: {fwd}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn untyped_seed_still_credits_later_typed_hops() {
        // query_relation = None: hop 1 has r_in = None and credits nothing, but its
        // mass must still reach hop 2, whose (A -> B) transition takes ALL the credit.
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(2);
        let p = PropagationParams { max_depth: 2, min_intensity: 0.0 };
        let c = credit(&g, &v, &[(0, 1.0)], None, 2, &p);
        assert_eq!(c.len(), 1, "only the typed second hop is credited");
        assert_eq!((c[0].0, c[0].1), (0, 1));
        assert!((c[0].2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn backward_matches_forward_with_attenuation() {
        // node 0 has two out-edges with different attenuation => p(0->1) = 1.0/1.5
        let e1 = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let e2 = (2u32, EdgeProps { attenuation: 0.5, relation: 0, is_portal: false });
        let g = Graph::from_adjacency(3, vec![vec![e1, e2], vec![], vec![]], NodeProps::default()).unwrap();
        let v = RelationVocab::uniform(1);
        let p = PropagationParams { max_depth: 1, min_intensity: 0.0 };
        let fwd = *propagate(&g, &v, &[(0, 1.0)], Some(0), &p).get(&1).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &[(0, 1.0)], Some(0), 1, &p);
        let expected = 0.85f32 * (1.0 / 1.5); // reflection * p(0->1)
        assert!((fwd - expected).abs() < 1e-5, "forward {fwd} != {expected}");
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }

    #[test]
    fn backward_matches_forward_with_multiple_seeds() {
        let g = chain();
        let v = RelationVocab::uniform(1);
        let p = exact();
        let seeds = [(0u32, 0.5f32), (1u32, 0.5f32)];
        let fwd = *propagate(&g, &v, &seeds, Some(0), &p).get(&3).unwrap();
        let bwd = backward_mass_at_target(&g, &v, &seeds, Some(0), 3, &p);
        assert!((bwd - fwd).abs() < 1e-5, "bwd {bwd} != fwd {fwd}");
    }
}
