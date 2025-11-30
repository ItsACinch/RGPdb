
type NodeId = u32;
type AngleBin = u8;

/// Number of discrete angle bins (2D "semantic directions").
const N_ANGLE_BINS: usize = 16;

/// Physical-ish properties of a node.
#[derive(Debug, Clone, Copy)]
struct NodeProps {
    /// Intrinsic emission from this node.
    luminance: f32,
    /// Fraction of incoming intensity that gets re-emitted.
    reflection: f32,
    /// Refraction index controlling how strongly direction changes are penalized.
    refraction_index: f32,
}

/// Physical-ish properties of an edge.
#[derive(Debug, Clone, Copy)]
struct EdgeProps {
    /// Fraction of intensity lost along this edge.
    attenuation: f32,
    /// Discrete direction bin for this relationship.
    angle_bin: AngleBin,
}

/// Simple CSR graph layout with node + edge properties.
///
/// row_ptr[u]..row_ptr[u+1] indexes into col_idx / edge_props
#[derive(Debug)]
struct Graph {
    num_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<NodeId>,
    node_props: Vec<NodeProps>,
    edge_props: Vec<EdgeProps>,
}

impl Graph {
    /// Create an empty graph with a fixed number of nodes.
    fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            node_props: vec![
                NodeProps {
                    luminance: 0.0,
                    reflection: 1.0,
                    refraction_index: 1.0,
                };
                num_nodes
            ],
            edge_props: Vec::new(),
        }
    }

    /// Set properties for a node.
    fn set_node_props(&mut self, u: NodeId, props: NodeProps) {
        let idx = u as usize;
        assert!(idx < self.num_nodes);
        self.node_props[idx] = props;
    }

    /// Build graph from adjacency list with edges (v, EdgeProps).
    ///
    /// This is just for PoC convenience.
    fn from_adjacency(
        num_nodes: usize,
        adj: Vec<Vec<(NodeId, EdgeProps)>>,
        default_node_props: NodeProps,
    ) -> Self {
        assert_eq!(num_nodes, adj.len());
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);

        let mut col_idx = Vec::new();
        let mut edge_props = Vec::new();

        for neighbors in &adj {
            col_idx.extend(neighbors.iter().map(|(v, _)| *v));
            edge_props.extend(neighbors.iter().map(|(_, ep)| *ep));
            row_ptr.push(col_idx.len());
        }

        let node_props = vec![default_node_props; num_nodes];

        Self {
            num_nodes,
            row_ptr,
            col_idx,
            node_props,
            edge_props,
        }
    }

    /// Get outgoing edges of node u as (neighbor_id, &EdgeProps)
    fn neighbors(&self, u: NodeId) -> impl Iterator<Item = (NodeId, &EdgeProps)> {
        let idx = u as usize;
        let start = self.row_ptr[idx];
        let end = self.row_ptr[idx + 1];
        self.col_idx[start..end]
            .iter()
            .copied()
            .zip(self.edge_props[start..end].iter())
    }
}

/// Parameters for the light propagation.
#[derive(Debug, Clone, Copy)]
struct LightParams {
    /// Global sharpness factor for refraction penalty.
    k: f32,
    /// Minimum intensity to keep propagating.
    min_intensity: f32,
    /// Maximum BFS-like depth (number of propagation steps).
    max_depth: usize,
    /// Number of angle bins (kept here so we can change it later).
    num_angle_bins: usize,
}

/// Compute circular angular distance between two bins on [0, B).
fn angular_distance(b1: AngleBin, b2: AngleBin, num_bins: usize) -> u8 {
    let b1 = b1 as i32;
    let b2 = b2 as i32;
    let b = num_bins as i32;
    let diff = (b1 - b2).abs();
    let wrapped = b - diff;
    diff.min(wrapped) as u8
}

/// Compute refraction factor ρ based on incoming/outgoing bins and node refraction index.
///
/// ρ = exp(-k * n * (Δ/B)^2)
fn refraction_factor(
    bin_in: AngleBin,
    bin_out: AngleBin,
    n: f32,
    params: &LightParams,
) -> f32 {
    let delta = angular_distance(bin_in, bin_out, params.num_angle_bins) as f32;
    let b = params.num_angle_bins as f32;
    let x = (delta / b).powi(2);
    (-params.k * n * x).exp()
}

/// A single state in the frontier: node u, incoming direction bin, and intensity.
#[derive(Debug, Clone, Copy)]
struct FrontierState {
    node: NodeId,
    angle_bin: AngleBin,
    intensity: f32,
}

/// Perform refractive light propagation from a source node and initial direction bin.
///
/// Returns: total intensity per node (I_total[node]) as a Vec<f32>.
fn propagate_light(
    graph: &Graph,
    source: NodeId,
    initial_bin: AngleBin,
    params: LightParams,
) -> Vec<f32> {
    let n = graph.num_nodes;
    let b = params.num_angle_bins;

    // Intensities per (node, angle_bin) for current step.
    let mut intensities = vec![0.0_f32; n * b];
    // Total accumulated intensities per node.
    let mut total_intensity = vec![0.0_f32; n];

    let src_idx = source as usize;
    let src_props = graph.node_props[src_idx];

    // Seed: starting node gets its luminance in the initial direction.
    intensities[src_idx * b + initial_bin as usize] = src_props.luminance.max(1.0);

    // Initialize frontier with this state.
    let mut frontier = Vec::new();
    frontier.push(FrontierState {
        node: source,
        angle_bin: initial_bin,
        intensity: intensities[src_idx * b + initial_bin as usize],
    });

    // We can push the source node's own intensity into total_intensity.
    total_intensity[src_idx] += intensities[src_idx * b + initial_bin as usize];

    for depth in 0..params.max_depth {
        if frontier.is_empty() {
            println!("Terminating at depth {}: frontier empty", depth);
            break;
        }

        let mut next_frontier = Vec::new();

        for state in frontier.iter().copied() {
            let u = state.node;
            let u_idx = u as usize;
            let bin_in = state.angle_bin;
            let intensity_in = state.intensity;
            if intensity_in < params.min_intensity {
                continue;
            }

            let u_props = graph.node_props[u_idx];
            let reflected = intensity_in * u_props.reflection;

            for (v, eprops) in graph.neighbors(u) {
                let bin_out = eprops.angle_bin;
                let n_u = u_props.refraction_index;
                let rho = refraction_factor(bin_in, bin_out, n_u, &params);
                let transmitted = reflected * (1.0 - eprops.attenuation) * rho;
                if transmitted < params.min_intensity {
                    continue;
                }

                let v_idx = v as usize;
                // Update per-angle intensity
                let idx = v_idx * b + bin_out as usize;
                // For PoC: just overwrite or max; for full: consider max or sum with atomic.
                if transmitted > intensities[idx] {
                    intensities[idx] = transmitted;
                    next_frontier.push(FrontierState {
                        node: v,
                        angle_bin: bin_out,
                        intensity: transmitted,
                    });
                }

                // Update total intensity accumulation per node.
                total_intensity[v_idx] += transmitted;
            }
        }

        frontier = next_frontier;
    }

    total_intensity
}

/// Convert total intensity per node into a "light distance":
/// d = -log(I + eps)
fn intensity_to_distance(intensities: &[f32], eps: f32) -> Vec<f32> {
    intensities
        .iter()
        .map(|&i| -(i + eps).ln()) // natural log
        .collect()
}

fn main() {
    // Build a tiny graph:
    //
    // 0 (A) -> 1 (B) -> 2 (C)
    //
    // A->B has angle_bin = 2
    // B->C has angle_bin = 2 (aligned) OR try 8 (misaligned) to see refraction effect.

    let num_nodes = 3;

    let default_node_props = NodeProps {
        luminance: 1.0,
        reflection: 0.9,
        refraction_index: 2.0,
    };

    // Adjacency list: each entry is Vec<(neighbor, EdgeProps)>
    let adj = vec![
        // Node 0 neighbors
        vec![(
            1,
            EdgeProps {
                attenuation: 0.1,
                angle_bin: 2,
            },
        )],
        // Node 1 neighbors
        vec![(
            2,
            EdgeProps {
                attenuation: 0.2,
                angle_bin: 2, // try 8 to simulate strong bending
            },
        )],
        // Node 2 neighbors
        vec![], // no outgoing
    ];

    let mut graph = Graph::from_adjacency(num_nodes, adj, default_node_props);

    // Let’s make node 0 a bit brighter and node 1 more reflective.
    graph.set_node_props(
        0,
        NodeProps {
            luminance: 2.0,
            reflection: 0.9,
            refraction_index: 1.0,
        },
    );
    graph.set_node_props(
        1,
        NodeProps {
            luminance: 0.5,
            reflection: 0.8,
            refraction_index: 2.0,
        },
    );
    graph.set_node_props(
        2,
        NodeProps {
            luminance: 0.0,
            reflection: 0.5,
            refraction_index: 1.5,
        },
    );

    let params = LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: N_ANGLE_BINS,
    };

    let source: NodeId = 0;
    let initial_bin: AngleBin = 2;

    let intensities = propagate_light(&graph, source, initial_bin, params);
    let distances = intensity_to_distance(&intensities, 1e-6);

    println!("Total intensities per node:");
    for (i, val) in intensities.iter().enumerate() {
        println!("  Node {}: {:.6}", i, val);
    }

    println!("\nLight distances per node:");
    for (i, d) in distances.iter().enumerate() {
        println!("  Node {}: d = {:.6}", i, d);
    }
}
