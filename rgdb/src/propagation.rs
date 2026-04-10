use crate::graph::{Graph, NodeId, AngleBin};
use crate::pvs::PVS;
use crate::property_map::{RelationshipProperty, DEFAULT_PROPERTY_MAP};

/// Default minimum luminance value
const DEFAULT_MIN_LUMINANCE: f32 = 1.0;

/// Minimum property similarity to allow propagation (early termination threshold)
const MIN_PROPERTY_SIMILARITY: f32 = 0.2;

/// Parameters for the light propagation.
#[derive(Debug, Clone, Copy)]
pub struct LightParams {
    /// Global sharpness factor for refraction penalty.
    pub k: f32,
    /// Minimum intensity to keep propagating.
    pub min_intensity: f32,
    /// Maximum BFS-like depth (number of propagation steps).
    pub max_depth: usize,
    /// Number of angle bins (kept here so we can change it later).
    pub num_angle_bins: usize,
}

impl Default for LightParams {
    fn default() -> Self {
        Self {
            k: 5.0,
            min_intensity: 1e-3,
            max_depth: 4,
            num_angle_bins: crate::graph::N_ANGLE_BINS,
        }
    }
}

/// Compute circular angular distance between two bins on [0, B).
pub fn angular_distance(b1: AngleBin, b2: AngleBin, num_bins: usize) -> u8 {
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
///
/// # Safety
/// Returns 0.0 if num_angle_bins is 0 to prevent division by zero.
pub fn refraction_factor(
    bin_in: AngleBin,
    bin_out: AngleBin,
    n: f32,
    params: &LightParams,
) -> f32 {
    if params.num_angle_bins == 0 {
        return 0.0; // Prevent division by zero
    }
    
    let delta = angular_distance(bin_in, bin_out, params.num_angle_bins) as f32;
    let b = params.num_angle_bins as f32;
    
    // Safe division - we already checked b != 0
    let x = (delta / b).powi(2);
    (-params.k * n * x).exp()
}

/// A single state in the frontier: node u, incoming direction bin, and intensity.
#[derive(Debug, Clone, Copy)]
pub struct FrontierState {
    pub node: NodeId,
    pub angle_bin: AngleBin,
    pub intensity: f32,
}

/// Perform refractive light propagation from a source node and initial direction bin.
///
/// Returns: total intensity per node (I_total[node]) as a Vec<f32>.
///
/// # Panics
/// Panics if source node is out of bounds. Use `propagate_light_with_pvs` for safe access.
pub fn propagate_light(
    graph: &Graph,
    source: NodeId,
    initial_bin: AngleBin,
    params: LightParams,
) -> Vec<f32> {
    propagate_light_with_pvs(graph, source, initial_bin, params, None)
}

/// Perform refractive light propagation with optional PVS pruning.
///
/// If pvs is Some, uses PVS to skip non-visible rooms during propagation.
///
/// # Panics
/// Panics if source node is out of bounds or if index calculations overflow.
pub fn propagate_light_with_pvs(
    graph: &Graph,
    source: NodeId,
    initial_bin: AngleBin,
    params: LightParams,
    pvs: Option<&PVS>,
) -> Vec<f32> {
    let n = graph.num_nodes();
    let b = params.num_angle_bins;
    
    if n == 0 || b == 0 {
        return Vec::new();
    }

    // Intensities per (node, angle_bin) for current step.
    let mut intensities = vec![0.0_f32; n * b];
    // Total accumulated intensities per node.
    let mut total_intensity = vec![0.0_f32; n];

    let src_idx = source as usize;
    if src_idx >= n {
        // Invalid source node - return zeros
        return total_intensity;
    }
    
    let src_props = graph.node_props()[src_idx];

    // Initialize frontier with directional or uniform luminance
    let mut frontier = Vec::new();
    
    // NEW: Directional luminance initialization
    if let Some(_property) = src_props.relationship_property {
        // Emit in each direction based on directional_luminance
        for angle_bin in 0..b {
            let luminance = src_props.directional_luminance[angle_bin];
            if luminance > params.min_intensity {
                let angle_bin_u8 = angle_bin as AngleBin;
                let src_intensity_idx = src_idx
                    .checked_mul(b)
                    .and_then(|x| x.checked_add(angle_bin));
                
                if let Some(idx) = src_intensity_idx {
                    if idx < intensities.len() {
                        intensities[idx] = luminance;
                        total_intensity[src_idx] += luminance;
                        frontier.push(FrontierState {
                            node: source,
                            angle_bin: angle_bin_u8,
                            intensity: luminance,
                        });
                    }
                }
            }
        }
    } else {
        // Fallback: Uniform emission (backward compatibility)
        let initial_bin_idx = initial_bin as usize;
        if initial_bin_idx < b {
            let src_intensity_idx = src_idx
                .checked_mul(b)
                .and_then(|x| x.checked_add(initial_bin_idx));
            
            if let Some(idx) = src_intensity_idx {
                if idx < intensities.len() {
                    let initial_intensity = src_props.luminance.max(DEFAULT_MIN_LUMINANCE);
                    intensities[idx] = initial_intensity;
                    total_intensity[src_idx] += initial_intensity;
                    frontier.push(FrontierState {
                        node: source,
                        angle_bin: initial_bin,
                        intensity: initial_intensity,
                    });
                }
            }
        }
    }
    
    // Continue with propagation if frontier is not empty
    if !frontier.is_empty() {

                for _depth in 0..params.max_depth {
                    if frontier.is_empty() {
                        break;
                    }

                    let mut next_frontier = Vec::new();

                    for state in frontier.iter().copied() {
                        let u = state.node;
                        let u_idx = u as usize;
                        if u_idx >= n {
                            continue;
                        }
                        
                        let bin_in = state.angle_bin;
                        let intensity_in = state.intensity;
                        if intensity_in < params.min_intensity {
                            continue;
                        }

                        let u_props = graph.node_props()[u_idx];
                        let reflected = intensity_in * u_props.reflection;
                        let u_room = graph.get_room(u).unwrap_or(0);

                        for (v, eprops) in graph.neighbors(u) {
                            let v_room = graph.get_room(v).unwrap_or(0);
                            
                            // PVS pruning: skip if room is not visible
                            if let Some(pvs) = pvs {
                                if pvs.is_visible(u_room, bin_in, v_room).unwrap_or(false) == false {
                                    continue;
                                }
                            }
                            
                            // NEW: Early termination - check property compatibility
                            let v_idx = v as usize;
                            if v_idx < n {
                                let v_props = graph.node_props()[v_idx];
                                if let Some(u_property) = u_props.relationship_property {
                                    if let Some(v_property) = v_props.relationship_property {
                                        // Check semantic compatibility
                                        let similarity = DEFAULT_PROPERTY_MAP.similarity(u_property, v_property);
                                        if similarity < MIN_PROPERTY_SIMILARITY {
                                            continue; // Skip this edge - properties incompatible
                                        }
                                    }
                                }
                            }

                            let bin_out = eprops.angle_bin;
                            let n_u = u_props.refraction_index;
                            let rho = refraction_factor(bin_in, bin_out, n_u, &params);
                            let attenuation_factor = 1.0 - eprops.attenuation;
                            let transmitted = reflected * attenuation_factor * rho;
                            
                            if transmitted < params.min_intensity {
                                continue;
                            }

                            // v_idx already checked above
                            if v_idx >= n {
                                continue;
                            }
                            
                            // Update per-angle intensity with bounds checking
                            let bin_out_idx = bin_out as usize;
                            if bin_out_idx < b {
                                let v_intensity_idx = v_idx
                                    .checked_mul(b)
                                    .and_then(|x| x.checked_add(bin_out_idx));
                                
                                if let Some(idx) = v_intensity_idx {
                                    if idx < intensities.len() {
                                        // For PoC: just overwrite or max; for full: consider max or sum with atomic.
                                        if transmitted > intensities[idx] {
                                            intensities[idx] = transmitted;
                                            next_frontier.push(FrontierState {
                                                node: v,
                                                angle_bin: bin_out,
                                                intensity: transmitted,
                                            });
                                        }
                                    }
                                }
                            }

                            // Update total intensity accumulation per node.
                            if v_idx < total_intensity.len() {
                                total_intensity[v_idx] += transmitted;
                            }
                        }
                    }

                    frontier = next_frontier;
                }
    }

    total_intensity
}

/// Convert total intensity per node into a "light distance":
/// d = -log(I + eps)
///
/// # Safety
/// Returns infinity for negative intensities. Uses `eps` to prevent log(0).
pub fn intensity_to_distance(intensities: &[f32], eps: f32) -> Vec<f32> {
    intensities
        .iter()
        .map(|&i| {
            let value = i + eps;
            if value > 0.0 {
                -value.ln()
            } else {
                f32::INFINITY
            }
        })
        .collect()
}
