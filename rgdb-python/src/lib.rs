//! PyO3 bindings for RGDB - exposes graph construction and light propagation to Python.

use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};

// Re-export core types from rgdb
use rgdb::graph::{
    Graph, NodeProps as RustNodeProps, EdgeProps as RustEdgeProps,
    NodeId, AngleBin, N_ANGLE_BINS,
};
use rgdb::propagation::{
    LightParams as RustLightParams,
    propagate_light as rust_propagate_light,
};

/// Python wrapper for RGDB NodeProps.
#[pyclass(name = "NodeProps")]
#[derive(Clone)]
struct PyNodeProps {
    inner: RustNodeProps,
}

#[pymethods]
impl PyNodeProps {
    #[new]
    #[pyo3(signature = (luminance=1.0, reflection=1.0, refraction_index=1.0, default_angle_bin=0))]
    fn new(luminance: f32, reflection: f32, refraction_index: f32, default_angle_bin: u8) -> Self {
        let mut props = RustNodeProps::from_uniform_luminance(luminance);
        props.reflection = reflection;
        props.refraction_index = refraction_index;
        props.default_angle_bin = default_angle_bin;
        PyNodeProps { inner: props }
    }

    /// Create with uniform luminance (convenience constructor).
    #[staticmethod]
    #[pyo3(signature = (luminance=1.0))]
    fn uniform(luminance: f32) -> Self {
        PyNodeProps {
            inner: RustNodeProps::from_uniform_luminance(luminance),
        }
    }
}

/// Python wrapper for RGDB EdgeProps.
#[pyclass(name = "EdgeProps")]
#[derive(Clone)]
struct PyEdgeProps {
    inner: RustEdgeProps,
}

#[pymethods]
impl PyEdgeProps {
    #[new]
    #[pyo3(signature = (attenuation=0.0, angle_bin=0, is_portal=false))]
    fn new(attenuation: f32, angle_bin: u8, is_portal: bool) -> Self {
        PyEdgeProps {
            inner: RustEdgeProps {
                attenuation,
                angle_bin,
                is_portal,
            },
        }
    }
}

/// Python wrapper for RGDB LightParams.
#[pyclass(name = "LightParams")]
#[derive(Clone)]
struct PyLightParams {
    inner: RustLightParams,
}

#[pymethods]
impl PyLightParams {
    #[new]
    #[pyo3(signature = (k=5.0, min_intensity=1e-3, max_depth=4))]
    fn new(k: f32, min_intensity: f32, max_depth: usize) -> Self {
        PyLightParams {
            inner: RustLightParams {
                k,
                min_intensity,
                max_depth,
                num_angle_bins: N_ANGLE_BINS,
            },
        }
    }

    #[getter]
    fn k(&self) -> f32 { self.inner.k }

    #[getter]
    fn min_intensity(&self) -> f32 { self.inner.min_intensity }

    #[getter]
    fn max_depth(&self) -> usize { self.inner.max_depth }
}

/// Python wrapper for RGDB Graph (frozen CSR format).
#[pyclass(name = "RgdbGraph", dict)]
struct PyGraph {
    inner: Graph,
}

#[pymethods]
impl PyGraph {
    /// Number of nodes in the graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of edges in the graph.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }
}

/// Build a graph from adjacency data.
///
/// Args:
///     num_nodes: Number of nodes.
///     adjacency: List of neighbor lists. adjacency[u] = [(dst, attenuation, angle_bin), ...]
///     node_luminances: Optional list of luminance values per node (default 1.0).
///     node_reflections: Optional list of reflection values per node (default 1.0).
///     node_refractions: Optional list of refraction index values per node (default 1.0).
///
/// Returns:
///     RgdbGraph instance.
#[pyfunction]
#[pyo3(signature = (num_nodes, adjacency, node_luminances=None, node_reflections=None, node_refractions=None))]
fn build_graph(
    num_nodes: usize,
    adjacency: Vec<Vec<(u32, f32, u8)>>,
    node_luminances: Option<Vec<f32>>,
    node_reflections: Option<Vec<f32>>,
    node_refractions: Option<Vec<f32>>,
) -> PyResult<PyGraph> {
    // Convert adjacency to Rust format
    let adj: Vec<Vec<(NodeId, RustEdgeProps)>> = adjacency
        .into_iter()
        .map(|neighbors| {
            neighbors
                .into_iter()
                .map(|(dst, attenuation, angle_bin)| {
                    (
                        dst,
                        RustEdgeProps {
                            attenuation,
                            angle_bin,
                            is_portal: false,
                        },
                    )
                })
                .collect()
        })
        .collect();

    // Build graph with default node props, then set per-node overrides
    let default_props = RustNodeProps::default();

    let mut graph = Graph::from_adjacency(num_nodes, adj, default_props)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

    // Apply per-node property overrides
    for i in 0..num_nodes {
        let mut needs_update = false;
        let mut p = default_props;

        if let Some(ref lums) = node_luminances {
            if i < lums.len() && lums[i] != 1.0 {
                p.luminance = lums[i];
                p.directional_luminance = [lums[i]; N_ANGLE_BINS];
                needs_update = true;
            }
        }
        if let Some(ref refs) = node_reflections {
            if i < refs.len() && refs[i] != 1.0 {
                p.reflection = refs[i];
                needs_update = true;
            }
        }
        if let Some(ref refrs) = node_refractions {
            if i < refrs.len() && refrs[i] != 1.0 {
                p.refraction_index = refrs[i];
                needs_update = true;
            }
        }

        if needs_update {
            let _ = graph.set_node_props(i as NodeId, p);
        }
    }

    Ok(PyGraph { inner: graph })
}

/// Perform light propagation from a source node.
///
/// Args:
///     graph: RgdbGraph instance.
///     source: Source node ID.
///     initial_bin: Starting angle bin (default 0).
///     params: LightParams instance (optional, uses defaults).
///
/// Returns:
///     numpy array of total intensity per node (shape: [num_nodes]).
#[pyfunction]
#[pyo3(signature = (graph, source, initial_bin=0, params=None))]
fn propagate_light<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    source: u32,
    initial_bin: u8,
    params: Option<&PyLightParams>,
) -> Bound<'py, PyArray1<f32>> {
    let lp = params
        .map(|p| p.inner)
        .unwrap_or_default();

    let result = rust_propagate_light(&graph.inner, source, initial_bin, lp);
    result.into_pyarray(py)
}

/// Find top-k nodes most influenced by source via light propagation.
///
/// Args:
///     graph: RgdbGraph instance.
///     source: Source node ID.
///     initial_bin: Starting angle bin.
///     k: Number of results.
///     params: LightParams instance (optional).
///
/// Returns:
///     List of (node_id, intensity, distance) tuples, sorted by intensity descending.
#[pyfunction]
#[pyo3(signature = (graph, source, initial_bin=0, k=20, params=None))]
fn query_top_k(
    graph: &PyGraph,
    source: u32,
    initial_bin: u8,
    k: usize,
    params: Option<&PyLightParams>,
) -> Vec<(u32, f32, f32)> {
    let lp = params
        .map(|p| p.inner)
        .unwrap_or_default();

    let intensities = rust_propagate_light(&graph.inner, source, initial_bin, lp);

    // Collect non-zero, non-source results
    let mut results: Vec<(u32, f32)> = intensities
        .iter()
        .enumerate()
        .filter(|(i, &v)| v > 0.0 && *i as u32 != source)
        .map(|(i, &v)| (i as u32, v))
        .collect();

    // Sort by intensity descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);

    // Convert to (node_id, intensity, distance) tuples
    results
        .into_iter()
        .map(|(node_id, intensity)| {
            let distance = -(intensity + 1e-10_f32).ln();
            (node_id, intensity, distance)
        })
        .collect()
}

/// RGDB native core module for Python.
#[pymodule]
fn _rgdb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("N_ANGLE_BINS", N_ANGLE_BINS)?;
    m.add_class::<PyNodeProps>()?;
    m.add_class::<PyEdgeProps>()?;
    m.add_class::<PyLightParams>()?;
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_light, m)?)?;
    m.add_function(wrap_pyfunction!(query_top_k, m)?)?;
    Ok(())
}
