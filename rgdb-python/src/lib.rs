//! PyO3 bindings for the sparse typed-PPR RGDB core.

use pyo3::prelude::*;
use rgdb::graph::{Graph, NodeProps, EdgeProps, NodeId, RelationId};
use rgdb::relation::RelationVocab;
use rgdb::propagation::{propagate as rust_propagate, propagate_layered as rust_propagate_layered, PropagationParams};
use rgdb::engine::{RgdbEngine, QueryId};
use rgdb::transitions::TransitionConfig;
use rgdb::depth_weights::DepthWeights;

#[pyclass(name = "RgdbGraph")]
struct PyGraph { inner: Graph }

#[pymethods]
impl PyGraph {
    #[getter]
    fn num_nodes(&self) -> usize { self.inner.num_nodes() }
    #[getter]
    fn num_edges(&self) -> usize { self.inner.num_edges() }
}

#[pyclass(name = "RelationVocab")]
#[derive(Clone)]
struct PyVocab { inner: RelationVocab }

#[pyfunction]
fn uniform_vocab(n: usize) -> PyVocab {
    PyVocab { inner: RelationVocab::uniform(n) }
}

#[pyfunction]
fn vocab_from_matrix(names: Vec<String>, flat_similarity: Vec<f32>) -> PyResult<PyVocab> {
    RelationVocab::new(names, flat_similarity)
        .map(|inner| PyVocab { inner })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
#[pyo3(signature = (num_nodes, adjacency, node_reflections=None, node_refractions=None))]
fn build_graph(
    num_nodes: usize,
    adjacency: Vec<Vec<(u32, f32, u16)>>,
    node_reflections: Option<Vec<f32>>,
    node_refractions: Option<Vec<f32>>,
) -> PyResult<PyGraph> {
    let adj: Vec<Vec<(NodeId, EdgeProps)>> = adjacency
        .into_iter()
        .map(|ns| ns.into_iter().map(|(dst, attenuation, relation)| {
            (dst, EdgeProps { attenuation, relation: relation as RelationId, is_portal: false })
        }).collect())
        .collect();

    let mut graph = Graph::from_adjacency(num_nodes, adj, NodeProps::default())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;

    for i in 0..num_nodes {
        let mut p = NodeProps::default();
        let mut changed = false;
        if let Some(ref v) = node_reflections {
            if i < v.len() { p.reflection = v[i]; changed = true; }
        }
        if let Some(ref v) = node_refractions {
            if i < v.len() { p.refraction_index = v[i]; changed = true; }
        }
        if changed {
            let _ = graph.set_node_props(i as NodeId, p);
        }
    }
    Ok(PyGraph { inner: graph })
}

fn to_depth_weights(
    depth_weights: Option<Vec<f32>>,
    max_depth: usize,
) -> PyResult<Option<DepthWeights>> {
    match depth_weights {
        None => Ok(None),
        Some(v) => DepthWeights::from_vec(v, max_depth)
            .map(Some)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None))]
fn propagate(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
    depth_weights: Option<Vec<f32>>,
) -> PyResult<Vec<(u32, f32)>> {
    let params = PropagationParams {
        max_depth,
        min_intensity,
        depth_weights: to_depth_weights(depth_weights, max_depth)?,
    };
    let totals = rust_propagate(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    Ok(totals.into_iter().collect())
}

#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3))]
fn propagate_layered(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
) -> (Vec<(u32, Vec<f32>)>, Vec<(u32, u16)>) {
    let params = PropagationParams { max_depth, min_intensity, depth_weights: None };
    let r = rust_propagate_layered(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    let per_depth = r.per_depth.into_iter().collect();
    let dominant = r.dominant_incoming.into_iter().collect();
    (per_depth, dominant)
}

#[pyclass(name = "Engine")]
struct PyEngine { inner: RgdbEngine }

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (graph, vocab, prior_strength=10.0, floor=0.05, decay=1.0, rebuild_every_n=64))]
    fn new(graph: &PyGraph, vocab: &PyVocab, prior_strength: f32, floor: f32, decay: f32, rebuild_every_n: u32) -> Self {
        let cfg = TransitionConfig { prior_strength, floor, decay, rebuild_every_n };
        PyEngine { inner: RgdbEngine::new(graph.inner.clone(), vocab.inner.clone(), cfg) }
    }

    #[staticmethod]
    fn load(graph: &PyGraph, path: &str) -> PyResult<Self> {
        RgdbEngine::load(graph.inner.clone(), path)
            .map(|inner| PyEngine { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None))]
    fn query(
        &self,
        seeds: Vec<(u32, f32)>,
        query_relation: Option<u16>,
        max_depth: usize,
        min_intensity: f32,
        depth_weights: Option<Vec<f32>>,
    ) -> PyResult<(Vec<(u32, f32)>, u64)> {
        let params = PropagationParams {
            max_depth,
            min_intensity,
            depth_weights: to_depth_weights(depth_weights, max_depth)?,
        };
        let r = self.inner.query(&seeds, query_relation, &params);
        Ok((r.ranked, r.query_id))
    }

    #[pyo3(signature = (query_id, target, signal=1.0))]
    fn record_feedback(&self, query_id: u64, target: u32, signal: f32) -> PyResult<()> {
        self.inner
            .record_feedback(query_id as QueryId, target, signal)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn refresh(&self) { self.inner.refresh(); }
    fn matrix(&self) -> Vec<f32> { self.inner.matrix() }
    fn counts(&self) -> Vec<f32> { self.inner.counts_snapshot() }
    fn events_since_rebuild(&self) -> u32 { self.inner.events_since_rebuild() }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }
}

#[pymodule]
fn _rgdb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyVocab>()?;
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(vocab_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_layered, m)?)?;
    Ok(())
}
