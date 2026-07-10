//! PyO3 bindings for the sparse typed-PPR RGDB core.

use pyo3::prelude::*;
use rgdb::graph::{Graph, NodeProps, EdgeProps, NodeId, RelationId};
use rgdb::relation::RelationVocab;
use rgdb::propagation::{propagate as rust_propagate, PropagationParams};

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

#[pyfunction]
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3))]
fn propagate(
    graph: &PyGraph,
    vocab: &PyVocab,
    seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>,
    max_depth: usize,
    min_intensity: f32,
) -> Vec<(u32, f32)> {
    let params = PropagationParams { max_depth, min_intensity };
    let totals = rust_propagate(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    totals.into_iter().collect()
}

#[pymodule]
fn _rgdb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyVocab>()?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(vocab_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    Ok(())
}
