/// Error types for RGDB operations

use crate::graph::{NodeId, RoomId};
use thiserror::Error;

/// Errors for graph operations
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Invalid node ID: {0} (graph has {1} nodes)")]
    InvalidNodeId(NodeId, usize),
    #[error("Invalid room ID: {0}")]
    InvalidRoomId(RoomId),
    #[error("Graph structure corrupted: {0}")]
    CorruptedStructure(String),
    #[error("Adjacency list length mismatch: expected {0}, got {1}")]
    AdjacencyLengthMismatch(usize, usize),
    #[error("Invalid size: {0}")]
    InvalidSize(String),
    #[error("Integer overflow in calculation")]
    IntegerOverflow,
}

/// Errors for propagation operations
#[derive(Debug, Error)]
pub enum PropagationError {
    #[error("Invalid source node: {0}")]
    InvalidSourceNode(NodeId),
    #[error("Propagation failed: {0}")]
    PropagationFailed(String),
}

/// Helper for safe type conversions
pub fn node_id_to_usize(id: NodeId) -> Result<usize, GraphError> {
    usize::try_from(id).map_err(|_| GraphError::InvalidNodeId(id, usize::MAX))
}

pub fn usize_to_node_id(val: usize) -> Result<NodeId, GraphError> {
    NodeId::try_from(val).map_err(|_| GraphError::InvalidSize(format!(
        "Value {} exceeds NodeId::MAX ({})", val, NodeId::MAX
    )))
}

