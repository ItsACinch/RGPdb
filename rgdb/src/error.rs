/// Error types for RGDB operations

use crate::graph::{NodeId, RoomId, AngleBin};
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

/// Errors for PVS operations
#[derive(Debug, Error)]
pub enum PVSError {
    #[error("Invalid room ID: {0}")]
    InvalidRoomId(RoomId),
    #[error("Invalid angle bin: {0}")]
    InvalidAngleBin(AngleBin),
    #[error("Out of bounds: room_id={room_id}, angle_bin={angle_bin}, num_rooms={num_rooms}, num_angle_bins={num_angle_bins}")]
    OutOfBounds {
        room_id: RoomId,
        angle_bin: AngleBin,
        num_rooms: usize,
        num_angle_bins: usize,
    },
    #[error("Integer overflow in index calculation")]
    IntegerOverflow,
    #[error("Room {0} is empty (no nodes)")]
    EmptyRoom(RoomId),
}

/// Errors for propagation operations
#[derive(Debug, Error)]
pub enum PropagationError {
    #[error("Invalid source node: {0}")]
    InvalidSourceNode(NodeId),
    #[error("Invalid angle bin: {0} (max: {1})")]
    InvalidAngleBin(AngleBin, usize),
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

