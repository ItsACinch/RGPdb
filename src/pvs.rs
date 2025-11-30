use crate::graph::{Graph, RoomId, AngleBin};
use crate::rooms::RoomCollection;
use crate::propagation::LightParams;
use crate::error::PVSError;
use hashbrown::HashSet;

/// Maximum depth for PVS computation (prevents excessive computation)
const PVS_MAX_DEPTH_LIMIT: usize = 10;

/// Potentially Visible Set (PVS) structure.
///
/// For each (room, angle_bin) pair, stores the set of rooms that could
/// receive influence when starting from that room with that angle bin.
#[derive(Debug, Clone)]
pub struct PVS {
    /// For each (room_id, angle_bin) -> set of visible room_ids
    visibility: Vec<HashSet<RoomId>>,
    num_rooms: usize,
    num_angle_bins: usize,
}

impl PVS {
    /// Create empty PVS
    pub fn new(num_rooms: usize, num_angle_bins: usize) -> Self {
        let size = num_rooms * num_angle_bins;
        Self {
            visibility: vec![HashSet::new(); size],
            num_rooms,
            num_angle_bins,
        }
    }
    
    /// Get index into visibility array for (room_id, angle_bin) with bounds checking
    fn index(&self, room_id: RoomId, angle_bin: AngleBin) -> Result<usize, PVSError> {
        let room_idx = usize::try_from(room_id)
            .map_err(|_| PVSError::InvalidRoomId(room_id))?;
        let bin_idx = usize::try_from(angle_bin)
            .map_err(|_| PVSError::InvalidAngleBin(angle_bin))?;
        
        // Check bounds before multiplication
        if room_idx >= self.num_rooms || bin_idx >= self.num_angle_bins {
            return Err(PVSError::OutOfBounds {
                room_id,
                angle_bin,
                num_rooms: self.num_rooms,
                num_angle_bins: self.num_angle_bins,
            });
        }
        
        // Use checked arithmetic
        let idx = room_idx
            .checked_mul(self.num_angle_bins)
            .and_then(|x| x.checked_add(bin_idx))
            .ok_or(PVSError::IntegerOverflow)?;
        
        Ok(idx)
    }
    
    /// Mark that room `to_room` is visible from `from_room` with `angle_bin`
    ///
    /// # Errors
    /// Returns `PVSError` if room_id or angle_bin is out of bounds.
    pub fn mark_visible(&mut self, from_room: RoomId, angle_bin: AngleBin, to_room: RoomId) -> Result<(), PVSError> {
        let idx = self.index(from_room, angle_bin)?;
        if idx >= self.visibility.len() {
            return Err(PVSError::OutOfBounds {
                room_id: from_room,
                angle_bin,
                num_rooms: self.num_rooms,
                num_angle_bins: self.num_angle_bins,
            });
        }
        self.visibility[idx].insert(to_room);
        Ok(())
    }
    
    /// Check if `to_room` is visible from `from_room` with `angle_bin`
    ///
    /// # Errors
    /// Returns `PVSError` if room_id or angle_bin is out of bounds.
    pub fn is_visible(&self, from_room: RoomId, angle_bin: AngleBin, to_room: RoomId) -> Result<bool, PVSError> {
        let idx = self.index(from_room, angle_bin)?;
        if idx >= self.visibility.len() {
            return Err(PVSError::OutOfBounds {
                room_id: from_room,
                angle_bin,
                num_rooms: self.num_rooms,
                num_angle_bins: self.num_angle_bins,
            });
        }
        Ok(self.visibility[idx].contains(&to_room))
    }
    
    /// Get all visible rooms from a given room and angle bin
    ///
    /// # Errors
    /// Returns `PVSError` if room_id or angle_bin is out of bounds.
    pub fn get_visible_rooms(&self, from_room: RoomId, angle_bin: AngleBin) -> Result<&HashSet<RoomId>, PVSError> {
        let idx = self.index(from_room, angle_bin)?;
        if idx >= self.visibility.len() {
            return Err(PVSError::OutOfBounds {
                room_id: from_room,
                angle_bin,
                num_rooms: self.num_rooms,
                num_angle_bins: self.num_angle_bins,
            });
        }
        Ok(&self.visibility[idx])
    }
}

/// Compute PVS for a graph.
///
/// Uses propagation simulation to determine which rooms are reachable
/// from each room when starting with each angle bin.
///
/// # Errors
/// Returns `PVSError` if graph structure is invalid or computation fails.
pub fn compute_pvs(
    graph: &Graph,
    rooms: &RoomCollection,
    params: &LightParams,
) -> Result<PVS, PVSError> {
    let num_rooms = rooms.num_rooms();
    let num_angle_bins = params.num_angle_bins;
    let mut pvs = PVS::new(num_rooms, num_angle_bins);
    
    // Pre-compute representative nodes for efficiency
    let representatives = precompute_representative_nodes(graph, rooms)?;
    
    // For each room and each angle bin, simulate propagation
    for room in &rooms.rooms {
        for angle_bin in 0..num_angle_bins as AngleBin {
            // Get representative node (pre-computed)
            let start_node = *representatives.get(&room.id)
                .ok_or_else(|| PVSError::EmptyRoom(room.id))?;
            
            // Simulate propagation with limited depth to find reachable rooms
            let reachable_rooms = find_reachable_rooms(
                graph,
                start_node,
                angle_bin,
                params,
            )?;
            
            // Mark all reachable rooms as visible
            for &reachable_room in &reachable_rooms {
                pvs.mark_visible(room.id, angle_bin, reachable_room)?;
            }
        }
    }
    
    Ok(pvs)
}

/// Pre-compute representative nodes per room (optimization)
fn precompute_representative_nodes(
    graph: &Graph,
    rooms: &RoomCollection,
) -> Result<std::collections::HashMap<RoomId, crate::graph::NodeId>, PVSError> {
    use std::collections::HashMap;
    let mut representatives = HashMap::new();
    
    // Single pass through all nodes
    for node_id in 0..graph.num_nodes() {
        let room_id = graph.get_room(node_id as crate::graph::NodeId)
            .map_err(|_| PVSError::InvalidRoomId(0))?; // This shouldn't happen, but handle it
        
        // Only set if not already set (prefer nodes with edges)
        if !representatives.contains_key(&room_id) {
            let start = graph.row_ptr()[node_id];
            let end = graph.row_ptr()[node_id + 1];
            if start < end {
                representatives.insert(room_id, node_id as crate::graph::NodeId);
            }
        }
    }
    
    // Fill in any missing rooms with any node
    for room in &rooms.rooms {
        if !representatives.contains_key(&room.id) {
            for node_id in 0..graph.num_nodes() {
                if let Ok(r) = graph.get_room(node_id as crate::graph::NodeId) {
                    if r == room.id {
                        representatives.insert(room.id, node_id as crate::graph::NodeId);
                        break;
                    }
                }
            }
        }
    }
    
    Ok(representatives)
}

/// Find a representative node in a room (preferably one with outgoing edges)
fn find_representative_node(
    graph: &Graph,
    room: &crate::rooms::Room,
) -> Result<crate::graph::NodeId, PVSError> {
    // Try to find a node with outgoing edges
    for node_id in 0..graph.num_nodes() {
        if let Ok(room_id) = graph.get_room(node_id as crate::graph::NodeId) {
            if room_id == room.id {
                let start = graph.row_ptr()[node_id];
                let end = graph.row_ptr()[node_id + 1];
                if start < end {
                    return Ok(node_id as crate::graph::NodeId);
                }
            }
        }
    }
    
    // Fallback: any node in the room
    for node_id in 0..graph.num_nodes() {
        if let Ok(room_id) = graph.get_room(node_id as crate::graph::NodeId) {
            if room_id == room.id {
                return Ok(node_id as crate::graph::NodeId);
            }
        }
    }
    
    // No nodes found - this is an error
    Err(PVSError::EmptyRoom(room.id))
}

/// Find all rooms reachable from a starting node with a given angle bin.
fn find_reachable_rooms(
    graph: &Graph,
    start_node: crate::graph::NodeId,
    initial_angle_bin: AngleBin,
    params: &LightParams,
) -> Result<HashSet<RoomId>, PVSError> {
    use crate::propagation::propagate_light;
    
    // Use a limited-depth propagation to find reachable rooms
    let mut limited_params = *params;
    limited_params.max_depth = params.max_depth.min(PVS_MAX_DEPTH_LIMIT);
    
    let intensities = propagate_light(graph, start_node, initial_angle_bin, limited_params);
    
    // Collect all rooms that received non-zero intensity
    let mut reachable = HashSet::new();
    for (node_id, &intensity) in intensities.iter().enumerate() {
        if intensity > params.min_intensity {
            if let Ok(room_id) = graph.get_room(node_id as crate::graph::NodeId) {
                reachable.insert(room_id);
            }
        }
    }
    
    Ok(reachable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_pvs_basic() {
        let mut pvs = PVS::new(3, 16);
        pvs.mark_visible(0, 2, 1).unwrap();
        pvs.mark_visible(0, 2, 2).unwrap();
        
        assert!(pvs.is_visible(0, 2, 1).unwrap());
        assert!(pvs.is_visible(0, 2, 2).unwrap());
        assert!(!pvs.is_visible(0, 2, 0).unwrap());
        assert!(!pvs.is_visible(1, 2, 1).unwrap());
    }
}
