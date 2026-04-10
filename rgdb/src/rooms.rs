use crate::graph::{Graph, NodeId, RoomId};

/// Room metadata structure
#[derive(Debug, Clone)]
pub struct Room {
    pub id: RoomId,
    /// Starting index of nodes in this room (for contiguous memory layout)
    pub node_start: usize,
    /// Number of nodes in this room
    pub node_count: usize,
}

/// Collection of rooms with metadata
#[derive(Debug, Clone)]
pub struct RoomCollection {
    pub rooms: Vec<Room>,
    /// Mapping from room_id to index in rooms vector
    pub room_index: Vec<usize>,
}

impl RoomCollection {
    /// Build room collection from room_map.
    /// 
    /// Assumes nodes within each room are contiguous in memory
    /// (this should be enforced by partitioning + reordering).
    pub fn from_room_map(room_map: &[RoomId]) -> Self {
        let _num_nodes = room_map.len();
        
        // Count nodes per room
        let mut room_counts: std::collections::HashMap<RoomId, usize> = 
            std::collections::HashMap::new();
        for &room_id in room_map {
            *room_counts.entry(room_id).or_insert(0) += 1;
        }
        
        // Build rooms
        let mut rooms = Vec::new();
        let mut room_to_index = std::collections::HashMap::new();
        let mut node_start = 0;
        
        for (room_id, &count) in &room_counts {
            let room = Room {
                id: *room_id,
                node_start,
                node_count: count,
            };
            room_to_index.insert(*room_id, rooms.len());
            rooms.push(room);
            node_start += count;
        }
        
        // Build room_index lookup
        let max_room_id = room_map.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut room_index = vec![usize::MAX; max_room_id];
        for (room_id, &idx) in &room_to_index {
            room_index[*room_id as usize] = idx;
        }
        
        Self { rooms, room_index }
    }
    
    /// Get room metadata by room_id
    pub fn get_room(&self, room_id: RoomId) -> Option<&Room> {
        let idx = room_id as usize;
        if idx < self.room_index.len() {
            let room_idx = self.room_index[idx];
            if room_idx != usize::MAX {
                return self.rooms.get(room_idx);
            }
        }
        None
    }
    
    /// Get number of rooms
    pub fn num_rooms(&self) -> usize {
        self.rooms.len()
    }
}

/// Detect portal edges (edges that cross room boundaries).
///
/// Returns a list of (u, v) pairs that are portals.
pub fn detect_portals(graph: &Graph) -> Vec<(NodeId, NodeId)> {
    let mut portals = Vec::new();
    
    for u in 0..graph.num_nodes() {
        let u_room = graph.get_room(u as NodeId).unwrap_or(0);
        for (v, _eprops) in graph.neighbors(u as NodeId) {
            let v_room = graph.get_room(v).unwrap_or(0);
            if u_room != v_room {
                portals.push((u as NodeId, v));
            }
        }
    }
    
    portals
}

/// Mark portal edges in the graph.
///
/// Note: This requires mutable access to edge properties, which is not
/// directly available through the current Graph API. This function is
/// kept for compatibility but may need to be refactored.
pub fn mark_portals(_graph: &mut Graph) {
    // TODO: Implement when Graph provides mutable edge access
    // For now, portals are detected but not marked in the graph structure
    // The is_portal flag in EdgeProps is set during graph construction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, EdgeProps, NodeProps};

    #[test]
    fn test_portal_detection() {
        // Create graph with two rooms: [0,1] and [2,3]
        let adj = vec![
            vec![(1, EdgeProps::default())], // Node 0 -> Node 1
            vec![(2, EdgeProps::default())], // Node 1 -> Node 2 (portal)
            vec![(3, EdgeProps::default())], // Node 2 -> Node 3
            vec![], // Node 3
        ];
        let mut graph = Graph::from_adjacency(4, adj, NodeProps::default()).unwrap();
        
        // Set room assignments
        graph.set_room(0, 0).unwrap();
        graph.set_room(1, 0).unwrap();
        graph.set_room(2, 1).unwrap();
        graph.set_room(3, 1).unwrap();
        
        let portals = detect_portals(&graph);
        assert_eq!(portals.len(), 1);
        assert_eq!(portals[0], (1, 2));
    }
}
