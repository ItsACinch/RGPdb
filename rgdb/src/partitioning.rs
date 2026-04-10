use crate::graph::{Graph, NodeId, RoomId};

/// Partitioning algorithm options
#[derive(Debug, Clone, Copy)]
pub enum PartitioningAlgorithm {
    /// Simple connected components (each component is a room)
    ConnectedComponents,
    /// BFS-based partitioning with target room size
    BFSPartitioning { target_room_size: usize },
    /// Manual partitioning (user provides room assignments)
    Manual,
}

/// Partition a graph into rooms using the specified algorithm.
///
/// Returns a room_map: Vec<RoomId> where room_map[node_id] = room_id
pub fn partition_graph(
    graph: &Graph,
    algorithm: PartitioningAlgorithm,
) -> Vec<RoomId> {
    match algorithm {
        PartitioningAlgorithm::ConnectedComponents => {
            partition_connected_components(graph)
        }
        PartitioningAlgorithm::BFSPartitioning { target_room_size } => {
            partition_bfs(graph, target_room_size)
        }
        PartitioningAlgorithm::Manual => {
            // Return existing room_map
            graph.room_map().to_vec()
        }
    }
}

/// Partition using connected components (each component becomes a room).
fn partition_connected_components(graph: &Graph) -> Vec<RoomId> {
    // Simple DFS-based connected components
    let num_nodes = graph.num_nodes();
    let mut room_map = vec![RoomId::MAX; num_nodes];
    let mut visited = vec![false; num_nodes];
    let mut current_room: RoomId = 0;
    
    // DFS helper function (iterative to avoid stack overflow)
    fn dfs_iterative(
        graph: &Graph,
        start_node: NodeId,
        room: RoomId,
        room_map: &mut Vec<RoomId>,
        visited: &mut Vec<bool>,
    ) {
        let mut stack = vec![start_node];
        
        while let Some(node) = stack.pop() {
            let node_idx = node as usize;
            if node_idx >= visited.len() || visited[node_idx] {
                continue;
            }
            visited[node_idx] = true;
            if node_idx < room_map.len() {
                room_map[node_idx] = room;
            }
            
            // Visit all neighbors
            for (neighbor, _) in graph.neighbors(node) {
                let neighbor_idx = neighbor as usize;
                if neighbor_idx < visited.len() && !visited[neighbor_idx] {
                    stack.push(neighbor);
                }
            }
        }
    }
    
    // Find all connected components
    for start_node in 0..num_nodes {
        if !visited[start_node] {
            dfs_iterative(graph, start_node as NodeId, current_room, &mut room_map, &mut visited);
            current_room = current_room.checked_add(1).unwrap_or(RoomId::MAX);
            if current_room == RoomId::MAX {
                break; // Prevent overflow
            }
        }
    }
    
    room_map
}

/// Partition using BFS with target room size.
/// 
/// This creates rooms by doing BFS from unassigned nodes until
/// we reach the target room size, then starting a new room.
fn partition_bfs(graph: &Graph, target_room_size: usize) -> Vec<RoomId> {
    let num_nodes = graph.num_nodes();
    let mut room_map = vec![RoomId::MAX; num_nodes];
    let mut current_room: RoomId = 0;
    let mut visited = vec![false; num_nodes];
    
    // Validate target_room_size
    let target_room_size = target_room_size.max(1); // At least 1 node per room
    
    for start_node in 0..num_nodes {
        if visited[start_node] {
            continue;
        }
        
        // BFS from this node (using VecDeque for proper queue behavior)
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();
        queue.push_back(start_node as NodeId);
        let mut room_nodes = Vec::new();
        visited[start_node] = true;
        
        while let Some(u) = queue.pop_front() {
            room_nodes.push(u);
            
            // Stop if we've reached target size
            if room_nodes.len() >= target_room_size {
                break;
            }
            
            // Add unvisited neighbors
            for (v, _) in graph.neighbors(u) {
                let v_idx = v as usize;
                if v_idx < visited.len() && !visited[v_idx] {
                    visited[v_idx] = true;
                    queue.push_back(v);
                }
            }
        }
        
        // Assign all nodes in this BFS tree to current_room
        for node in &room_nodes {
            let node_idx = *node as usize;
            if node_idx < room_map.len() {
                room_map[node_idx] = current_room;
            }
        }
        
        current_room = current_room.checked_add(1).unwrap_or(RoomId::MAX);
        if current_room == RoomId::MAX {
            break; // Prevent overflow
        }
    }
    
    // Assign any remaining unvisited nodes to the last room
    for i in 0..num_nodes {
        if room_map[i] == RoomId::MAX {
            room_map[i] = current_room;
        }
    }
    
    room_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, EdgeProps, NodeProps};

    #[test]
    fn test_connected_components() {
        // Create a graph with two disconnected components using from_adjacency
        let adj = vec![
            vec![(1, EdgeProps::default())], // Node 0 -> Node 1
            vec![(0, EdgeProps::default())], // Node 1 -> Node 0
            vec![(3, EdgeProps::default())], // Node 2 -> Node 3
            vec![(2, EdgeProps::default())], // Node 3 -> Node 2
        ];
        let graph = Graph::from_adjacency(4, adj, NodeProps::default()).unwrap();
        
        let room_map = partition_graph(&graph, PartitioningAlgorithm::ConnectedComponents);
        
        // Nodes 0 and 1 should be in same room
        assert_eq!(room_map[0], room_map[1]);
        // Nodes 2 and 3 should be in same room
        assert_eq!(room_map[2], room_map[3]);
        // But different rooms
        assert_ne!(room_map[0], room_map[2]);
    }
}

