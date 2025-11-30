/// Core graph structures for RGDB

use crate::error::{GraphError, node_id_to_usize};

pub type NodeId = u32;
pub type AngleBin = u8;
pub type RoomId = u32;

/// Number of discrete angle bins (2D "semantic directions").
pub const N_ANGLE_BINS: usize = 16;

/// Maximum allowed nodes in a graph (prevents DoS)
const MAX_NODES: usize = 100_000_000;
/// Maximum allowed edges in a graph (prevents DoS)
const MAX_EDGES: usize = 1_000_000_000;

/// Physical-ish properties of a node.
#[derive(Debug, Clone, Copy)]
pub struct NodeProps {
    /// Intrinsic emission from this node.
    pub luminance: f32,
    /// Fraction of incoming intensity that gets re-emitted.
    pub reflection: f32,
    /// Refraction index controlling how strongly direction changes are penalized.
    pub refraction_index: f32,
    /// Default angle bin for this node (used for PVS computation).
    pub default_angle_bin: AngleBin,
}

impl Default for NodeProps {
    fn default() -> Self {
        Self {
            luminance: 0.0,
            reflection: 1.0,
            refraction_index: 1.0,
            default_angle_bin: 0,
        }
    }
}

/// Physical-ish properties of an edge.
#[derive(Debug, Clone, Copy)]
pub struct EdgeProps {
    /// Fraction of intensity lost along this edge.
    pub attenuation: f32,
    /// Discrete direction bin for this relationship.
    pub angle_bin: AngleBin,
    /// Whether this edge is a portal (crosses room boundaries).
    pub is_portal: bool,
}

impl Default for EdgeProps {
    fn default() -> Self {
        Self {
            attenuation: 0.0,
            angle_bin: 0,
            is_portal: false,
        }
    }
}

/// Simple CSR graph layout with node + edge properties.
///
/// row_ptr[u]..row_ptr[u+1] indexes into col_idx / edge_props
///
/// # Thread Safety
/// `Graph` is `Send` but not `Sync`. Multiple threads can have their own
/// `Graph` instances, but sharing a single `Graph` across threads requires
/// external synchronization (e.g., `Arc<Mutex<Graph>>`).
#[derive(Debug, Clone)]
pub struct Graph {
    num_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<NodeId>,
    node_props: Vec<NodeProps>,
    edge_props: Vec<EdgeProps>,
    /// Room assignment for each node (room_map[node_id] = room_id)
    room_map: Vec<RoomId>,
}

impl Graph {
    /// Create an empty graph with a fixed number of nodes.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidSize` if `num_nodes` is 0 or exceeds `MAX_NODES`.
    pub fn new(num_nodes: usize) -> Result<Self, GraphError> {
        if num_nodes == 0 {
            return Err(GraphError::InvalidSize("num_nodes must be > 0".to_string()));
        }
        if num_nodes > MAX_NODES {
            return Err(GraphError::InvalidSize(format!(
                "num_nodes {} exceeds maximum {}", num_nodes, MAX_NODES
            )));
        }
        
        Ok(Self {
            num_nodes,
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            node_props: vec![NodeProps::default(); num_nodes],
            edge_props: Vec::new(),
            room_map: vec![0; num_nodes], // All nodes in room 0 by default
        })
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Get read-only access to row pointers
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Get read-only access to column indices
    pub fn col_idx(&self) -> &[NodeId] {
        &self.col_idx
    }

    /// Get read-only access to node properties
    pub fn node_props(&self) -> &[NodeProps] {
        &self.node_props
    }

    /// Get mutable access to node properties
    pub fn node_props_mut(&mut self) -> &mut [NodeProps] {
        &mut self.node_props
    }

    /// Get read-only access to edge properties
    pub fn edge_props(&self) -> &[EdgeProps] {
        &self.edge_props
    }

    /// Get read-only access to room map
    pub fn room_map(&self) -> &[RoomId] {
        &self.room_map
    }

    /// Set properties for a node.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidNodeId` if `u` is out of bounds.
    pub fn set_node_props(&mut self, u: NodeId, props: NodeProps) -> Result<(), GraphError> {
        let idx = node_id_to_usize(u)?;
        if idx >= self.num_nodes {
            return Err(GraphError::InvalidNodeId(u, self.num_nodes));
        }
        self.node_props[idx] = props;
        Ok(())
    }

    /// Set room assignment for a node.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidNodeId` if `u` is out of bounds.
    pub fn set_room(&mut self, u: NodeId, room: RoomId) -> Result<(), GraphError> {
        let idx = node_id_to_usize(u)?;
        if idx >= self.num_nodes {
            return Err(GraphError::InvalidNodeId(u, self.num_nodes));
        }
        if idx >= self.room_map.len() {
            return Err(GraphError::CorruptedStructure(format!(
                "room_map length {} < node index {}", self.room_map.len(), idx
            )));
        }
        self.room_map[idx] = room;
        Ok(())
    }

    /// Get room assignment for a node.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidNodeId` if `u` is out of bounds.
    pub fn get_room(&self, u: NodeId) -> Result<RoomId, GraphError> {
        let idx = node_id_to_usize(u)?;
        if idx >= self.room_map.len() {
            return Err(GraphError::InvalidNodeId(u, self.num_nodes));
        }
        Ok(self.room_map[idx])
    }

    /// Build graph from CSR (Compressed Sparse Row) format.
    ///
    /// This is useful for reconstructing graphs from serialized data.
    ///
    /// # Errors
    /// Returns `GraphError` if:
    /// - `num_nodes` is invalid
    /// - CSR structure is invalid (row_ptr length, col_idx/edge_props mismatch, etc.)
    /// - Any node ID in edges is invalid
    pub fn from_csr(
        num_nodes: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<NodeId>,
        node_props: Vec<NodeProps>,
        edge_props: Vec<EdgeProps>,
        room_map: Vec<RoomId>,
    ) -> Result<Self, GraphError> {
        if num_nodes == 0 {
            return Err(GraphError::InvalidSize("num_nodes must be > 0".to_string()));
        }
        if row_ptr.len() != num_nodes + 1 {
            return Err(GraphError::CorruptedStructure(format!(
                "row_ptr length {} != num_nodes + 1 ({})", row_ptr.len(), num_nodes + 1
            )));
        }
        if col_idx.len() != edge_props.len() {
            return Err(GraphError::CorruptedStructure(format!(
                "col_idx length {} != edge_props length {}", col_idx.len(), edge_props.len()
            )));
        }
        if node_props.len() != num_nodes {
            return Err(GraphError::CorruptedStructure(format!(
                "node_props length {} != num_nodes {}", node_props.len(), num_nodes
            )));
        }
        if room_map.len() != num_nodes {
            return Err(GraphError::CorruptedStructure(format!(
                "room_map length {} != num_nodes {}", room_map.len(), num_nodes
            )));
        }
        
        let graph = Self {
            num_nodes,
            row_ptr,
            col_idx,
            node_props,
            edge_props,
            room_map,
        };
        
        // Validate the constructed graph
        graph.validate()?;
        
        Ok(graph)
    }

    /// Build graph from adjacency list with edges (v, EdgeProps).
    ///
    /// # Errors
    /// Returns `GraphError` if:
    /// - `num_nodes` is invalid
    /// - `adj.len() != num_nodes`
    /// - Total edges exceed `MAX_EDGES`
    /// - Any node ID in edges is invalid
    pub fn from_adjacency(
        num_nodes: usize,
        adj: Vec<Vec<(NodeId, EdgeProps)>>,
        default_node_props: NodeProps,
    ) -> Result<Self, GraphError> {
        if num_nodes == 0 {
            return Err(GraphError::InvalidSize("num_nodes must be > 0".to_string()));
        }
        if num_nodes != adj.len() {
            return Err(GraphError::AdjacencyLengthMismatch(num_nodes, adj.len()));
        }
        
        // Validate total edge count
        let total_edges: usize = adj.iter().map(|v| v.len()).sum();
        if total_edges > MAX_EDGES {
            return Err(GraphError::InvalidSize(format!(
                "Total edges {} exceeds maximum {}", total_edges, MAX_EDGES
            )));
        }

        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);

        let mut col_idx = Vec::new();
        let mut edge_props = Vec::new();

        for (u, neighbors) in adj.iter().enumerate() {
            for (v, ep) in neighbors {
                // Validate neighbor node ID
                let v_idx = node_id_to_usize(*v)?;
                if v_idx >= num_nodes {
                    return Err(GraphError::CorruptedStructure(format!(
                        "Invalid neighbor node ID {} in adjacency list for node {}", v, u
                    )));
                }
                col_idx.push(*v);
                edge_props.push(*ep);
            }
            row_ptr.push(col_idx.len());
        }

        let node_props = vec![default_node_props; num_nodes];

        let graph = Self {
            num_nodes,
            row_ptr,
            col_idx,
            node_props,
            edge_props,
            room_map: vec![0; num_nodes],
        };

        // Validate the constructed graph
        graph.validate()?;

        Ok(graph)
    }

    /// Validate graph structure integrity.
    ///
    /// Checks:
    /// - CSR structure consistency
    /// - Node ID validity in edges
    /// - Room map consistency
    ///
    /// # Errors
    /// Returns `GraphError::CorruptedStructure` if any validation fails.
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check row_ptr is monotonic
        for i in 0..self.num_nodes {
            if self.row_ptr[i] > self.row_ptr[i + 1] {
                return Err(GraphError::CorruptedStructure(format!(
                    "row_ptr not monotonic at index {}", i
                )));
            }
        }
        
        // Check row_ptr[0] == 0
        if self.row_ptr[0] != 0 {
            return Err(GraphError::CorruptedStructure(
                "row_ptr[0] must be 0".to_string(),
            ));
        }
        
        // Check row_ptr[num_nodes] == num_edges
        if self.row_ptr[self.num_nodes] != self.col_idx.len() {
            return Err(GraphError::CorruptedStructure(format!(
                "row_ptr[{}] = {} but col_idx.len() = {}",
                self.num_nodes, self.row_ptr[self.num_nodes], self.col_idx.len()
            )));
        }
        
        // Check all node IDs are valid
        for &node_id in &self.col_idx {
            let node_idx = node_id_to_usize(node_id)?;
            if node_idx >= self.num_nodes {
                return Err(GraphError::CorruptedStructure(format!(
                    "Invalid node ID in edge: {}", node_id
                )));
            }
        }
        
        // Check room_map consistency
        if self.room_map.len() != self.num_nodes {
            return Err(GraphError::CorruptedStructure(format!(
                "room_map length {} != num_nodes {}",
                self.room_map.len(), self.num_nodes
            )));
        }
        
        // Check edge_props length matches col_idx
        if self.edge_props.len() != self.col_idx.len() {
            return Err(GraphError::CorruptedStructure(format!(
                "edge_props length {} != col_idx length {}",
                self.edge_props.len(), self.col_idx.len()
            )));
        }
        
        Ok(())
    }

    /// Get outgoing edges of node u as (neighbor_id, &EdgeProps)
    ///
    /// # Panics
    /// Panics if `u` is out of bounds. Use `get_room()` for safe access.
    pub fn neighbors(&self, u: NodeId) -> impl Iterator<Item = (NodeId, &EdgeProps)> {
        let idx = u as usize;
        // Bounds check - in hot path, we use debug_assert for performance
        debug_assert!(idx < self.num_nodes, "Node ID {} out of bounds", u);
        debug_assert!(idx < self.row_ptr.len() - 1, "Node index {} >= row_ptr.len() - 1", idx);
        
        let start = self.row_ptr[idx];
        let end = self.row_ptr[idx + 1];
        
        // Additional bounds check
        debug_assert!(end <= self.col_idx.len(), "row_ptr[{}] = {} > col_idx.len()", idx + 1, end);
        debug_assert!(end <= self.edge_props.len(), "row_ptr[{}] = {} > edge_props.len()", idx + 1, end);
        
        self.col_idx[start..end]
            .iter()
            .copied()
            .zip(self.edge_props[start..end].iter())
    }

    /// Check if edge (u, v) exists and return its properties
    ///
    /// # Errors
    /// Returns `None` if edge doesn't exist or if `u` is out of bounds.
    pub fn get_edge(&self, u: NodeId, v: NodeId) -> Option<&EdgeProps> {
        let u_idx = match node_id_to_usize(u) {
            Ok(idx) if idx < self.num_nodes => idx,
            _ => return None,
        };
        
        if u_idx >= self.row_ptr.len() - 1 {
            return None;
        }
        
        let start = self.row_ptr[u_idx];
        let end = self.row_ptr[u_idx + 1];
        
        if end > self.col_idx.len() || end > self.edge_props.len() {
            return None;
        }
        
        for (i, &neighbor) in self.col_idx[start..end].iter().enumerate() {
            if neighbor == v {
                let edge_idx = start + i;
                if edge_idx < self.edge_props.len() {
                    return Some(&self.edge_props[edge_idx]);
                }
            }
        }
        None
    }
}
