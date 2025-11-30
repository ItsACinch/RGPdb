# Getting Started with RGDB

This guide provides practical examples for using the Refractive Graph Database (RGDB).

---

## Table of Contents
- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Adding Data](#adding-data)
- [Querying Data](#querying-data)
- [Updating Data](#updating-data)
- [Saving and Loading](#saving-and-loading)
- [Deleting Data](#deleting-data)
- [Advanced Examples](#advanced-examples)

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RGP

# Build the project
cargo build --release

# Run tests
cargo test
```

---

## Basic Concepts

RGDB uses a **graph structure** where:
- **Nodes** represent entities (documents, concepts, etc.)
- **Edges** represent relationships with directional semantics
- **Rooms** partition the graph for efficient queries
- **PVS** (Potentially Visible Sets) optimize propagation

Each node has:
- `luminance`: Intrinsic emission strength
- `reflection`: How much influence it re-emits
- `refraction_index`: How strongly direction changes are penalized
- `default_angle_bin`: Default direction for this node

Each edge has:
- `attenuation`: Loss along the edge (0.0 = no loss, 1.0 = complete loss)
- `angle_bin`: Direction of the relationship (0-15 for 16 bins)
- `is_portal`: Whether this edge crosses room boundaries

---

## Adding Data

### Example 1: Creating a Simple Graph

```rust
use rgdb::*;

// Create an empty graph with 5 nodes
let mut graph = graph::Graph::new(5);

// Set node properties
graph.set_node_props(0, graph::NodeProps {
    luminance: 2.0,           // Bright source node
    reflection: 0.9,          // Highly reflective
    refraction_index: 1.0,     // Low refraction (easy direction changes)
    default_angle_bin: 2,     // Default direction: bin 2
});

graph.set_node_props(1, graph::NodeProps {
    luminance: 0.5,
    reflection: 0.8,
    refraction_index: 2.0,    // Higher refraction (direction changes penalized)
    default_angle_bin: 2,
});

// Add edges manually (for small graphs)
// Note: For larger graphs, use from_adjacency() or build programmatically
```

### Example 2: Building from Adjacency List

```rust
use rgdb::*;

let num_nodes = 4;
let default_props = graph::NodeProps {
    luminance: 1.0,
    reflection: 0.9,
    refraction_index: 1.5,
    default_angle_bin: 0,
};

// Create adjacency list: node -> [(neighbor, edge_props), ...]
let adj = vec![
    // Node 0: connects to nodes 1 and 2
    vec![
        (1, graph::EdgeProps {
            attenuation: 0.1,
            angle_bin: 2,
            is_portal: false,
        }),
        (2, graph::EdgeProps {
            attenuation: 0.2,
            angle_bin: 4,
            is_portal: false,
        }),
    ],
    // Node 1: connects to node 3
    vec![
        (3, graph::EdgeProps {
            attenuation: 0.15,
            angle_bin: 2,  // Same direction as incoming
            is_portal: false,
        }),
    ],
    // Node 2: connects to node 3
    vec![
        (3, graph::EdgeProps {
            attenuation: 0.1,
            angle_bin: 8,  // Different direction (will be refracted)
            is_portal: false,
        }),
    ],
    // Node 3: no outgoing edges
    vec![],
];

let graph = graph::Graph::from_adjacency(num_nodes, adj, default_props);
```

### Example 3: Programmatically Building a Large Graph

```rust
use rgdb::*;

fn build_knowledge_graph(num_documents: usize) -> graph::Graph {
    let mut graph = graph::Graph::new(num_documents);
    
    // Set properties for each document
    for i in 0..num_documents {
        graph.set_node_props(i as graph::NodeId, graph::NodeProps {
            luminance: 1.0 + (i as f32) * 0.1,  // Varying brightness
            reflection: 0.8,
            refraction_index: 1.0 + (i % 3) as f32 * 0.5,
            default_angle_bin: (i % 16) as graph::AngleBin,
        });
    }
    
    // Add edges based on similarity or relationships
    // This is a simplified example - in practice, you'd compute
    // relationships from embeddings, semantic analysis, etc.
    
    // For now, we'll need to manually build the CSR structure
    // In a full implementation, you'd have helper methods for this
    
    graph
}
```

### Example 4: Adding Edges to Existing Graph

```rust
use rgdb::*;

// Note: The current Graph structure uses CSR format which is immutable
// after construction. To add edges, you need to rebuild the graph.
// Here's a helper function to add edges:

fn add_edge_to_graph(
    graph: &mut graph::Graph,
    from: graph::NodeId,
    to: graph::NodeId,
    edge_props: graph::EdgeProps,
) {
    // Find insertion point in CSR structure
    let from_idx = from as usize;
    let insert_pos = graph.row_ptr[from_idx + 1];
    
    // Insert edge
    graph.col_idx.insert(insert_pos, to);
    graph.edge_props.insert(insert_pos, edge_props);
    
    // Update row pointers for all subsequent nodes
    for i in (from_idx + 1)..=graph.num_nodes {
        graph.row_ptr[i] += 1;
    }
}

// Usage:
let mut graph = graph::Graph::new(5);
add_edge_to_graph(
    &mut graph,
    0,
    1,
    graph::EdgeProps {
        attenuation: 0.1,
        angle_bin: 2,
        is_portal: false,
    },
);
```

---

## Querying Data

### Example 1: Top-K Influence Query

Find the top 10 nodes most influenced by a source node:

```rust
use rgdb::*;

let graph = /* your graph */;
let params = propagation::LightParams::default();

// Query top 10 nodes influenced by node 0, starting with angle bin 2
let top_k = queries::query_top_k_influence(
    &graph,
    0,              // source node
    2,              // initial angle bin
    10,             // k (top 10)
    params,
    None,           // PVS (optional)
);

println!("Top 10 most influenced nodes:");
for (i, result) in top_k.iter().enumerate() {
    println!("  {}. Node {}: intensity={:.6}, distance={:.6}",
        i + 1, result.node, result.intensity, result.distance);
}
```

### Example 2: Distance Query

Find the "light distance" between two specific nodes:

```rust
use rgdb::*;

let graph = /* your graph */;
let params = propagation::LightParams::default();

// Query distance from node 0 to node 5
if let Some(distance) = queries::query_distance(
    &graph,
    0,              // source
    5,              // target
    2,              // initial angle bin
    params,
    None,           // PVS
) {
    println!("Distance from node 0 to node 5: {:.6}", distance);
} else {
    println!("Node 5 is not reachable from node 0");
}
```

### Example 3: Hybrid Query (Graph + Vector Similarity)

Combine graph propagation with vector embeddings:

```rust
use rgdb::*;
use ndarray::Array1;

let graph = /* your graph */;
let params = propagation::LightParams::default();

// Prepare embeddings (one per node)
let mut embeddings = Vec::new();
for i in 0..graph.num_nodes {
    // In practice, these would come from your embedding model
    let emb = Array1::from_vec(vec![0.1 * i as f32; 128]); // 128-dim embeddings
    embeddings.push(emb);
}

// Query embedding (what you're looking for)
let query_emb = Array1::from_vec(vec![1.0; 128]);

// Hybrid query: 70% graph influence, 30% vector similarity
let results = queries::query_hybrid(
    &graph,
    0,              // source node
    2,              // initial angle bin
    &embeddings,
    &query_emb,
    0.7,            // alpha: weight for graph (1-alpha for vectors)
    10,             // top 10
    params,
    None,
);

println!("Hybrid query results:");
for result in &results {
    println!("  Node {}: score={:.6}", result.node, result.intensity);
}
```

### Example 4: Query with PVS Optimization

Use PVS (Potentially Visible Sets) for faster queries on large graphs:

```rust
use rgdb::*;

let mut graph = /* your graph */;
let params = propagation::LightParams::default();

// Partition graph into rooms
let room_map = partitioning::partition_graph(
    &graph,
    partitioning::PartitioningAlgorithm::BFSPartitioning {
        target_room_size: 100,  // ~100 nodes per room
    },
);
graph.room_map = room_map;

// Build room collection
let rooms = rooms::RoomCollection::from_room_map(&graph.room_map);

// Compute PVS for optimization
let pvs = pvs::compute_pvs(&graph, &rooms, &params);

// Query with PVS (much faster on large graphs)
let top_k = queries::query_top_k_influence(
    &graph,
    0,
    2,
    10,
    params,
    Some(&pvs),  // Use PVS for pruning
);
```

---

## Updating Data

### Example 1: Update Node Properties

```rust
use rgdb::*;

let mut graph = /* your graph */;

// Update a node's properties
graph.set_node_props(3, graph::NodeProps {
    luminance: 5.0,          // Make it brighter
    reflection: 0.95,        // More reflective
    refraction_index: 0.5,    // Less refraction (easier direction changes)
    default_angle_bin: 4,    // Change default direction
});
```

### Example 2: Update Edge Properties

```rust
use rgdb::*;

let mut graph = /* your graph */;

// Find and update an edge
if let Some(edge) = graph.get_edge(0, 1) {
    // Get mutable reference to edge (requires accessing internal structure)
    let from_idx = 0 as usize;
    let start = graph.row_ptr[from_idx];
    let end = graph.row_ptr[from_idx + 1];
    
    for i in start..end {
        if graph.col_idx[i] == 1 {
            // Update edge properties
            graph.edge_props[i].attenuation = 0.05;  // Less loss
            graph.edge_props[i].angle_bin = 3;       // Change direction
            break;
        }
    }
}
```

### Example 3: Recompute PVS After Updates

```rust
use rgdb::*;

let mut graph = /* your graph */;
let rooms = /* your rooms */;
let params = propagation::LightParams::default();

// After modifying graph structure or properties,
// recompute PVS for optimal performance
let updated_pvs = pvs::compute_pvs(&graph, &rooms, &params);
```

### Example 4: Incremental Graph Updates

```rust
use rgdb::*;

// For incremental updates, you can:
// 1. Track which nodes/edges changed
// 2. Only recompute PVS for affected rooms
// 3. Update propagation results incrementally

fn update_graph_incrementally(
    graph: &mut graph::Graph,
    changed_nodes: &[graph::NodeId],
    changed_edges: &[(graph::NodeId, graph::NodeId)],
) {
    // Update node properties
    for &node_id in changed_nodes {
        // Your update logic here
        graph.set_node_props(node_id, graph::NodeProps {
            luminance: 2.0,
            reflection: 0.9,
            refraction_index: 1.0,
            default_angle_bin: 0,
        });
    }
    
    // Update edge properties
    for &(from, to) in changed_edges {
        if let Some(_edge) = graph.get_edge(from, to) {
            // Update edge (see Example 2)
        }
    }
}
```

---

## Saving and Loading

### Example 1: Save Graph to File

```rust
use rgdb::*;

let graph = /* your graph */;
let rooms = /* your rooms */;
let pvs = /* your PVS */;

// Save to level file
level_file::write_level_file(
    &graph,
    &rooms,
    &pvs,
    "my_graph.rgdb",
)?;

println!("Graph saved to my_graph.rgdb");
```

### Example 2: Load Graph from File

```rust
use rgdb::*;

// Load using memory-mapped I/O (fast, zero-copy)
let (graph, rooms, pvs) = level_file::read_level_file_mmap("my_graph.rgdb")?;

println!("Loaded graph with {} nodes, {} edges, {} rooms",
    graph.num_nodes,
    graph.num_edges(),
    rooms.num_rooms());
```

### Example 3: Complete Save/Load Workflow

```rust
use rgdb::*;
use std::path::Path;

fn save_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create and populate graph
    let mut graph = graph::Graph::new(1000);
    // ... populate graph ...
    
    // 2. Partition into rooms
    let room_map = partitioning::partition_graph(
        &graph,
        partitioning::PartitioningAlgorithm::ConnectedComponents,
    );
    graph.room_map = room_map;
    
    // 3. Build rooms and compute PVS
    let rooms = rooms::RoomCollection::from_room_map(&graph.room_map);
    let params = propagation::LightParams::default();
    let pvs = pvs::compute_pvs(&graph, &rooms, &params);
    
    // 4. Mark portals
    rooms::mark_portals(&mut graph);
    
    // 5. Save everything
    level_file::write_level_file(&graph, &rooms, &pvs, "database.rgdb")?;
    
    Ok(())
}

fn load_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Load everything from file
    let (graph, rooms, pvs) = level_file::read_level_file_mmap("database.rgdb")?;
    
    // Verify loaded data
    println!("Loaded: {} nodes, {} edges", graph.num_nodes, graph.num_edges());
    
    // Use the loaded graph
    let intensities = propagation::propagate_light_with_pvs(
        &graph,
        0,
        2,
        propagation::LightParams::default(),
        Some(&pvs),
    );
    
    Ok(())
}
```

---

## Deleting Data

### Example 1: Remove an Edge

```rust
use rgdb::*;

fn remove_edge(
    graph: &mut graph::Graph,
    from: graph::NodeId,
    to: graph::NodeId,
) {
    let from_idx = from as usize;
    let start = graph.row_ptr[from_idx];
    let end = graph.row_ptr[from_idx + 1];
    
    // Find the edge
    if let Some(pos) = graph.col_idx[start..end]
        .iter()
        .position(|&n| n == to)
    {
        let remove_idx = start + pos;
        
        // Remove from CSR structure
        graph.col_idx.remove(remove_idx);
        graph.edge_props.remove(remove_idx);
        
        // Update row pointers
        for i in (from_idx + 1)..=graph.num_nodes {
            graph.row_ptr[i] -= 1;
        }
    }
}

// Usage:
let mut graph = /* your graph */;
remove_edge(&mut graph, 0, 1);
```

### Example 2: Remove a Node (Isolate It)

```rust
use rgdb::*;

fn isolate_node(graph: &mut graph::Graph, node_id: graph::NodeId) {
    let node_idx = node_id as usize;
    
    // Remove all outgoing edges
    let start = graph.row_ptr[node_idx];
    let end = graph.row_ptr[node_idx + 1];
    let num_edges_to_remove = end - start;
    
    // Remove edges
    graph.col_idx.drain(start..end);
    graph.edge_props.drain(start..end);
    
    // Update row pointers
    for i in (node_idx + 1)..=graph.num_nodes {
        graph.row_ptr[i] -= num_edges_to_remove;
    }
    
    // Remove incoming edges (edges pointing TO this node)
    for u in 0..graph.num_nodes {
        if u == node_idx {
            continue;
        }
        
        let u_start = graph.row_ptr[u];
        let u_end = graph.row_ptr[u + 1];
        
        // Find and remove edges to this node
        let mut to_remove = Vec::new();
        for i in u_start..u_end {
            if graph.col_idx[i] == node_id {
                to_remove.push(i);
            }
        }
        
        // Remove in reverse order to maintain indices
        for &idx in to_remove.iter().rev() {
            graph.col_idx.remove(idx);
            graph.edge_props.remove(idx);
            
            // Update row pointers for node u
            for j in (u + 1)..=graph.num_nodes {
                graph.row_ptr[j] -= 1;
            }
        }
    }
}

// Usage:
let mut graph = /* your graph */;
isolate_node(&mut graph, 5);  // Remove all edges to/from node 5
```

### Example 3: Compact Graph (Remove Isolated Nodes)

```rust
use rgdb::*;

fn compact_graph(graph: &mut graph::Graph) -> Vec<graph::NodeId> {
    // Find isolated nodes (no edges)
    let mut isolated = Vec::new();
    for i in 0..graph.num_nodes {
        let start = graph.row_ptr[i];
        let end = graph.row_ptr[i + 1];
        if start == end {
            // Check if any other node points to this one
            let mut has_incoming = false;
            for u in 0..graph.num_nodes {
                if u == i {
                    continue;
                }
                for (v, _) in graph.neighbors(u as graph::NodeId) {
                    if v == i as graph::NodeId {
                        has_incoming = true;
                        break;
                    }
                }
                if has_incoming {
                    break;
                }
            }
            if !has_incoming {
                isolated.push(i as graph::NodeId);
            }
        }
    }
    
    // Note: Actually removing nodes requires rebuilding the graph
    // This is a simplified example that just identifies them
    isolated
}

// Usage:
let mut graph = /* your graph */;
let isolated = compact_graph(&mut graph);
println!("Found {} isolated nodes", isolated.len());
```

---

## Advanced Examples

### Example 1: Complete Database Workflow

```rust
use rgdb::*;

fn complete_example() -> Result<(), Box<dyn std::error::Error>> {
    // === 1. CREATE DATABASE ===
    println!("Creating graph...");
    let mut graph = graph::Graph::new(100);
    
    // Add some nodes with properties
    for i in 0..100 {
        graph.set_node_props(i as graph::NodeId, graph::NodeProps {
            luminance: 1.0 + (i as f32) * 0.01,
            reflection: 0.8 + (i % 10) as f32 * 0.02,
            refraction_index: 1.0 + (i % 5) as f32 * 0.2,
            default_angle_bin: (i % 16) as graph::AngleBin,
        });
    }
    
    // Add edges (simplified - in practice, compute from your data)
    // ... (would need to build CSR structure properly)
    
    // === 2. PARTITION INTO ROOMS ===
    println!("Partitioning graph...");
    let room_map = partitioning::partition_graph(
        &graph,
        partitioning::PartitioningAlgorithm::BFSPartitioning {
            target_room_size: 20,
        },
    );
    graph.room_map = room_map;
    
    // === 3. BUILD ROOMS AND PVS ===
    println!("Building rooms and PVS...");
    let rooms = rooms::RoomCollection::from_room_map(&graph.room_map);
    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 6,
        num_angle_bins: 16,
    };
    let pvs = pvs::compute_pvs(&graph, &rooms, &params);
    
    // === 4. SAVE DATABASE ===
    println!("Saving database...");
    level_file::write_level_file(&graph, &rooms, &pvs, "database.rgdb")?;
    
    // === 5. QUERY DATABASE ===
    println!("Querying database...");
    let top_10 = queries::query_top_k_influence(
        &graph,
        0,  // source
        2,  // initial bin
        10, // top 10
        params,
        Some(&pvs),
    );
    
    println!("Top 10 results:");
    for (i, result) in top_10.iter().enumerate() {
        println!("  {}. Node {}: {:.6}", i + 1, result.node, result.intensity);
    }
    
    // === 6. UPDATE AND RE-SAVE ===
    println!("Updating node...");
    graph.set_node_props(5, graph::NodeProps {
        luminance: 10.0,  // Much brighter
        reflection: 0.95,
        refraction_index: 0.5,
        default_angle_bin: 2,
    });
    
    // Recompute PVS if needed (or use incremental update)
    let updated_pvs = pvs::compute_pvs(&graph, &rooms, &params);
    
    // Save updated version
    level_file::write_level_file(&graph, &rooms, &updated_pvs, "database_updated.rgdb")?;
    
    Ok(())
}
```

### Example 2: Using GPU Acceleration

```rust
use rgdb::*;

fn gpu_accelerated_query() -> Result<(), Box<dyn std::error::Error>> {
    let graph = /* your graph */;
    let params = propagation::LightParams::default();
    
    // Check if CUDA is available
    if cuda::CudaContext::is_available() {
        println!("CUDA available - using GPU acceleration");
        
        // Initialize CUDA context
        let ctx = cuda::CudaContext::new()?;
        
        if ctx.kernels_loaded() {
            // Run GPU-accelerated propagation
            let intensities = cuda::propagate_light_cuda(
                &ctx,
                &graph,
                0,      // source
                2,      // initial bin
                params,
                None,   // PVS (optional)
            )?;
            
            println!("GPU propagation complete!");
            println!("Intensities for first 10 nodes:");
            for (i, &intensity) in intensities.iter().take(10).enumerate() {
                println!("  Node {}: {:.6}", i, intensity);
            }
        } else {
            println!("CUDA kernels not loaded, falling back to CPU");
            let intensities = propagation::propagate_light(&graph, 0, 2, params);
        }
    } else {
        println!("CUDA not available - using CPU");
        let intensities = propagation::propagate_light(&graph, 0, 2, params);
    }
    
    Ok(())
}
```

### Example 3: Building a Knowledge Graph

```rust
use rgdb::*;

struct Document {
    id: usize,
    content: String,
    embedding: Vec<f32>,
}

fn build_knowledge_graph(documents: &[Document]) -> graph::Graph {
    let num_docs = documents.len();
    let mut graph = graph::Graph::new(num_docs);
    
    // Set node properties based on document importance
    for (i, doc) in documents.iter().enumerate() {
        // Luminance based on document importance/score
        let luminance = 1.0 + (doc.content.len() as f32) / 1000.0;
        
        // Default angle bin from embedding (simplified)
        let default_bin = (doc.embedding[0] * 8.0 + 8.0) as u8 % 16;
        
        graph.set_node_props(i as graph::NodeId, graph::NodeProps {
            luminance,
            reflection: 0.9,
            refraction_index: 1.0,
            default_angle_bin: default_bin,
        });
    }
    
    // Add edges based on similarity
    // In practice, you'd compute cosine similarity between embeddings
    // and add edges for similar documents
    
    graph
}

// Usage:
let documents = vec![
    Document { id: 0, content: "Document 1".to_string(), embedding: vec![0.1; 128] },
    Document { id: 1, content: "Document 2".to_string(), embedding: vec![0.2; 128] },
    // ... more documents
];

let graph = build_knowledge_graph(&documents);
```

### Example 4: Semantic Search with RGDB

```rust
use rgdb::*;
use ndarray::Array1;

fn semantic_search(
    graph: &graph::Graph,
    query_embedding: &Array1<f32>,
    document_embeddings: &[Array1<f32>],
    k: usize,
) -> Vec<queries::InfluenceResult> {
    let params = propagation::LightParams::default();
    
    // Use hybrid query combining graph structure and vector similarity
    queries::query_hybrid(
        graph,
        0,                      // Start from first document
        0,                      // Initial direction
        document_embeddings,
        query_embedding,
        0.6,                    // 60% graph, 40% vector
        k,
        params,
        None,
    )
}

// Usage:
let graph = /* your knowledge graph */;
let query_emb = Array1::from_vec(vec![0.5; 128]);
let doc_embeddings: Vec<Array1<f32>> = /* your document embeddings */;

let results = semantic_search(&graph, &query_emb, &doc_embeddings, 10);
println!("Top 10 semantically similar documents:");
for result in &results {
    println!("  Document {}: score {:.6}", result.node, result.intensity);
}
```

### Example 5: Incremental Updates with PVS

```rust
use rgdb::*;

fn incremental_update(
    graph: &mut graph::Graph,
    rooms: &rooms::RoomCollection,
    pvs: &mut pvs::PVS,
    updated_node: graph::NodeId,
) {
    // Update node properties
    graph.set_node_props(updated_node, graph::NodeProps {
        luminance: 5.0,
        reflection: 0.95,
        refraction_index: 1.0,
        default_angle_bin: 2,
    });
    
    // Find which room contains this node
    let room_id = graph.get_room(updated_node);
    
    // Incrementally update PVS for affected rooms only
    // (This is a simplified example - full implementation would
    //  only recompute PVS for rooms reachable from the updated node)
    let params = propagation::LightParams::default();
    
    // For now, recompute PVS for the affected room
    if let Some(room) = rooms.get_room(room_id) {
        // Recompute PVS entries for this room
        for angle_bin in 0..16 {
            let start_node = find_representative_node(graph, room);
            let reachable = find_reachable_rooms(
                graph,
                start_node,
                angle_bin,
                &params,
            );
            
            // Update PVS
            for &reachable_room in &reachable {
                pvs.mark_visible(room_id, angle_bin, reachable_room);
            }
        }
    }
}
```

---

## Common Patterns

### Pattern 1: Batch Operations

```rust
use rgdb::*;

fn batch_update_nodes(
    graph: &mut graph::Graph,
    updates: &[(graph::NodeId, graph::NodeProps)],
) {
    for (node_id, props) in updates {
        graph.set_node_props(*node_id, *props);
    }
}

// Usage:
let updates = vec![
    (0, graph::NodeProps { luminance: 2.0, ..Default::default() }),
    (1, graph::NodeProps { luminance: 3.0, ..Default::default() }),
    (2, graph::NodeProps { luminance: 1.5, ..Default::default() }),
];
batch_update_nodes(&mut graph, &updates);
```

### Pattern 2: Query Multiple Sources

```rust
use rgdb::*;

fn multi_source_query(
    graph: &graph::Graph,
    sources: &[graph::NodeId],
    k: usize,
) -> Vec<queries::InfluenceResult> {
    let params = propagation::LightParams::default();
    let mut all_results = Vec::new();
    
    for &source in sources {
        let results = queries::query_top_k_influence(
            graph,
            source,
            0,  // default angle bin
            k,
            params,
            None,
        );
        all_results.extend(results);
    }
    
    // Aggregate and return top-K across all sources
    all_results.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
    all_results.into_iter().take(k).collect()
}
```

### Pattern 3: Graph Validation

```rust
use rgdb::*;

fn validate_graph(graph: &graph::Graph) -> Result<(), String> {
    // Check CSR structure integrity
    if graph.row_ptr.len() != graph.num_nodes + 1 {
        return Err("row_ptr length mismatch".to_string());
    }
    
    if graph.col_idx.len() != graph.edge_props.len() {
        return Err("col_idx and edge_props length mismatch".to_string());
    }
    
    // Check row_ptr is monotonic
    for i in 0..graph.num_nodes {
        if graph.row_ptr[i] > graph.row_ptr[i + 1] {
            return Err(format!("row_ptr not monotonic at index {}", i));
        }
    }
    
    // Check all node IDs are valid
    for &node_id in &graph.col_idx {
        if node_id as usize >= graph.num_nodes {
            return Err(format!("Invalid node ID in edge: {}", node_id));
        }
    }
    
    // Check room_map consistency
    if graph.room_map.len() != graph.num_nodes {
        return Err("room_map length mismatch".to_string());
    }
    
    Ok(())
}
```

---

## Performance Tips

1. **Use PVS for Large Graphs**: Always compute and use PVS for graphs with > 10K nodes
2. **Batch Updates**: Update multiple nodes/edges at once, then recompute PVS
3. **GPU Acceleration**: Use CUDA for graphs with > 100K nodes
4. **Save Frequently**: Save your graph after major updates
5. **Room Size**: Target 50-500 nodes per room for optimal performance

---

## Error Handling

```rust
use rgdb::*;

fn robust_operation() -> Result<(), Box<dyn std::error::Error>> {
    // All operations return Results
    let graph = graph::Graph::new(100);
    
    // Handle errors gracefully
    match graph.set_node_props(150, graph::NodeProps::default()) {
        Ok(_) => println!("Updated node"),
        Err(e) => println!("Error: {}", e),
    }
    
    // Or use ? operator
    let intensities = propagation::propagate_light(
        &graph,
        0,
        2,
        propagation::LightParams::default(),
    );
    
    Ok(())
}
```

---

## Next Steps

- Read the [README.md](README.md) for conceptual overview
- Check [ACTION_PLAN.md](ACTION_PLAN.md) for implementation details
- Review [CODE_REVIEW.md](CODE_REVIEW.md) for best practices
- See [CUDA_COMPLETE.md](CUDA_COMPLETE.md) for GPU acceleration details

---

## Appendix: Key Terms and Concepts

This section provides detailed explanations of the core concepts and terminology used in RGDB.

### RGDB (Refractive Graph Database)

**RGDB** stands for **Refractive Graph Database**, a novel graph database that uses physics-inspired light propagation to model contextual, directional, multi-hop relationships between entities.

#### Core Philosophy

RGDB treats data relationships like light traveling through a physical medium:
- **Light** represents influence or information flow
- **Refraction** models how relationships change direction
- **Attenuation** represents loss of signal strength over distance
- **Reflection** models how entities re-emit or amplify influence

This approach enables:
- **Directional semantics**: Relationships have inherent directionality (angle bins)
- **Contextual similarity**: Multi-hop paths create contextual relationships
- **Natural ranking**: Light intensity naturally ranks influence strength
- **Efficient queries**: Physics-based pruning (PVS) optimizes large-scale queries

#### Key Advantages

1. **Multi-hop reasoning**: Finds relationships that span multiple connections
2. **Directional awareness**: Understands that relationships have semantic directions
3. **Contextual relevance**: Nodes are ranked by contextual influence, not just direct connections
4. **Scalability**: Room-based partitioning and PVS enable efficient queries on billion-node graphs

---

### Nodes

**Nodes** are the fundamental entities in an RGDB graph. Each node represents a discrete piece of information, such as:
- A document in a knowledge base
- A concept in a semantic network
- A user in a social network
- A product in a recommendation system
- Any entity you want to model relationships between

#### Node Properties

Each node has four key properties that control how it interacts with light propagation:

1. **`luminance`** (f32)
   - **Definition**: The intrinsic emission strength of the node
   - **Range**: Typically 0.0 to 10.0+ (no upper limit)
   - **Meaning**: How "bright" or "important" this node is as a source
   - **Example**: A highly-cited research paper might have `luminance: 5.0`, while a minor document has `luminance: 0.5`
   - **Impact**: Higher luminance means the node emits more influence, affecting more nodes in the graph

2. **`reflection`** (f32)
   - **Definition**: Fraction of incoming intensity that gets re-emitted
   - **Range**: 0.0 (no reflection) to 1.0 (perfect reflection)
   - **Meaning**: How much influence this node passes along to its neighbors
   - **Example**: A hub node (like Wikipedia) might have `reflection: 0.95`, while a terminal node has `reflection: 0.1`
   - **Impact**: High reflection means the node amplifies and propagates influence; low reflection means it absorbs influence

3. **`refraction_index`** (f32)
   - **Definition**: Controls how strongly direction changes are penalized
   - **Range**: Typically 0.5 (easy direction changes) to 5.0+ (strict direction)
   - **Meaning**: How "bendy" or "rigid" the node is to direction changes
   - **Example**: A general-purpose node might have `refraction_index: 1.0`, while a specialized node has `refraction_index: 3.0`
   - **Impact**: Higher refraction means light must maintain direction (semantic consistency); lower refraction allows more semantic flexibility

4. **`default_angle_bin`** (AngleBin, u8)
   - **Definition**: The default semantic direction for this node
   - **Range**: 0 to 15 (for 16 angle bins)
   - **Meaning**: The primary "direction" this node faces in semantic space
   - **Example**: A node about "science" might have `default_angle_bin: 2`, while "art" has `default_angle_bin: 10`
   - **Impact**: Used for PVS computation and initial propagation direction

#### Node Identification

- **`NodeId`**: A unique identifier (u32) for each node
- Nodes are indexed from 0 to `num_nodes - 1`
- Node IDs are stable within a graph instance

---

### Edges

**Edges** represent relationships between nodes. Unlike traditional graph databases, RGDB edges have **directional semantics** - they encode not just that a relationship exists, but *how* entities relate to each other.

#### Edge Properties

Each edge has three properties:

1. **`attenuation`** (f32)
   - **Definition**: Fraction of intensity lost along this edge
   - **Range**: 0.0 (no loss) to 1.0 (complete loss)
   - **Meaning**: How much signal strength is lost when traversing this relationship
   - **Example**: 
     - `attenuation: 0.0` = perfect relationship (no loss)
     - `attenuation: 0.1` = strong relationship (10% loss)
     - `attenuation: 0.9` = weak relationship (90% loss)
   - **Impact**: Lower attenuation means influence propagates more strongly; higher attenuation weakens the signal

2. **`angle_bin`** (AngleBin, u8)
   - **Definition**: The discrete semantic direction of this relationship
   - **Range**: 0 to 15 (for 16 angle bins, representing 360° / 16 = 22.5° increments)
   - **Meaning**: The "direction" in semantic space that this edge represents
   - **Example**:
     - `angle_bin: 0` might represent "is-a" relationships
     - `angle_bin: 4` might represent "related-to" relationships
     - `angle_bin: 8` might represent "opposite-of" relationships
   - **Impact**: When light travels from one node to another, the angle bin determines how much refraction penalty is applied (based on the difference between incoming and outgoing angles)

3. **`is_portal`** (bool)
   - **Definition**: Whether this edge crosses room boundaries
   - **Meaning**: Portals are special edges that connect different rooms in the partitioned graph
   - **Example**: In a knowledge graph partitioned by topic, an edge from "Physics" to "Mathematics" would be a portal
   - **Impact**: Portals are used by PVS to determine which rooms are reachable from a given starting room

#### Edge Direction

- Edges are **directed**: `(u, v)` means influence flows from node `u` to node `v`
- The graph uses **Compressed Sparse Row (CSR)** format for efficient edge storage
- Each node stores its outgoing edges in a contiguous array

#### Edge Semantics

The combination of `angle_bin` and `attenuation` creates rich semantic meaning:
- **Strong, aligned edges** (`low attenuation`, `matching angle_bin`): Direct, strong relationships
- **Strong, refracted edges** (`low attenuation`, `different angle_bin`): Related but semantically shifted
- **Weak edges** (`high attenuation`): Distant or tenuous relationships

---

### Rooms

**Rooms** are subgraphs that partition the main graph for optimization purposes. Think of rooms as "neighborhoods" or "clusters" of related nodes.

#### Purpose of Rooms

1. **Query Optimization**: PVS (Potentially Visible Sets) use rooms to prune unreachable parts of the graph
2. **Parallel Processing**: Different rooms can be processed in parallel
3. **Cache Locality**: Nodes in the same room are stored contiguously for better cache performance
4. **Scalability**: Enables efficient queries on billion-node graphs

#### Room Properties

Each room has:
- **`id`** (RoomId, u32): Unique identifier for the room
- **`node_start`** (usize): Starting index of nodes in this room (for contiguous storage)
- **`node_count`** (usize): Number of nodes in this room

#### Room Assignment

- Each node belongs to exactly one room
- Room assignment is stored in `graph.room_map[node_id] = room_id`
- Rooms are typically created by graph partitioning algorithms:
  - **Connected Components**: Each connected component becomes a room
  - **BFS Partitioning**: Creates rooms of approximately equal size using BFS
  - **Manual**: User-specified room assignments

#### Room Size Guidelines

- **Too small** (< 10 nodes): Overhead of room management exceeds benefits
- **Optimal** (50-500 nodes): Good balance of locality and parallelism
- **Too large** (> 10,000 nodes): PVS pruning becomes less effective

#### Portals Between Rooms

- **Portals** are edges that cross room boundaries
- Portals are automatically detected by `detect_portals()`
- PVS uses portals to determine which rooms are reachable from a starting room

---

### PVS (Potentially Visible Set)

**PVS** stands for **Potentially Visible Set**, a precomputed data structure that stores which rooms are reachable from each room when starting with a given angle bin.

#### Purpose

PVS dramatically speeds up queries on large graphs by:
1. **Pruning unreachable rooms**: Before propagation, PVS tells us which rooms can possibly receive influence
2. **Early termination**: If a room isn't in the PVS, we skip all nodes in that room
3. **Reducing computation**: Instead of checking every node, we only check nodes in visible rooms

#### PVS Structure

PVS is a 3D lookup table:
```
PVS[source_room_id][angle_bin][target_room_id] = is_visible
```

For each combination of:
- **Source room**: Where propagation starts
- **Angle bin**: Initial direction of propagation
- **Target room**: Whether this room can receive influence

#### PVS Computation

PVS is computed by:
1. For each room and each angle bin:
   - Find a representative node in that room
   - Simulate limited-depth light propagation
   - Record which rooms received non-zero intensity
2. Store the results in the PVS structure

#### PVS Usage

During propagation:
```rust
// Check if target room is visible before propagating
if let Some(pvs) = pvs {
    if !pvs.is_visible(source_room, angle_bin, target_room)? {
        continue; // Skip this room - not reachable
    }
}
```

#### PVS Trade-offs

- **Memory**: PVS requires O(num_rooms² × num_angle_bins) storage
- **Computation**: Computing PVS requires simulating propagation for each room/angle combination
- **Accuracy**: PVS is conservative (may include rooms that don't actually receive influence)
- **Benefit**: On large graphs, PVS can reduce query time by 10-100x

#### When to Use PVS

- **Always use PVS** for graphs with > 10,000 nodes
- **Optional** for small graphs (< 1,000 nodes) where overhead may exceed benefit
- **Recompute PVS** after major graph structure changes

---

### Angle Bins

**Angle bins** discretize the 2D semantic space into 16 directions (by default). They represent different "semantic directions" that relationships can take.

#### Concept

Think of angle bins as compass directions in semantic space:
- **Bin 0**: One semantic direction (e.g., "is-a", "contains")
- **Bin 4**: A different direction (e.g., "related-to", "similar-to")
- **Bin 8**: Opposite direction (e.g., "opposite-of", "conflicts-with")
- **Bin 12**: Another direction (e.g., "causes", "enables")

#### Default Configuration

- **`N_ANGLE_BINS`**: 16 (configurable)
- Each bin represents 360° / 16 = 22.5° of semantic space
- Bins are circular: bin 15 wraps to bin 0

#### Angular Distance

The **angular distance** between two bins measures how semantically different they are:
- **Distance 0**: Same direction (no refraction penalty)
- **Distance 1-4**: Similar directions (small penalty)
- **Distance 8**: Opposite directions (maximum penalty)

#### Refraction Penalty

When light changes direction (from `angle_bin_in` to `angle_bin_out`), the refraction factor is:
```
ρ = exp(-k * n * (Δ/B)²)
```
Where:
- `k`: Global sharpness factor (default: 5.0)
- `n`: Node's refraction index
- `Δ`: Angular distance between bins
- `B`: Number of angle bins (16)

This means:
- **Aligned edges** (same angle bin): No penalty (ρ ≈ 1.0)
- **Slightly refracted** (distance 1-2): Small penalty (ρ ≈ 0.8-0.9)
- **Strongly refracted** (distance 8): Large penalty (ρ ≈ 0.01-0.1)

#### Practical Usage

- **Set edge `angle_bin`** based on relationship type
- **Set node `default_angle_bin`** based on primary semantic direction
- **Use PVS** with angle bins to find directionally-relevant paths

---

### Light Propagation

**Light propagation** is the core algorithm that simulates how influence flows through the graph.

#### The Algorithm

1. **Initialize**: Source node emits light in initial angle bin
2. **Iterate** (for `max_depth` steps):
   - For each node in the frontier:
     - Calculate reflected intensity
     - For each outgoing edge:
       - Check PVS (if available)
       - Calculate refraction factor
       - Calculate transmitted intensity
       - Update target node's intensity
       - Add to next frontier
3. **Accumulate**: Sum intensities across all angle bins per node

#### Key Parameters

- **`k`**: Global sharpness for refraction (default: 5.0)
- **`min_intensity`**: Minimum intensity to continue propagating (default: 1e-3)
- **`max_depth`**: Maximum number of propagation steps (default: 4)
- **`num_angle_bins`**: Number of discrete directions (default: 16)

#### Intensity Calculation

For each edge traversal:
```
intensity_out = intensity_in × reflection × (1 - attenuation) × refraction_factor
```

Where:
- `intensity_in`: Intensity arriving at source node
- `reflection`: Source node's reflection coefficient
- `attenuation`: Edge's attenuation
- `refraction_factor`: Penalty for direction change

#### Light Distance

Intensity is converted to "light distance" for ranking:
```
distance = -log(intensity + ε)
```

Lower intensity = higher distance = less influence.

---

### Additional Terms

#### CSR (Compressed Sparse Row)

**CSR** is the graph storage format used by RGDB:
- **`row_ptr`**: Array of size `num_nodes + 1`, where `row_ptr[i]..row_ptr[i+1]` indexes edges for node `i`
- **`col_idx`**: Array of target node IDs for each edge
- **`edge_props`**: Array of edge properties, parallel to `col_idx`

Benefits:
- Memory efficient (only stores existing edges)
- Cache-friendly (contiguous access patterns)
- GPU-friendly (can be directly transferred to GPU)

#### Propagation Frontier

The **frontier** is the set of nodes actively propagating light in the current iteration:
- Starts with the source node
- Grows as light propagates to neighbors
- Shrinks as intensity falls below `min_intensity`
- Terminates when frontier is empty or `max_depth` is reached

#### Intensity Accumulation

Each node accumulates intensity across:
- **Per-angle intensities**: `intensities[node_id * num_angle_bins + angle_bin]`
- **Total intensity**: Sum of all per-angle intensities for that node

Total intensity is used for ranking and distance calculations.

---

**Happy querying!** 🚀

