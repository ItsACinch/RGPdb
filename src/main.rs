use rgdb::*;

fn main() {
    // Build a tiny graph:
    //
    // 0 (A) -> 1 (B) -> 2 (C)
    //
    // A->B has angle_bin = 2
    // B->C has angle_bin = 2 (aligned) OR try 8 (misaligned) to see refraction effect.

    let num_nodes = 3;

    let default_node_props = graph::NodeProps {
        luminance: 1.0,
        reflection: 0.9,
        refraction_index: 2.0,
        default_angle_bin: 0,
    };

    // Adjacency list: each entry is Vec<(neighbor, EdgeProps)>
    let adj = vec![
        // Node 0 neighbors
        vec![(
            1,
            graph::EdgeProps {
                attenuation: 0.1,
                angle_bin: 2,
                is_portal: false,
            },
        )],
        // Node 1 neighbors
        vec![(
            2,
            graph::EdgeProps {
                attenuation: 0.2,
                angle_bin: 2, // try 8 to simulate strong bending
                is_portal: false,
            },
        )],
        // Node 2 neighbors
        vec![], // no outgoing
    ];

    let mut graph = graph::Graph::from_adjacency(num_nodes, adj, default_node_props)
        .expect("Failed to create graph");

    // Let's make node 0 a bit brighter and node 1 more reflective.
    graph.set_node_props(
        0,
        graph::NodeProps {
            luminance: 2.0,
            reflection: 0.9,
            refraction_index: 1.0,
            default_angle_bin: 2,
        },
    ).expect("Failed to set node 0 props");
    graph.set_node_props(
        1,
        graph::NodeProps {
            luminance: 0.5,
            reflection: 0.8,
            refraction_index: 2.0,
            default_angle_bin: 2,
        },
    ).expect("Failed to set node 1 props");
    graph.set_node_props(
        2,
        graph::NodeProps {
            luminance: 0.0,
            reflection: 0.5,
            refraction_index: 1.5,
            default_angle_bin: 0,
        },
    ).expect("Failed to set node 2 props");

    let params = propagation::LightParams {
        k: 5.0,
        min_intensity: 1e-3,
        max_depth: 4,
        num_angle_bins: graph::N_ANGLE_BINS,
    };

    let source: graph::NodeId = 0;
    let initial_bin: graph::AngleBin = 2;

    let intensities = propagation::propagate_light(&graph, source, initial_bin, params);
    let distances = propagation::intensity_to_distance(&intensities, 1e-6);

    println!("Total intensities per node:");
    for (i, val) in intensities.iter().enumerate() {
        println!("  Node {}: {:.6}", i, val);
    }

    println!("\nLight distances per node:");
    for (i, d) in distances.iter().enumerate() {
        println!("  Node {}: d = {:.6}", i, d);
    }

    // Test Phase 2: Rooms & PVS
    println!("\n=== Testing Phase 2: Rooms & PVS ===");
    
    // Partition graph
    let room_map = partitioning::partition_graph(
        &graph,
        partitioning::PartitioningAlgorithm::ConnectedComponents,
    );
    
    // Apply room assignments
    for (node_id, &room_id) in room_map.iter().enumerate() {
        graph.set_room(node_id as graph::NodeId, room_id)
            .expect("Failed to set room");
    }
    
    println!("Room assignments:");
    for i in 0..graph.num_nodes() {
        if let Ok(room) = graph.get_room(i as graph::NodeId) {
            println!("  Node {} -> Room {}", i, room);
        }
    }
    
    // Build room collection
    let rooms = rooms::RoomCollection::from_room_map(graph.room_map());
    println!("\nRooms: {}", rooms.num_rooms());
    
    // Detect portals
    let portals = rooms::detect_portals(&graph);
    println!("Portals detected: {}", portals.len());
    
    // Compute PVS
    let pvs = pvs::compute_pvs(&graph, &rooms, &params)
        .expect("Failed to compute PVS");
    println!("PVS computed for {} rooms", rooms.num_rooms());
    
    // Test propagation with PVS
    let intensities_with_pvs = propagation::propagate_light_with_pvs(
        &graph,
        source,
        initial_bin,
        params,
        Some(&pvs),
    );
    println!("\nIntensities with PVS pruning:");
    for (i, val) in intensities_with_pvs.iter().enumerate() {
        println!("  Node {}: {:.6}", i, val);
    }
}
