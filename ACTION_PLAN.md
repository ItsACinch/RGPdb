# RGDB Development Action Plan

## Current Status ✅
**Phase 1 - Core Engine: COMPLETE**
- ✅ CSR graph structure implemented
- ✅ Node/edge properties (luminance, reflection, refraction_index, attenuation, angle_bin)
- ✅ Influence propagation algorithm working
- ✅ Refraction model implemented
- ✅ Rust PoC running successfully

**Test Results:**
- Graph: 3 nodes (0→1→2)
- Light propagation from node 0 with initial bin 2
- Intensities: Node 0: 2.0, Node 1: 1.62, Node 2: 1.037
- Light distances computed correctly
- Propagation terminates correctly when frontier empties

---

## Phase 2: Rooms & PVS (Next Priority)

### 2.1 Graph Partitioning
**Goal:** Partition graph into rooms (communities/subgraphs)

**Required Information:**
- Partitioning algorithm choice:
  - [ ] Metis-style k-way partitioning
  - [ ] Louvain community detection
  - [ ] Manual room assignment
  - [ ] Simple connected component analysis
- Target room size (nodes per room)
- Balance vs. cut quality tradeoff

**Tasks:**
1. Add `room_id: u32` field to node properties or separate `room_map: Vec<RoomId>`
2. Implement or integrate partitioning algorithm
3. Add room metadata structure:
   ```rust
   struct Room {
       id: RoomId,
       node_start: usize,
       node_count: usize,
   }
   ```
4. Ensure nodes within rooms are contiguous in memory (for cache efficiency)
5. Add tests with various graph sizes and structures

**Deliverables:**
- `partition_graph()` function
- Room metadata structure
- Unit tests for partitioning

**Estimated Complexity:** Medium
**Dependencies:** None (can use existing graph structure)

---

### 2.2 Portal Detection
**Goal:** Identify edges that cross room boundaries

**Required Information:**
- Portal detection strategy:
  - [ ] All inter-room edges are portals
  - [ ] Only high-weight inter-room edges
  - [ ] User-specified portal edges
- Portal metadata needed (just flag, or additional properties?)

**Tasks:**
1. Add `is_portal: bool` flag to `EdgeProps` (or separate portal list)
2. Implement portal detection:
   ```rust
   fn detect_portals(graph: &Graph, room_map: &[RoomId]) -> Vec<(NodeId, NodeId)>
   ```
3. Mark portal edges in graph structure
4. Add portal statistics/logging

**Deliverables:**
- Portal detection function
- Portal-marked edges in graph
- Tests with multi-room graphs

**Estimated Complexity:** Low
**Dependencies:** 2.1 (Graph Partitioning)

---

### 2.3 PVS Computation
**Goal:** Precompute Potentially Visible Sets for each (room, angle_bin) pair

**Required Information:**
- PVS computation algorithm:
  - [ ] BFS from each room with angle bin filtering
  - [ ] Conservative over-approximation (all reachable rooms)
  - [ ] Exact reachability with angle constraints
- PVS storage format:
  - [ ] Bitmask per (room, bin) → room set
  - [ ] Sparse list of room IDs
  - [ ] Hierarchical structure
- Update strategy when graph changes

**Tasks:**
1. Design PVS data structure:
   ```rust
   struct PVS {
       // For each (room_id, angle_bin) → set of reachable rooms
       visibility: HashMap<(RoomId, AngleBin), HashSet<RoomId>>,
   }
   ```
2. Implement PVS computation:
   ```rust
   fn compute_pvs(graph: &Graph, rooms: &[Room], room_map: &[RoomId]) -> PVS
   ```
3. Add PVS to graph structure
4. Optimize computation (parallel, incremental updates)
5. Add validation tests

**Deliverables:**
- PVS computation function
- PVS data structure
- Integration with graph
- Performance benchmarks

**Estimated Complexity:** Medium-High
**Dependencies:** 2.1 (Graph Partitioning), 2.2 (Portal Detection)

---

### 2.4 Propagation Pruning with PVS
**Goal:** Use PVS to skip irrelevant rooms during propagation

**Required Information:**
- Pruning strategy:
  - [ ] Skip entire rooms not in PVS
  - [ ] Skip nodes within non-visible rooms
  - [ ] Early termination when no visible rooms remain
- Performance target (speedup vs. accuracy tradeoff)

**Tasks:**
1. Modify `propagate_light()` to accept PVS
2. Add room-based pruning:
   ```rust
   // Before processing node v:
   if !pvs.is_visible(current_room, angle_bin, v_room) {
       continue; // Skip this neighbor
   }
   ```
3. Add room tracking to frontier states
4. Measure performance improvement
5. Add correctness tests (ensure pruning doesn't change results)

**Deliverables:**
- Updated propagation with PVS pruning
- Performance benchmarks
- Correctness validation

**Estimated Complexity:** Medium
**Dependencies:** 2.3 (PVS Computation)

---

## Phase 3: Level File Format

### 3.1 Binary Format Specification
**Goal:** Define complete binary file format specification

**Required Information:**
- Endianness (little-endian standard)
- Alignment requirements (8-byte, 16-byte?)
- Versioning scheme
- Magic number/header format
- Section ordering and sizes

**Tasks:**
1. Write formal specification document
2. Define header structure:
   ```rust
   struct LevelHeader {
       magic: [u8; 4],        // "RGDB"
       version: u32,
       num_nodes: u32,
       num_edges: u32,
       num_rooms: u32,
       num_angle_bins: u8,
       // ... offsets to sections
   }
   ```
3. Define section layouts (NodeSection, EdgeSection, etc.)
4. Document alignment requirements
5. Create format diagram

**Deliverables:**
- Binary format specification document
- Header structure code
- Section layout definitions

**Estimated Complexity:** Low-Medium
**Dependencies:** None

---

### 3.2 Binary Writer
**Goal:** Serialize graph to binary level file

**Required Information:**
- Output file path/stream
- Compression (optional?)
- Checksum/hash for integrity

**Tasks:**
1. Implement `write_level_file()` function
2. Write header with correct offsets
3. Write each section:
   - NodeSection (node_props array)
   - EdgeSection (CSR: row_ptr, col_idx, edge_props)
   - RoomsSection (room metadata)
   - PortalsSection (portal edge list)
   - AngleTableSection (optional: precomputed angle tables)
   - PVSSection (PVS data)
   - UserMetadata (optional JSON/bytes)
4. Ensure proper alignment
5. Add validation (check file integrity)

**Deliverables:**
- Binary writer implementation
- File format validation
- Tests with various graph sizes

**Estimated Complexity:** Medium
**Dependencies:** 3.1 (Binary Format Specification), Phase 2 (Rooms & PVS)

---

### 3.3 Binary Reader / Mmap Integration
**Goal:** Load graph from binary file, optionally via mmap

**Required Information:**
- mmap library choice (memmap2 crate?)
- Read-only vs. read-write access
- Error handling strategy

**Tasks:**
1. Implement `read_level_file()` function
2. Parse header and validate magic/version
3. Load sections:
   - Option A: Read into memory (Vec)
   - Option B: mmap file (zero-copy)
4. Reconstruct graph structure from binary
5. Add validation and error handling
6. Benchmark mmap vs. read performance

**Deliverables:**
- Binary reader implementation
- mmap integration
- Performance benchmarks
- Error handling

**Estimated Complexity:** Medium
**Dependencies:** 3.2 (Binary Writer)

---

### 3.4 Versioning & Migration
**Goal:** Support multiple file format versions

**Required Information:**
- Version numbering scheme (semver? simple integer?)
- Migration strategy (convert old → new, or support multiple readers?)

**Tasks:**
1. Design versioning scheme
2. Implement version detection
3. Add migration utilities (if needed)
4. Document version history
5. Add tests for version compatibility

**Deliverables:**
- Versioning system
- Migration utilities (if needed)
- Documentation

**Estimated Complexity:** Low
**Dependencies:** 3.3 (Binary Reader)

---

## Phase 4: GPU Acceleration

### 4.1 GPU Framework Selection
**Goal:** Choose GPU compute framework

**Required Information:**
- Target platforms:
  - [ ] CUDA (NVIDIA only)
  - [ ] WGPU (cross-platform, WebGPU-based)
  - [ ] OpenCL (cross-platform, older)
  - [ ] Vulkan Compute
- Performance requirements
- Deployment targets (desktop, server, web?)

**Tasks:**
1. Research and compare frameworks
2. Create proof-of-concept for each candidate
3. Benchmark simple operations
4. Make decision based on requirements

**Deliverables:**
- Framework selection
- POC code
- Decision document

**Estimated Complexity:** Medium
**Dependencies:** None

---

### 4.2 GPU Kernel Development
**Goal:** Implement GPU compute kernels for propagation

**Required Information:**
- Kernel design:
  - [ ] One kernel per propagation step
  - [ ] Single large kernel with iterations
  - [ ] Hybrid CPU/GPU approach
- Memory layout (SoA vs. AoS)
- Atomic operations for intensity accumulation

**Tasks:**
1. Design kernel architecture
2. Implement frontier expansion kernel
3. Implement intensity update kernel
4. Handle atomic operations for concurrent updates
5. Optimize memory access patterns
6. Add GPU error handling

**Deliverables:**
- GPU kernels
- Integration with CPU code
- Performance benchmarks

**Estimated Complexity:** High
**Dependencies:** 4.1 (GPU Framework Selection)

---

### 4.3 GPU Memory Management
**Goal:** Efficient GPU buffer management and data transfer

**Required Information:**
- Buffer allocation strategy
- Transfer optimization (async, pinned memory?)
- Memory pool management

**Tasks:**
1. Implement GPU buffer allocation
2. Add data transfer utilities (CPU ↔ GPU)
3. Optimize transfer patterns (minimize transfers)
4. Add memory pooling/reuse
5. Profile memory usage

**Deliverables:**
- GPU memory management system
- Transfer utilities
- Performance optimizations

**Estimated Complexity:** Medium
**Dependencies:** 4.2 (GPU Kernel Development)

---

### 4.4 Hybrid CPU/GPU Propagation
**Goal:** Seamlessly switch between CPU and GPU execution

**Required Information:**
- When to use GPU vs. CPU:
  - [ ] Always GPU if available
  - [ ] GPU only for large graphs
  - [ ] User-configurable
- Fallback strategy if GPU unavailable

**Tasks:**
1. Create unified propagation interface
2. Add GPU availability detection
3. Implement automatic selection logic
4. Add manual override option
5. Benchmark CPU vs. GPU performance

**Deliverables:**
- Unified propagation API
- Automatic GPU/CPU selection
- Performance comparison

**Estimated Complexity:** Medium
**Dependencies:** 4.3 (GPU Memory Management)

---

## Phase 5: Query Engine / API

### 5.1 Top-K Influence Queries
**Goal:** Find top K nodes by influence from source

**Required Information:**
- K value (configurable?)
- Sorting strategy (by intensity, by distance?)
- Result format

**Tasks:**
1. Implement `query_top_k_influence()`:
   ```rust
   fn query_top_k_influence(
       graph: &Graph,
       source: NodeId,
       initial_bin: AngleBin,
       k: usize,
       params: LightParams,
   ) -> Vec<(NodeId, f32)> // (node, intensity)
   ```
2. Use heap/priority queue for top-K
3. Add early termination optimizations
4. Add tests and benchmarks

**Deliverables:**
- Top-K query function
- Performance optimizations
- Tests

**Estimated Complexity:** Low-Medium
**Dependencies:** Phase 1 (Core Engine)

---

### 5.2 Distance Queries
**Goal:** Compute light-distance between nodes

**Required Information:**
- Query interface (single source, all-pairs, specific pairs?)
- Caching strategy for repeated queries

**Tasks:**
1. Implement `query_distance()`:
   ```rust
   fn query_distance(
       graph: &Graph,
       source: NodeId,
       target: NodeId,
       initial_bin: AngleBin,
       params: LightParams,
   ) -> Option<f32>
   ```
2. Add distance caching (optional)
3. Optimize for single-target queries
4. Add tests

**Deliverables:**
- Distance query function
- Caching (if implemented)
- Tests

**Estimated Complexity:** Low
**Dependencies:** Phase 1 (Core Engine)

---

### 5.3 Hybrid Semantic Queries
**Goal:** Combine graph propagation with vector similarity

**Required Information:**
- Vector embedding format
- Similarity metric (cosine, euclidean?)
- How to combine graph influence + vector similarity

**Tasks:**
1. Add embedding storage to nodes
2. Implement vector similarity functions
3. Design hybrid scoring:
   ```rust
   score = α * graph_influence + (1-α) * vector_similarity
   ```
4. Add configuration for α (weight)
5. Add tests

**Deliverables:**
- Hybrid query function
- Embedding support
- Tests

**Estimated Complexity:** Medium
**Dependencies:** 5.1 (Top-K Influence), vector similarity library

---

### 5.4 Embedding Export
**Goal:** Export node embeddings based on propagation

**Required Information:**
- Embedding dimension
- Export format (numpy, CSV, binary?)
- Which nodes to export (all, top-K, filtered?)

**Tasks:**
1. Implement embedding generation from intensities:
   ```rust
   fn generate_embeddings(
       graph: &Graph,
       sources: &[NodeId],
       params: LightParams,
   ) -> Vec<Vec<f32>>
   ```
2. Add export functions (CSV, binary, numpy)
3. Add dimension reduction (PCA, t-SNE?) if needed
4. Add tests

**Deliverables:**
- Embedding generation
- Export utilities
- Tests

**Estimated Complexity:** Medium
**Dependencies:** Phase 1 (Core Engine)

---

## Phase 6: ML & LLM Integration

### 6.1 Multi-hop Contextual Embeddings
**Goal:** Generate embeddings that capture multi-hop context

**Required Information:**
- Embedding dimension
- How to aggregate multi-hop information
- Training data (if supervised)

**Tasks:**
1. Design embedding generation from propagation results
2. Implement multi-hop aggregation strategies
3. Add embedding quality metrics
4. Compare with traditional embeddings
5. Add tests

**Deliverables:**
- Multi-hop embedding generation
- Quality evaluation
- Documentation

**Estimated Complexity:** Medium-High
**Dependencies:** 5.4 (Embedding Export)

---

### 6.2 LM-driven Queries
**Goal:** Use language models to generate queries

**Required Information:**
- LLM API choice (OpenAI, local model, etc.)
- Query format (natural language → source node + angle bin?)
- Integration approach

**Tasks:**
1. Design query interface
2. Implement LLM integration for query parsing
3. Map natural language to graph queries
4. Add query validation
5. Add examples and tests

**Deliverables:**
- LLM query interface
- Query parsing
- Examples

**Estimated Complexity:** High
**Dependencies:** Phase 5 (Query Engine)

---

### 6.3 RAG with Graph + Physics Reasoning
**Goal:** Integrate RGDB into RAG pipeline

**Required Information:**
- RAG framework choice
- How to combine vector search + graph reasoning
- Use case requirements

**Tasks:**
1. Design RAG integration architecture
2. Implement hybrid retrieval (vector + graph)
3. Add reasoning layer using graph propagation
4. Create example RAG application
5. Evaluate quality vs. traditional RAG

**Deliverables:**
- RAG integration
- Example application
- Evaluation results

**Estimated Complexity:** High
**Dependencies:** 6.1 (Multi-hop Embeddings), 6.2 (LM-driven Queries)

---

## Implementation Priority Recommendations

### Immediate Next Steps (Phase 2):
1. **Start with 2.1 (Graph Partitioning)** - Foundation for all room-based features
2. **Then 2.2 (Portal Detection)** - Simple and enables PVS
3. **Then 2.3 (PVS Computation)** - Core optimization feature
4. **Finally 2.4 (Propagation Pruning)** - Complete Phase 2

### Medium-term (Phase 3):
- Can proceed in parallel with Phase 2 testing
- Binary format enables persistence and GPU transfer

### Long-term (Phases 4-6):
- Phase 4 (GPU) requires significant investment
- Phase 5 (Query Engine) provides user-facing API
- Phase 6 (ML/LLM) depends on Phases 4-5

---

## Testing Strategy

### Unit Tests Needed:
- [ ] Graph construction and manipulation
- [ ] Propagation correctness (known test cases)
- [ ] Partitioning algorithms
- [ ] Portal detection
- [ ] PVS computation
- [ ] Binary I/O round-trip
- [ ] Query functions

### Integration Tests Needed:
- [ ] End-to-end propagation with rooms/PVS
- [ ] File format compatibility
- [ ] GPU/CPU consistency
- [ ] Query performance

### Benchmark Tests Needed:
- [ ] Propagation performance (various graph sizes)
- [ ] PVS pruning effectiveness
- [ ] GPU speedup measurements
- [ ] Query latency

---

## Dependencies to Add

### Phase 2:
- Graph partitioning: Consider `metis` crate or `petgraph` for community detection
- Data structures: `hashbrown` for faster HashMaps

### Phase 3:
- Binary I/O: `byteorder` for endianness, `memmap2` for mmap

### Phase 4:
- GPU: `wgpu` or `cuda` bindings, `ash` for Vulkan

### Phase 5:
- Vector similarity: `ndarray` or `nalgebra` for embeddings

### Phase 6:
- LLM: `openai` crate or `llm-chain` for local models
- RAG: Integration with existing RAG frameworks

---

## Questions to Resolve

1. **Graph Partitioning Algorithm:** Which algorithm provides best balance for this use case?
2. **PVS Granularity:** Should PVS be per-room or per-node? Per-angle-bin or aggregated?
3. **GPU Framework:** What are the deployment targets? (affects framework choice)
4. **Embedding Dimension:** What dimension for multi-hop embeddings?
5. **Update Strategy:** How to handle graph updates (incremental vs. full recompute)?

---

## Success Metrics

### Phase 2:
- [ ] Graph partitions into reasonable rooms (10-1000 nodes each)
- [ ] PVS reduces propagation work by 50%+ on large graphs
- [ ] Correctness maintained (same results with/without PVS)

### Phase 3:
- [ ] Files load 10x+ faster with mmap vs. read
- [ ] File format supports graphs up to 10M nodes
- [ ] Version migration works correctly

### Phase 4:
- [ ] GPU provides 10x+ speedup on large graphs
- [ ] CPU/GPU results match (within numerical precision)

### Phase 5:
- [ ] Top-K queries complete in <100ms for 1M node graphs
- [ ] Hybrid queries outperform pure vector or pure graph

### Phase 6:
- [ ] Multi-hop embeddings show improved quality metrics
- [ ] RAG integration demonstrates reasoning capabilities

