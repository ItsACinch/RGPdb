# RGDB RAG Module Test Plan

This document outlines the testing strategy for the RAG (Retrieval-Augmented Generation) module in RGDB.

## Test Categories

### 1. Unit Tests

#### 1.1 Embedding Store (`src/rag/embedding_store.rs`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_embedding_store_basic` | Create store with known embeddings | Store contains correct number of nodes and dimensions |
| `test_cosine_similarity` | Compute similarity of identical vectors | Returns 1.0 |
| `test_cosine_similarity_orthogonal` | Compute similarity of orthogonal vectors | Returns 0.0 |
| `test_top_k_similar` | Find top-K similar nodes | Returns nodes in descending similarity order |
| `test_all_similarities` | Compute all similarities at once | Returns vector of correct length with valid values |
| `test_save_load_roundtrip` | Save and load embedding file | Loaded store matches original |
| `test_zero_vector_handling` | Query with zero vector | Returns zeros, no NaN/Inf |
| `test_set_embedding` | Update a single embedding | Norm is recomputed, similarity changes |

#### 1.2 Intent Classifier (`src/rag/intent.rs`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_definition_intent` | "What is X?" queries | Returns `QueryIntent::Definition` |
| `test_causal_intent` | "Why does X?" queries | Returns `QueryIntent::Causation` |
| `test_requirement_intent` | "How to X?" queries | Returns `QueryIntent::Requirements` |
| `test_similarity_intent` | "Similar to X" queries | Returns `QueryIntent::Similarity` |
| `test_composition_intent` | "Contains X" queries | Returns `QueryIntent::Composition` |
| `test_default_association` | Generic queries | Returns `QueryIntent::Association` |
| `test_angle_bin_mapping` | All intent types | Maps to correct angle bins (0-15) |
| `test_related_bins` | Multi-hop intents | Returns multiple related bins |
| `test_confidence_scoring` | Multiple keyword matches | Higher confidence for more matches |

#### 1.3 Personalization (`src/rag/personalization.rs`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_user_context_creation` | Create new user context | Default affinities are 1.0 |
| `test_record_interaction` | Record node interaction | History updated, session updated |
| `test_boost_calculation` | Compute boost for node | Boosted for interacted nodes |
| `test_access_control_empty` | Empty accessible_rooms | All rooms accessible |
| `test_access_control_restricted` | Specific rooms added | Only listed rooms accessible |
| `test_inaccessible_node_boost` | Boost for inaccessible node | Returns 0.0 |
| `test_recency_decay` | Multiple session interactions | Recent nodes get higher boost |
| `test_feed_seeds` | Get seeds for activity feed | Returns mix of recent and frequent |
| `test_modulate_luminance` | Apply topic affinities | Luminance scaled by affinity |

#### 1.4 Query Engine (`src/rag/query_engine.rs`)

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_query_engine_creation` | Create engine with graph and embeddings | Engine initializes correctly |
| `test_default_config` | Default QueryConfig | Reasonable defaults (top_k=10, etc.) |
| `test_query_result_ordering` | QueryResult comparison | Sorts by score descending |
| `test_hybrid_scoring` | Combine graph + vector scores | Weighted combination is correct |
| `test_personalization_boost` | Apply user context | Scores modified by boost factor |
| `test_query_from_sources` | Query with specific source nodes | Propagates from all sources |
| `test_generate_feed` | Generate activity feed | Returns personalized results |
| `test_find_reasoning_path` | Path between connected nodes | Returns valid path |
| `test_find_reasoning_path_disconnected` | Path between disconnected nodes | Returns None |

### 2. Integration Tests

#### 2.1 End-to-End Query Flow

```rust
#[test]
fn test_e2e_query_flow() {
    // 1. Build a small graph with known structure
    // 2. Create embeddings for each node
    // 3. Execute query with known answer
    // 4. Verify expected nodes are in top results
}
```

**Test Data Structure:**
```
Node 0: "Machine Learning" (Definition, bin 0)
    ├── Node 1: "Neural Networks" (IsA, bin 0)
    ├── Node 2: "Training Data" (Requires, bin 1)
    └── Node 3: "Predictions" (Enables, bin 14)

Node 4: "Deep Learning" (SimilarTo Node 0, bin 10)
```

**Test Queries:**
| Query | Expected Top Result | Reason |
|-------|---------------------|--------|
| "What is machine learning?" | Node 0, Node 1 | Definition intent → IsA direction |
| "What do I need for machine learning?" | Node 2 | Requirements intent → Requires direction |
| "What can machine learning do?" | Node 3 | Capability intent → Enables direction |
| "What's similar to machine learning?" | Node 4 | Similarity intent → SimilarTo direction |

#### 2.2 Personalization Integration

```rust
#[test]
fn test_personalization_integration() {
    // 1. Create graph and engine
    // 2. Create user context with specific affinities
    // 3. Execute same query with/without personalization
    // 4. Verify personalized results differ appropriately
}
```

#### 2.3 Multi-Source Propagation

```rust
#[test]
fn test_multi_source_propagation() {
    // 1. Create graph with multiple relevant source nodes
    // 2. Execute query that matches multiple sources
    // 3. Verify intensity accumulates from all sources
}
```

### 3. Performance Benchmarks

Location: `benches/rag_benchmark.rs`

#### 3.1 Query Latency

| Benchmark | Graph Size | Expected Latency |
|-----------|------------|------------------|
| `bench_query_small` | 1K nodes | < 1ms |
| `bench_query_medium` | 100K nodes | < 10ms |
| `bench_query_large` | 1M nodes | < 100ms |
| `bench_query_with_pvs` | 1M nodes + PVS | < 50ms |

#### 3.2 Embedding Operations

| Benchmark | Operation | Expected |
|-----------|-----------|----------|
| `bench_top_k_similar` | top-10 from 100K | < 5ms |
| `bench_all_similarities` | All sims for 100K | < 20ms |
| `bench_embedding_load` | Load 100K × 384 | < 100ms |

#### 3.3 Personalization Overhead

| Benchmark | Operation | Expected |
|-----------|-----------|----------|
| `bench_boost_calculation` | 100K nodes | < 1ms |
| `bench_feed_generation` | 100K nodes, 10 seeds | < 50ms |

### 4. Stress Tests

#### 4.1 Concurrent Queries

```rust
#[test]
fn test_concurrent_queries() {
    // Launch 100 concurrent queries
    // Verify all return valid results
    // Measure total throughput
}
```

#### 4.2 Large User History

```rust
#[test]
fn test_large_user_history() {
    // Create user with 10K interactions
    // Verify boost calculation still fast
    // Verify memory usage reasonable
}
```

#### 4.3 Edge Cases

| Test Case | Scenario | Expected |
|-----------|----------|----------|
| Empty graph | Query against 0 nodes | Returns empty results |
| Single node | Query against 1 node | Returns that node if relevant |
| No embeddings | Query without embedding store | Uses graph-only scoring |
| Zero query embedding | Query with zero vector | Returns empty or handles gracefully |
| Max depth reached | Very deep graph | Stops at max_depth |

### 5. Compatibility Tests

#### 5.1 Backward Compatibility

```rust
#[test]
fn test_legacy_influence_result() {
    // Execute RAG query
    // Convert to InfluenceResult
    // Verify fields match
}
```

#### 5.2 File Format Compatibility

```rust
#[test]
fn test_embedding_file_format() {
    // Save embeddings
    // Load with different version
    // Verify data integrity
}
```

## Test Fixtures

### Sample Graph for Testing

```rust
fn create_test_graph() -> Graph {
    // 10-node knowledge graph
    // Nodes: ML, NN, CNN, RNN, Data, Python, TensorFlow, Keras, Training, Inference
    // Edges: Various relationship types
}
```

### Sample Embeddings

```rust
fn create_test_embeddings() -> Array2<f32> {
    // 10 nodes × 16 dimensions
    // Each row is a unit vector in a known direction
}
```

### Sample User Contexts

```rust
fn create_test_user_contexts() -> Vec<UserContext> {
    vec![
        UserContext::new("data_scientist"),    // High affinity for ML topics
        UserContext::new("devops"),            // High affinity for infra topics
        UserContext::new("new_user"),          // No history, default affinities
        UserContext::new("restricted_user"),   // Limited room access
    ]
}
```

## Test Execution

### Running All RAG Tests

```bash
# Unit tests only
cargo test rag::

# Include integration tests
cargo test --test integration_rag

# With verbose output
cargo test rag:: -- --nocapture

# Single test
cargo test rag::embedding_store::tests::test_cosine_similarity
```

### Running Benchmarks

```bash
# All RAG benchmarks
cargo bench --bench rag_benchmark

# Specific benchmark
cargo bench --bench rag_benchmark -- query_latency
```

### Coverage Report

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage for RAG module
cargo tarpaulin --out Html -- --test-threads=1
```

## Acceptance Criteria

### Functional Requirements

- [ ] Intent classification accuracy > 90% on test queries
- [ ] Top-10 results contain expected nodes for all test cases
- [ ] Personalization visibly affects result ordering
- [ ] Access control correctly filters results
- [ ] Reasoning paths are valid (all edges exist)

### Performance Requirements

- [ ] Query latency < 100ms for 1M node graph
- [ ] Memory usage < 2GB for 1M nodes with 384-dim embeddings
- [ ] Throughput > 100 queries/second

### Quality Requirements

- [ ] All unit tests pass
- [ ] Code coverage > 80% for RAG module
- [ ] No panics on edge cases
- [ ] No memory leaks (valgrind clean)

## Test Schedule

| Phase | Tests | Duration |
|-------|-------|----------|
| Phase 1 | Unit tests for all modules | 2 days |
| Phase 2 | Integration tests | 2 days |
| Phase 3 | Performance benchmarks | 1 day |
| Phase 4 | Stress tests | 1 day |
| Phase 5 | Edge cases and fixes | 2 days |

## Known Issues to Test For

1. **Integer overflow**: Large node IDs × angle bins
2. **Division by zero**: Zero embeddings, zero max intensity
3. **Infinite loops**: Cyclic graphs in path finding
4. **Memory exhaustion**: Very deep propagation
5. **NaN propagation**: Invalid similarity calculations

## Future Test Additions

- [ ] GPU acceleration tests (CUDA)
- [ ] Distributed query tests
- [ ] Incremental update tests
- [ ] Model hot-swap tests
