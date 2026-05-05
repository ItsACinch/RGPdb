# Planning Document: Directional Luminance for Nodes

## Executive Summary

This document outlines the design and implementation plan for adding **directional luminance** to nodes in RGDB. This feature will allow nodes to emit light preferentially in specific semantic directions, improving query performance and enabling more precise relationship modeling.

**Status**: Planning Phase  
**Priority**: High  
**Estimated Effort**: 2-3 weeks  
**Target Version**: 0.2.0

---

## Table of Contents

1. [Motivation](#motivation)
2. [Current State Analysis](#current-state-analysis)
3. [Proposed Design](#proposed-design)
4. [Technical Specifications](#technical-specifications)
5. [Implementation Plan](#implementation-plan)
6. [Performance Analysis](#performance-analysis)
7. [Migration Strategy](#migration-strategy)
8. [Testing Strategy](#testing-strategy)
9. [Risks and Mitigations](#risks-and-mitigations)
10. [Future Enhancements](#future-enhancements)

---

## Motivation

### Problem Statement

Currently, nodes emit light uniformly in all directions. This leads to:

1. **Unnecessary Computation**: Light propagates to nodes that are semantically unrelated
2. **Inefficient Nearest Neighbor**: All nodes receive some intensity, making it harder to find truly related nodes
3. **Lack of Semantic Precision**: Cannot model nodes that are only relevant in specific contexts

### Solution Benefits

Directional luminance will:

1. **Improve Performance**: Skip propagation to semantically unrelated nodes early
2. **Better Nearest Neighbor**: Only nodes in relevant directions receive light
3. **Semantic Precision**: Model nodes that are context-specific (e.g., "Python" only relevant in "programming" direction)
4. **Reduced Compute Overhead**: Fewer edge traversals and intensity calculations

### Use Cases

- **Knowledge Graphs**: A node about "Python" should primarily emit in "programming language" direction, not "snake" direction
- **Recommendation Systems**: A product node should emit in its category direction (e.g., "electronics", "books")
- **Document Search**: A document about "quantum physics" should emit in "physics" direction, not "philosophy"

---

## Current State Analysis

### Current Node Properties

```rust
pub struct NodeProps {
    pub luminance: f32,              // Uniform emission in all directions
    pub reflection: f32,
    pub refraction_index: f32,
    pub default_angle_bin: AngleBin, // Used for PVS, not propagation
}
```

### Current Propagation Behavior

1. Source node emits `luminance` in the `initial_bin` direction
2. Light propagates to all neighbors regardless of semantic relationship
3. Refraction penalty applied based on angle difference
4. All nodes accumulate intensity, even if semantically unrelated

### Current Limitations

- **No directional emission**: `luminance` is scalar, applies to all directions equally
- **Wasteful propagation**: Light reaches nodes that shouldn't be related
- **Inefficient filtering**: Must compute intensity for all nodes, then filter

---

## Proposed Design

### Core Concept

Instead of uniform luminance, nodes will have **directional luminance** defined by:

1. **Relationship Property**: A semantic property that defines the node's primary relationship type
2. **Luminance Distribution**: How much light is emitted in each direction relative to the property
3. **Similarity-Based Angle**: Angle bins represent similarity to the relationship property

### Key Design Decisions

#### Decision 1: Relationship Property Type

**Option A: Enum-based (Recommended)**
```rust
enum RelationshipProperty {
    IsA,           // "Python" is-a "programming language"
    RelatedTo,     // "Python" related-to "data science"
    Causes,        // "Python" causes "productivity"
    Contains,      // "Document" contains "section"
    // ... extensible
}
```

**Option B: String-based**
```rust
relationship_property: String  // "programming_language", "snake", etc.
```

**Option C: Embedding-based**
```rust
relationship_embedding: Array1<f32>  // Vector representation
```

**Recommendation**: Option A (Enum-based) for:
- Type safety
- Performance (integer comparison vs. string/vector)
- Extensibility (can add new properties)
- Clear semantics

#### Decision 2: Luminance Distribution Model

**Option A: Single Peak (Recommended)**
```rust
// Node emits maximum luminance in one direction, falls off with angle
directional_luminance: [f32; N_ANGLE_BINS]  // Per-angle-bin luminance
```

**Option B: Gaussian Distribution**
```rust
// Node emits with Gaussian falloff around primary direction
primary_direction: AngleBin,
luminance_std: f32,  // Standard deviation of Gaussian
```

**Option C: Multi-Peak**
```rust
// Node can emit in multiple directions (e.g., "Python" in both "programming" and "data science")
peak_directions: Vec<(AngleBin, f32)>,  // (direction, strength)
```

**Recommendation**: Option A (Single Peak) initially, with path to Option C:
- Simpler implementation
- Clear semantics
- Can extend to multi-peak later
- Good performance (array lookup)

#### Decision 3: Angle-Property Mapping

**Approach**: Map relationship properties to angle bins based on semantic similarity

```rust
// Example mapping (configurable)
fn property_to_angle_bin(property: RelationshipProperty) -> AngleBin {
    match property {
        RelationshipProperty::IsA => 0,
        RelationshipProperty::RelatedTo => 4,
        RelationshipProperty::Causes => 8,
        RelationshipProperty::Contains => 12,
        // ...
    }
}

// Similarity between properties
fn property_similarity(p1: RelationshipProperty, p2: RelationshipProperty) -> f32 {
    // Returns 0.0 (opposite) to 1.0 (same)
    // Used to determine angle distance
}
```

---

## Technical Specifications

### Data Structure Changes

#### New Node Properties

```rust
pub struct NodeProps {
    // Existing properties
    pub reflection: f32,
    pub refraction_index: f32,
    pub default_angle_bin: AngleBin,
    
    // NEW: Directional luminance
    pub relationship_property: Option<RelationshipProperty>,  // None = uniform emission
    pub directional_luminance: [f32; N_ANGLE_BINS],          // Per-angle-bin emission
    
    // DEPRECATED: Will be computed from directional_luminance
    // pub luminance: f32,  // Remove or make computed property
}
```

#### New Types

```rust
/// Semantic relationship properties that define node directionality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationshipProperty {
    /// "is-a" relationship (hyponymy)
    IsA,
    /// "related-to" relationship (general association)
    RelatedTo,
    /// "causes" relationship (causation)
    Causes,
    /// "contains" relationship (meronymy)
    Contains,
    /// "part-of" relationship (reverse meronymy)
    PartOf,
    /// "similar-to" relationship (synonymy)
    SimilarTo,
    /// "opposite-of" relationship (antonymy)
    OppositeOf,
    /// "enables" relationship
    Enables,
    /// "requires" relationship
    Requires,
    /// "conflicts-with" relationship
    ConflictsWith,
    // Extensible: can add more as needed
}

/// Mapping from relationship properties to angle bins
pub struct PropertyAngleMap {
    /// Maps each property to its primary angle bin
    property_to_bin: HashMap<RelationshipProperty, AngleBin>,
    /// Similarity matrix between properties
    property_similarity: [[f32; N_PROPERTIES]; N_PROPERTIES],
}

impl PropertyAngleMap {
    /// Get angle bin for a property
    pub fn angle_bin(&self, property: RelationshipProperty) -> AngleBin {
        self.property_to_bin[&property]
    }
    
    /// Get similarity between two properties (0.0 to 1.0)
    pub fn similarity(&self, p1: RelationshipProperty, p2: RelationshipProperty) -> f32 {
        let idx1 = p1 as usize;
        let idx2 = p2 as usize;
        self.property_similarity[idx1][idx2]
    }
    
    /// Get angular distance between two properties
    pub fn angular_distance(&self, p1: RelationshipProperty, p2: RelationshipProperty) -> u8 {
        let bin1 = self.angle_bin(p1);
        let bin2 = self.angle_bin(p2);
        angular_distance(bin1, bin2, N_ANGLE_BINS)
    }
}
```

### Propagation Algorithm Changes

#### Current Algorithm
```rust
// Current: Uniform emission
intensities[src_idx * b + initial_bin] = src_props.luminance.max(1.0);
```

#### New Algorithm
```rust
// New: Directional emission
if let Some(property) = src_props.relationship_property {
    // Emit in each direction based on directional_luminance
    for angle_bin in 0..N_ANGLE_BINS {
        let luminance = src_props.directional_luminance[angle_bin];
        if luminance > min_luminance {
            intensities[src_idx * b + angle_bin] = luminance;
        }
    }
} else {
    // Fallback: Uniform emission (backward compatibility)
    intensities[src_idx * b + initial_bin] = src_props.luminance.max(1.0);
}
```

#### Early Termination Optimization

```rust
// During propagation, check if target node's property is compatible
if let Some(target_property) = target_node.relationship_property {
    if let Some(source_property) = source_node.relationship_property {
        // Check semantic compatibility
        let similarity = property_angle_map.similarity(source_property, target_property);
        if similarity < MIN_PROPERTY_SIMILARITY {
            continue; // Skip this edge - properties incompatible
        }
    }
}
```

### API Changes

#### Node Creation

```rust
// Old API (still supported for backward compatibility)
let props = NodeProps {
    luminance: 2.0,
    reflection: 0.9,
    refraction_index: 1.0,
    default_angle_bin: 2,
};

// New API (recommended)
let props = NodeProps {
    relationship_property: Some(RelationshipProperty::IsA),
    directional_luminance: create_directional_luminance(
        RelationshipProperty::IsA,
        peak_luminance: 2.0,
        falloff_rate: 0.8,  // How quickly luminance falls off with angle
    ),
    reflection: 0.9,
    refraction_index: 1.0,
    default_angle_bin: 2,
};
```

#### Helper Functions

```rust
/// Create directional luminance distribution for a property
pub fn create_directional_luminance(
    property: RelationshipProperty,
    peak_luminance: f32,
    falloff_rate: f32,  // 0.0 = uniform, 1.0 = only in primary direction
) -> [f32; N_ANGLE_BINS] {
    let primary_bin = property_angle_map.angle_bin(property);
    let mut luminance = [0.0; N_ANGLE_BINS];
    
    for angle_bin in 0..N_ANGLE_BINS {
        let angular_dist = angular_distance(primary_bin, angle_bin, N_ANGLE_BINS);
        let similarity = 1.0 - (angular_dist as f32 / N_ANGLE_BINS as f32) * falloff_rate;
        luminance[angle_bin] = peak_luminance * similarity.max(0.0);
    }
    
    luminance
}

/// Create uniform luminance (backward compatibility)
pub fn create_uniform_luminance(luminance: f32) -> [f32; N_ANGLE_BINS] {
    [luminance; N_ANGLE_BINS]
}
```

---

## Implementation Plan

### Phase 1: Core Data Structures (Week 1, Days 1-2)

**Tasks**:
1. Define `RelationshipProperty` enum
2. Define `PropertyAngleMap` structure
3. Add `directional_luminance` to `NodeProps`
4. Add helper functions for creating luminance distributions
5. Update `NodeProps::default()` to use uniform luminance

**Deliverables**:
- Updated `src/graph.rs` with new types
- Unit tests for property-angle mapping
- Migration helper functions

### Phase 2: Propagation Updates (Week 1, Days 3-5)

**Tasks**:
1. Update `propagate_light_with_pvs` to use directional luminance
2. Implement early termination based on property compatibility
3. Update intensity initialization logic
4. Add property similarity checks during edge traversal

**Deliverables**:
- Updated `src/propagation.rs`
- Integration tests for directional propagation
- Performance benchmarks

### Phase 3: Property-Angle Mapping (Week 2, Days 1-2)

**Tasks**:
1. Implement `PropertyAngleMap` with default mappings
2. Create similarity matrix for properties
3. Add configuration for custom mappings
4. Add validation for mapping consistency

**Deliverables**:
- `src/property_map.rs` module
- Default property-angle mappings
- Configuration system

### Phase 4: Optimization (Week 2, Days 3-5)

**Tasks**:
1. Implement early termination optimization
2. Add property-based PVS filtering
3. Optimize luminance distribution lookups
4. Profile and optimize hot paths

**Deliverables**:
- Optimized propagation algorithm
- Performance benchmarks showing improvement
- Documentation of optimizations

### Phase 5: CUDA Integration (Week 3, Days 1-3)

**Tasks**:
1. Update CUDA kernels to support directional luminance
2. Add property compatibility checks in GPU kernels
3. Optimize GPU memory layout for directional data
4. Benchmark GPU performance

**Deliverables**:
- Updated `kernels/propagate_kernel.cu`
- GPU-accelerated directional propagation
- Performance comparison (CPU vs. GPU)

### Phase 6: API and Documentation (Week 3, Days 4-5)

**Tasks**:
1. Update public API documentation
2. Add examples to `GETTING_STARTED.md`
3. Update `README.md` with directional luminance explanation
4. Create migration guide for existing code

**Deliverables**:
- Updated documentation
- Migration guide
- Example code

---

## Performance Analysis

### Expected Improvements

#### Computation Reduction

**Current**: For a graph with N nodes and average degree D:
- Edge traversals: O(N × D × max_depth)
- Intensity calculations: O(N × D × max_depth)

**With Directional Luminance**:
- Early termination: Skip ~30-50% of incompatible edges
- Reduced edge traversals: O(N × D × max_depth × 0.5-0.7)
- **Expected speedup**: 1.4x - 2.0x for typical graphs

#### Memory Impact

**Additional Memory per Node**:
- `directional_luminance`: 16 × 4 bytes = 64 bytes (vs. 4 bytes for scalar)
- `relationship_property`: 1 byte (Option<u8>)
- **Total overhead**: ~65 bytes per node

**For 1M nodes**: ~65 MB additional memory (acceptable)

#### Query Quality

- **Precision**: Improved (fewer false positives)
- **Recall**: Maintained (compatible nodes still found)
- **Ranking**: More accurate (semantically related nodes rank higher)

### Benchmarking Plan

1. **Synthetic Graphs**:
   - 10K, 100K, 1M nodes
   - Various property distributions
   - Measure propagation time, memory usage

2. **Real-World Graphs**:
   - Knowledge graph (e.g., Wikidata subset)
   - Document similarity graph
   - Social network

3. **Metrics**:
   - Propagation time
   - Memory usage
   - Query precision/recall
   - GPU acceleration speedup

---

## Migration Strategy

### Backward Compatibility

**Approach**: Maintain full backward compatibility

1. **Default Behavior**: If `relationship_property` is `None`, use uniform luminance
2. **Legacy API**: Keep `luminance` field (deprecated, computed from `directional_luminance`)
3. **Automatic Conversion**: Helper function to convert old `NodeProps` to new format

### Migration Path

#### Step 1: Add New Fields (Non-Breaking)
```rust
pub struct NodeProps {
    // Old fields (still work)
    pub luminance: f32,  // Now computed: luminance = max(directional_luminance)
    
    // New fields (optional)
    pub relationship_property: Option<RelationshipProperty>,
    pub directional_luminance: [f32; N_ANGLE_BINS],
    // ...
}
```

#### Step 2: Update Default Implementation
```rust
impl Default for NodeProps {
    fn default() -> Self {
        let uniform_lum = 1.0;
        Self {
            luminance: uniform_lum,  // For backward compatibility
            relationship_property: None,
            directional_luminance: create_uniform_luminance(uniform_lum),
            // ...
        }
    }
}
```

#### Step 3: Provide Migration Helpers
```rust
impl NodeProps {
    /// Convert from old format (luminance only) to new format
    pub fn from_uniform_luminance(luminance: f32) -> Self {
        Self {
            luminance,
            relationship_property: None,
            directional_luminance: create_uniform_luminance(luminance),
            // ...
        }
    }
    
    /// Get effective luminance (for backward compatibility)
    pub fn luminance(&self) -> f32 {
        if let Some(_) = self.relationship_property {
            // Return peak luminance
            self.directional_luminance.iter().copied().fold(0.0, f32::max)
        } else {
            // Uniform: return any value (they're all the same)
            self.directional_luminance[0]
        }
    }
}
```

### Deprecation Timeline

- **v0.2.0**: Add new fields, keep `luminance` (deprecated)
- **v0.3.0**: `luminance` becomes computed property
- **v0.4.0**: Remove `luminance` field (breaking change)

---

## Testing Strategy

### Unit Tests

1. **Property-Angle Mapping**:
   - Test property to angle bin conversion
   - Test property similarity calculations
   - Test angular distance between properties

2. **Luminance Distribution**:
   - Test uniform luminance creation
   - Test directional luminance creation
   - Test falloff behavior

3. **Propagation**:
   - Test directional emission
   - Test early termination
   - Test property compatibility checks

### Integration Tests

1. **End-to-End Queries**:
   - Test top-K queries with directional luminance
   - Test distance queries
   - Test hybrid queries

2. **Performance Tests**:
   - Benchmark propagation time
   - Benchmark memory usage
   - Compare with/without directional luminance

### Property Test Cases

1. **Uniform Nodes**: Nodes without directional luminance (backward compatibility)
2. **Single Property**: Nodes with one primary property
3. **Multi-Property**: Nodes with multiple properties (future)
4. **Property Conflicts**: Nodes with incompatible properties

---

## Risks and Mitigations

### Risk 1: Breaking Changes

**Risk**: Changes to `NodeProps` might break existing code

**Mitigation**:
- Maintain full backward compatibility
- Provide migration helpers
- Gradual deprecation timeline

### Risk 2: Performance Regression

**Risk**: Additional memory and computation might slow down queries

**Mitigation**:
- Early termination optimization
- Careful profiling and optimization
- Fallback to uniform luminance if needed

### Risk 3: Property Mapping Complexity

**Risk**: Mapping properties to angles might be too rigid or complex

**Mitigation**:
- Start with simple, extensible mapping
- Allow custom mappings
- Provide sensible defaults

### Risk 4: GPU Kernel Complexity

**Risk**: CUDA kernels become more complex with directional luminance

**Mitigation**:
- Incremental GPU updates
- Maintain CPU fallback
- Extensive GPU testing

---

## Future Enhancements

### Multi-Property Nodes

Allow nodes to have multiple relationship properties:
```rust
relationship_properties: Vec<(RelationshipProperty, f32)>,  // (property, weight)
```

### Adaptive Property Mapping

Learn property-angle mappings from data:
- Use embeddings to determine property similarity
- Automatically map properties to angle bins
- Update mappings based on query patterns

### Property-Aware PVS

Extend PVS to consider properties:
- PVS[source_room][source_property][target_room][target_property]
- More precise pruning based on property compatibility

### Dynamic Property Assignment

Assign properties automatically:
- Use embeddings to infer properties
- Learn from graph structure
- Update properties based on usage patterns

---

## Open Questions

1. **Property Set**: Should we start with a fixed set of properties, or allow user-defined properties?
   - **Recommendation**: Fixed set initially, extensible later

2. **Luminance Distribution**: Single peak vs. multi-peak vs. custom distribution?
   - **Recommendation**: Single peak initially, path to multi-peak

3. **Property Similarity**: How to determine similarity between properties?
   - **Recommendation**: Start with hand-crafted matrix, learn from data later

4. **Backward Compatibility**: How long to maintain `luminance` field?
   - **Recommendation**: 2-3 versions (6-9 months)

5. **Performance Target**: What speedup is acceptable?
   - **Recommendation**: 1.5x minimum, 2.0x target

---

## Success Criteria

### Functional Requirements

- [ ] Nodes can emit light directionally based on relationship properties
- [ ] Early termination skips incompatible nodes
- [ ] Backward compatibility maintained
- [ ] GPU acceleration supports directional luminance

### Performance Requirements

- [ ] 1.5x speedup for typical queries
- [ ] < 10% memory overhead
- [ ] No regression in query quality (precision/recall)

### Quality Requirements

- [ ] Comprehensive test coverage (> 80%)
- [ ] Documentation complete
- [ ] Migration guide available
- [ ] Examples provided

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Data Structures | 2 days | Core types and helpers |
| Phase 2: Propagation | 3 days | Updated algorithm |
| Phase 3: Property Mapping | 2 days | Mapping system |
| Phase 4: Optimization | 3 days | Performance improvements |
| Phase 5: CUDA Integration | 3 days | GPU support |
| Phase 6: Documentation | 2 days | Docs and examples |
| **Total** | **15 days** | **v0.2.0 Release** |

---

## References

- [README.md](README.md) - Core RGDB concepts
- [GETTING_STARTED.md](GETTING_STARTED.md) - User guide
- [CODE_REVIEW.md](CODE_REVIEW.md) - Code quality standards
- [ACTION_PLAN.md](ACTION_PLAN.md) - Overall project roadmap

---

**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Next Review**: After Phase 1 completion

