# RGDB LLM Integration Guide

This guide explains how to use RGDB as a RAG (Retrieval-Augmented Generation) and VectorDB for LLM applications, with a focus on enterprise search and personalized activity feeds.

## Overview

RGDB is a hybrid database that combines:
- **Graph reasoning**: Multi-hop traversal through semantic relationships
- **Vector similarity**: Dense embedding search with cosine similarity
- **Directional semantics**: 16 angle bins representing relationship types (IsA, Causes, Contains, etc.)
- **Personalization**: User context-aware relevance scoring

Unlike traditional VectorDBs that only match by embedding similarity, RGDB understands _how_ concepts relate (is-a vs. causes vs. contains) and can traverse multiple hops of reasoning.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           Intent Classification              │
│   "What causes X?" → Causation → bin 4      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Source Selection                   │
│   Vector similarity to find starting nodes   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│         Light Propagation                    │
│   Multi-hop graph traversal with refraction  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Hybrid Scoring                     │
│   α×graph + β×vector + γ×personalization    │
└─────────────────────────────────────────────┘
    │
    ▼
Top-K Results with reasoning paths
```

## Quick Start

### 1. Create a RAG Query Engine

```rust
use rgdb::{Graph, RAGQueryEngine, EmbeddingStore, QueryConfig, UserContext};
use ndarray::Array2;

// Load your graph
let graph = Graph::from_adjacency(/* ... */).unwrap();

// Load embeddings (e.g., from sentence-transformers)
let embeddings = Array2::from_shape_vec(
    (num_nodes, embedding_dim),
    embedding_data
).unwrap();

let embedding_store = EmbeddingStore::new(embeddings);

// Create the engine
let engine = RAGQueryEngine::new(graph, embedding_store, None);
```

### 2. Execute a Query

```rust
use ndarray::array;

// Your query embedding (from same model as node embeddings)
let query_embedding = array![0.1, 0.2, 0.3, /* ... */];

// Optional: user context for personalization
let mut user_ctx = UserContext::new("user_123");
user_ctx.record_interaction(42);  // Node 42 was recently viewed

// Query configuration
let config = QueryConfig {
    top_k: 10,
    max_hops: 4,
    alpha: 0.5,  // Graph weight
    beta: 0.4,   // Vector weight
    gamma: 0.1,  // Personalization weight
    ..Default::default()
};

// Execute query
let results = engine.query(
    "What causes supply chain disruptions?",
    &query_embedding,
    Some(&user_ctx),
    &config
);

for result in results {
    println!("Node: {}, Score: {:.3}", result.node_id, result.score);
}
```

## LLM Tool Integration

### OpenAI Function Calling Format

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "rgdb_search",
        "description": "Search the enterprise knowledge base using hybrid graph + vector reasoning. Returns relevant documents with multi-hop reasoning paths.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Natural language search query"
            },
            "query_type": {
              "type": "string",
              "enum": ["definition", "causal", "similarity", "requirements", "composition", "general"],
              "description": "Type of query to optimize search direction. 'definition' for 'What is X?', 'causal' for 'Why does X?', etc."
            },
            "max_hops": {
              "type": "integer",
              "description": "Maximum reasoning depth (1-5). Higher = more thorough but slower.",
              "default": 3
            },
            "top_k": {
              "type": "integer",
              "description": "Number of results to return",
              "default": 10
            }
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "rgdb_explore",
        "description": "Explore relationships from a specific node. Use after rgdb_search to drill deeper into a result.",
        "parameters": {
          "type": "object",
          "properties": {
            "node_id": {
              "type": "integer",
              "description": "ID of the node to explore from"
            },
            "relationship_types": {
              "type": "array",
              "items": {
                "type": "string",
                "enum": ["IsA", "RelatedTo", "Causes", "Contains", "PartOf", "SimilarTo", "Enables", "Requires", "ConflictsWith"]
              },
              "description": "Types of relationships to follow"
            },
            "depth": {
              "type": "integer",
              "default": 2
            }
          },
          "required": ["node_id"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "rgdb_reasoning_path",
        "description": "Find the reasoning path between two concepts. Explains how they are connected.",
        "parameters": {
          "type": "object",
          "properties": {
            "source_query": {
              "type": "string",
              "description": "Starting concept"
            },
            "target_query": {
              "type": "string",
              "description": "Target concept to find connection to"
            }
          },
          "required": ["source_query", "target_query"]
        }
      }
    }
  ]
}
```

### Claude MCP Tool Format

```json
{
  "name": "rgdb_search",
  "description": "Search enterprise knowledge using hybrid graph+vector reasoning",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "query_type": {
        "type": "string",
        "enum": ["definition", "causal", "similarity", "requirements", "general"]
      },
      "max_hops": { "type": "integer", "default": 3 },
      "top_k": { "type": "integer", "default": 10 }
    },
    "required": ["query"]
  }
}
```

## Query Types and When to Use Them

| Query Type | Intent | Angle Bin | Example Queries |
|------------|--------|-----------|-----------------|
| `definition` | What is X? | 0 (IsA) | "What is machine learning?", "Define REST API" |
| `causal` | Why does X? | 4 (Causes) | "Why do servers crash?", "What causes latency?" |
| `requirements` | What do I need? | 1 (Requires) | "Prerequisites for Kubernetes", "How to set up CI/CD" |
| `similarity` | What's like X? | 10 (SimilarTo) | "Alternatives to PostgreSQL", "Similar to React" |
| `composition` | What's inside X? | 6 (Contains) | "Components of microservices", "What's in a Docker image?" |
| `general` | Tell me about X | 2 (RelatedTo) | "Python best practices", "Cloud architecture" |

## Example Prompts for LLMs

### Enterprise Search Use Cases

**1. Technical Documentation Search**
```
User: "How do I configure authentication in our API gateway?"

LLM should call:
rgdb_search(
  query="configure authentication API gateway",
  query_type="requirements",
  max_hops=3
)
```

**2. Incident Investigation**
```
User: "What could cause the payment service to timeout?"

LLM should call:
rgdb_search(
  query="payment service timeout causes",
  query_type="causal",
  max_hops=4
)
```

**3. Architecture Exploration**
```
User: "What components depend on the user database?"

LLM should call:
rgdb_explore(
  node_id=<user_database_node>,
  relationship_types=["Requires", "Contains", "PartOf"],
  depth=2
)
```

**4. Knowledge Discovery**
```
User: "How is our caching layer related to the search service?"

LLM should call:
rgdb_reasoning_path(
  source_query="caching layer",
  target_query="search service"
)
```

### Personalized Activity Feed Use Cases

**1. Daily Digest**
```
User: "What's new that I should know about?"

System generates feed based on:
- User's recent document views
- User's topic affinities (which relationship types they care about)
- Team/department context
```

**2. Follow-up Recommendations**
```
After user reads a document about "Kubernetes deployment":

LLM can suggest:
"Based on what you just read, you might also be interested in:"
- Related: container orchestration patterns
- Requires: Docker basics
- Enables: auto-scaling configurations
```

## Personalization Guide

### Setting Up User Context

```rust
let mut user_ctx = UserContext::new("user_123");

// Set topic interests (0.0-1.0 per angle bin)
user_ctx.set_topic_affinity(0, 0.8);  // IsA (definitions) - high interest
user_ctx.set_topic_affinity(4, 0.9);  // Causes (debugging) - very interested
user_ctx.set_topic_affinity(10, 0.3); // SimilarTo (alternatives) - low interest

// Record interactions
user_ctx.record_interaction(node_id);  // User viewed this node

// Set access control
user_ctx.add_accessible_room(room_id);  // User can only see certain rooms
```

### How Personalization Affects Results

1. **Topic Affinities**: Boost results in user's preferred relationship directions
2. **Interaction History**: Recently/frequently viewed nodes rank higher
3. **Session Context**: Continue from where user left off
4. **Access Control**: Filter results by room permissions

## Performance Tuning

### Query Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 10 | Number of results. Increase for more comprehensive search. |
| `max_hops` | 4 | Reasoning depth. 2-3 for speed, 4-5 for thoroughness. |
| `alpha` | 0.5 | Graph weight. Higher = trust graph structure more. |
| `beta` | 0.4 | Vector weight. Higher = trust embedding similarity more. |
| `gamma` | 0.1 | Personalization weight. Higher = more personalized. |
| `min_relevance` | 1e-3 | Threshold for including results. |
| `refraction_sharpness` | 5.0 | How much to penalize direction changes. |

### Recommended Configurations

**Fast Lookup** (< 10ms):
```rust
QueryConfig {
    top_k: 5,
    max_hops: 2,
    alpha: 0.3,
    beta: 0.7,  // Trust vectors more (faster)
    ..Default::default()
}
```

**Deep Reasoning** (50-100ms):
```rust
QueryConfig {
    top_k: 20,
    max_hops: 5,
    alpha: 0.7,  // Trust graph more
    beta: 0.2,
    use_related_bins: true,  // Explore multiple directions
    ..Default::default()
}
```

**Personalized Feed**:
```rust
QueryConfig {
    top_k: 50,
    max_hops: 3,
    alpha: 0.6,
    beta: 0.0,  // Pure graph-based
    gamma: 0.4,  // High personalization
    ..Default::default()
}
```

## Building the Knowledge Graph

### Document Ingestion Pattern

```
Document → Chunks (512 tokens) → Embeddings
                ↓
         Entity Extraction (NER)
                ↓
         Relationship Extraction
                ↓
    Nodes + Edges with angle bins
```

### Relationship to Angle Bin Mapping

When ingesting documents, map extracted relationships to angle bins:

```rust
fn relation_to_angle_bin(relation: &str) -> AngleBin {
    match relation.to_lowercase().as_str() {
        "is_a" | "type_of" | "instance_of" => 0,
        "requires" | "depends_on" | "needs" => 1,
        "related_to" | "associated_with" => 2,
        "causes" | "leads_to" | "results_in" => 4,
        "contains" | "has" | "includes" => 6,
        "part_of" | "belongs_to" | "member_of" => 8,
        "similar_to" | "like" | "resembles" => 10,
        "opposite_of" | "contrasts_with" => 12,
        "enables" | "allows" | "supports" => 14,
        "conflicts_with" | "incompatible" => 15,
        _ => 2,  // Default to RelatedTo
    }
}
```

## Comparison with Traditional VectorDBs

| Feature | RGDB | Traditional VectorDB |
|---------|------|---------------------|
| Embedding similarity | Yes | Yes |
| Multi-hop reasoning | Yes (native) | No (requires multiple queries) |
| Relationship types | 16 semantic directions | None |
| Query intent | Auto-detected | Manual filtering |
| Personalization | Built-in | External service |
| Explainability | Reasoning paths | Distance only |
| Access control | Room-based | Metadata filtering |

## Troubleshooting

### Low Quality Results

1. **Check embedding quality**: Ensure embeddings are from same model for nodes and queries
2. **Increase max_hops**: May need deeper reasoning for complex queries
3. **Adjust alpha/beta**: If graph structure is unreliable, increase beta (vector weight)
4. **Verify angle bin mapping**: Wrong relationship types → wrong search directions

### Slow Queries

1. **Reduce max_hops**: 2-3 hops usually sufficient
2. **Enable PVS**: Precompute Potentially Visible Sets for large graphs
3. **Reduce top_k**: Only fetch what you need
4. **Use room partitioning**: Partition large graphs into rooms

### Personalization Not Working

1. **Check user context**: Ensure interactions are being recorded
2. **Verify topic affinities**: Default is 1.0 (neutral) for all bins
3. **Check access control**: Empty accessible_rooms means all accessible
4. **Increase gamma**: Default 0.1 may be too low for visible effect

## Next Steps

1. **Ingestion Pipeline**: Implement document chunking and entity extraction
2. **API Server**: Add REST endpoints for tool integration
3. **Monitoring**: Track query latency and result quality
4. **Feedback Loop**: Use user clicks to improve personalization
