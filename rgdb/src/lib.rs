pub mod graph;
pub mod propagation;
pub mod partitioning;
pub mod rooms;
pub mod level_file;
pub mod queries;
pub mod embeddings;
pub mod relation;
pub mod transitions;
pub mod credit;
pub mod engine;
pub mod error;
pub mod rag;

#[cfg(feature = "llm")]
pub mod llm_integration;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use graph::*;
pub use propagation::*;
pub use partitioning::*;
pub use rooms::*;
pub use relation::*;
// NOTE: `rag::QueryResult` is intentionally left out of this re-export — it would
// collide with `engine::QueryResult` below (both are `pub use ... QueryResult`
// at the crate root, which is E0252). It remains reachable as `rag::QueryResult`.
pub use rag::{RAGQueryEngine, QueryConfig, EmbeddingStore, IntentClassifier, QueryIntent, UserContext};
pub use engine::{EngineConfig, FeedbackError, QueryId, QueryResult, RgdbEngine};
pub use transitions::{TransitionConfig, TransitionError, TransitionStore};
