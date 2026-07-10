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
// NOTE: `engine::QueryResult` and `rag::QueryResult` are structurally unrelated
// types that happen to share a name (E0252 if both re-exported bare). Re-export
// each under a distinct alias so a stale `use rgdb::QueryResult;` fails to
// compile immediately instead of silently resolving to the wrong type. Both
// remain reachable at their module paths too (`rgdb::engine::QueryResult`,
// `rgdb::rag::QueryResult`).
pub use engine::{EngineConfig, FeedbackError, QueryId, QueryResult as EngineQueryResult, RgdbEngine};
pub use rag::{RAGQueryEngine, QueryConfig, QueryResult as RagQueryResult, EmbeddingStore, IntentClassifier, QueryIntent, UserContext};
pub use transitions::{TransitionConfig, TransitionError, TransitionStore};
