//! RAG (Retrieval-Augmented Generation) module for RGDB
//!
//! This module provides hybrid retrieval combining:
//! - Graph-based multi-hop reasoning via light propagation
//! - Vector similarity search on dense embeddings
//! - Query intent classification for directional semantics
//! - Personalization via user context

pub mod embedding_store;
pub mod intent;
pub mod query_engine;
pub mod personalization;

pub use embedding_store::EmbeddingStore;
pub use intent::{QueryIntent, IntentClassifier};
pub use query_engine::{RAGQueryEngine, QueryConfig, QueryResult};
pub use personalization::UserContext;
