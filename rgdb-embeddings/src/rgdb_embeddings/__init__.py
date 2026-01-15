"""
RGDB Embeddings - Train embeddings for RGDB graph database.

This package provides tools for generating embeddings compatible with RGDB:
- Pretrained sentence transformers
- Fine-tuning with contrastive learning
- RotatE-style relation-aware embeddings
- GNN encoders respecting RGDB's 16 angle bins

Quick Start:
    >>> from rgdb_embeddings import PretrainedEmbedder, export_to_rgdb
    >>> embedder = PretrainedEmbedder("all-MiniLM-L6-v2")
    >>> embeddings = embedder.encode(["Hello world", "Machine learning"])
    >>> export_to_rgdb(embeddings, "embeddings.emb")

For knowledge graphs:
    >>> from rgdb_embeddings import RotatEForRGDB, TripleLoader, RotatETrainer
    >>> loader = TripleLoader()
    >>> triples = loader.load("knowledge.csv")
    >>> model = RotatEForRGDB(num_entities=1000, dim=256)
    >>> trainer = RotatETrainer(model)
    >>> trainer.train(triples)
"""

__version__ = "0.1.0"

# Configuration
from .config import (
    Config,
    DataConfig,
    EvaluationConfig,
    ExportConfig,
    NUM_ANGLE_BINS,
    RELATION_TO_BIN,
    TrainingConfig,
    relation_to_bin,
)

# Data loading
from .data import (
    Chunk,
    ContrastiveDataset,
    Document,
    DocumentLoader,
    MappedTriple,
    TextChunker,
    Triple,
    TripleDataset,
    TripleLoader,
)

# Models
from .models import (
    FineTuneModel,
    PretrainedEmbedder,
    RGDBGraphEncoder,
    RotatEForRGDB,
)

# Training
from .training import (
    Callback,
    CheckpointCallback,
    DirectionalContrastiveLoss,
    EarlyStoppingCallback,
    EmbeddingTrainer,
    LoggingCallback,
    RotatELoss,
    RotatETrainer,
    TripletLoss,
)

# Export
from .export import export_to_rgdb, load_from_rgdb, validate_embeddings

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "DataConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExportConfig",
    "RELATION_TO_BIN",
    "NUM_ANGLE_BINS",
    "relation_to_bin",
    # Data
    "Document",
    "DocumentLoader",
    "Chunk",
    "TextChunker",
    "Triple",
    "MappedTriple",
    "TripleLoader",
    "ContrastiveDataset",
    "TripleDataset",
    # Models
    "PretrainedEmbedder",
    "RotatEForRGDB",
    "FineTuneModel",
    "RGDBGraphEncoder",
    # Training
    "EmbeddingTrainer",
    "RotatETrainer",
    "TripletLoss",
    "RotatELoss",
    "DirectionalContrastiveLoss",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    # Export
    "export_to_rgdb",
    "load_from_rgdb",
    "validate_embeddings",
]
