"""Configuration classes for RGDB embeddings training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


# RGDB angle bin mapping - maps relationship strings to angle bins (0-15)
RELATION_TO_BIN: Dict[str, int] = {
    # Bin 0: IsA / Type hierarchy
    "is_a": 0,
    "type_of": 0,
    "instance_of": 0,
    "isa": 0,
    "typeof": 0,
    "instanceof": 0,
    # Bin 1: Requirements / Dependencies
    "requires": 1,
    "depends_on": 1,
    "needs": 1,
    "dependson": 1,
    "prerequisite": 1,
    # Bin 2: General association
    "related_to": 2,
    "associated_with": 2,
    "relatedto": 2,
    "associatedwith": 2,
    "about": 2,
    # Bin 4: Causation
    "causes": 4,
    "leads_to": 4,
    "results_in": 4,
    "leadsto": 4,
    "resultsin": 4,
    "produces": 4,
    # Bin 6: Containment
    "contains": 6,
    "has": 6,
    "includes": 6,
    "comprises": 6,
    # Bin 8: Part-of relationship
    "part_of": 8,
    "belongs_to": 8,
    "member_of": 8,
    "partof": 8,
    "belongsto": 8,
    "memberof": 8,
    # Bin 10: Similarity
    "similar_to": 10,
    "like": 10,
    "resembles": 10,
    "similarto": 10,
    "analogous_to": 10,
    "analogousto": 10,
    # Bin 12: Opposition
    "opposite_of": 12,
    "contrasts_with": 12,
    "oppositeof": 12,
    "contrastswith": 12,
    "antonym_of": 12,
    "antonymof": 12,
    # Bin 14: Enablement
    "enables": 14,
    "allows": 14,
    "supports": 14,
    "facilitates": 14,
    # Bin 15: Conflict
    "conflicts_with": 15,
    "incompatible": 15,
    "conflictswith": 15,
    "incompatible_with": 15,
    "incompatiblewith": 15,
}

# Inverse mapping: bin to canonical relationship name
BIN_TO_RELATION: Dict[int, str] = {
    0: "IsA",
    1: "Requires",
    2: "RelatedTo",
    4: "Causes",
    6: "Contains",
    8: "PartOf",
    10: "SimilarTo",
    12: "OppositeOf",
    14: "Enables",
    15: "ConflictsWith",
}

# Number of angle bins in RGDB
NUM_ANGLE_BINS = 16


def normalize_relation(relation: str) -> str:
    """Normalize a relation string for lookup in RELATION_TO_BIN."""
    return relation.lower().replace(" ", "_").replace("-", "_")


def relation_to_bin(relation: str, default: int = 2) -> int:
    """
    Convert a relation string to an RGDB angle bin.

    Args:
        relation: The relation string (e.g., "is_a", "causes", "part of")
        default: Default bin if relation not found (default: 2 = RelatedTo)

    Returns:
        The angle bin index (0-15)
    """
    normalized = normalize_relation(relation)
    return RELATION_TO_BIN.get(normalized, default)


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    # Document paths
    documents_path: Optional[Path] = None
    triples_path: Optional[Path] = None

    # Chunking parameters
    chunk_size: int = 512  # Tokens per chunk
    chunk_overlap: int = 50  # Overlapping tokens between chunks
    min_chunk_size: int = 50  # Minimum chunk size (skip smaller chunks)

    # Processing options
    include_metadata: bool = True  # Include document metadata in chunks
    lowercase: bool = False  # Convert text to lowercase


@dataclass
class TrainingConfig:
    """Configuration for embedding training."""

    # Training mode
    mode: Literal["pretrained", "finetune", "rotate", "gnn"] = "pretrained"

    # Model parameters
    embedding_dim: int = 384  # Embedding dimension
    num_angle_bins: int = NUM_ANGLE_BINS  # Number of RGDB angle bins
    pretrained_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Loss parameters
    margin: float = 1.0  # Margin for ranking loss
    negative_samples: int = 10  # Number of negative samples per positive

    # Device and optimization
    device: str = "cuda"  # cuda or cpu
    fp16: bool = False  # Use mixed precision training
    num_workers: int = 4  # DataLoader workers

    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_every: int = 1  # Save checkpoint every N epochs
    early_stopping_patience: int = 3  # Stop if no improvement for N epochs

    # Logging
    log_every: int = 100  # Log every N steps
    wandb_project: Optional[str] = None  # W&B project name


@dataclass
class EvaluationConfig:
    """Configuration for embedding evaluation."""

    # Metrics to compute
    compute_mrr: bool = True  # Mean Reciprocal Rank
    compute_hits: bool = True  # Hits@K
    hits_k: List[int] = field(default_factory=lambda: [1, 3, 10])

    # Evaluation settings
    batch_size: int = 256
    filter_known: bool = True  # Filter known triples when ranking


@dataclass
class ExportConfig:
    """Configuration for embedding export."""

    # Output format
    output_path: Path = field(default_factory=lambda: Path("embeddings.emb"))
    normalize: bool = True  # L2 normalize embeddings before export

    # Optional metadata
    include_vocab: bool = False  # Export entity vocabulary
    vocab_path: Optional[Path] = None


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**d.get("data", {})),
            training=TrainingConfig(**d.get("training", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
            export=ExportConfig(**d.get("export", {})),
        )
