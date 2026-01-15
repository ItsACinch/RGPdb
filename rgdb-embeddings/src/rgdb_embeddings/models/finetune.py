"""Fine-tuning models for embedding training."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineTuneModel(nn.Module):
    """
    Fine-tuning wrapper for sentence transformer models.

    Adds a projection layer and supports contrastive learning
    for domain-specific fine-tuning.
    """

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        output_dim: Optional[int] = None,
        pooling: str = "mean",
        freeze_base: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize the fine-tuning model.

        Args:
            base_model_name: Name of the base sentence-transformers model
            output_dim: Output dimension (None = same as base model)
            pooling: Pooling strategy (mean, cls, max)
            freeze_base: Whether to freeze base model weights
            dropout: Dropout rate for projection layer
        """
        super().__init__()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required")

        self.base_model = SentenceTransformer(base_model_name)
        self.base_dim = self.base_model.get_sentence_embedding_dimension()
        self.output_dim = output_dim or self.base_dim
        self.pooling = pooling

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Projection layer
        if self.output_dim != self.base_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.base_dim, self.output_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(self.output_dim),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Embeddings [batch_size, output_dim]
        """
        # Get base embeddings
        base_emb = self.base_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        # Project
        emb = self.projection(base_emb)

        # Normalize
        emb = F.normalize(emb, p=2, dim=-1)

        return emb

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """Alias for forward with optional normalization."""
        emb = self.forward(texts)
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.

        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim]
            margin: Margin for triplet loss

        Returns:
            Loss scalar
        """
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

    def get_embedding_dimension(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim


class MultipleNegativesRankingModel(nn.Module):
    """
    Model optimized for Multiple Negatives Ranking Loss.

    This is an effective approach for fine-tuning where all other
    samples in the batch serve as negatives.
    """

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        scale: float = 20.0,
    ):
        """
        Initialize the model.

        Args:
            base_model_name: Base sentence-transformers model
            scale: Temperature scaling factor
        """
        super().__init__()

        from sentence_transformers import SentenceTransformer

        self.base_model = SentenceTransformer(base_model_name)
        self.scale = scale

    def forward(
        self,
        queries: List[str],
        positives: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode queries and positive documents.

        Args:
            queries: Query texts
            positives: Positive document texts (one per query)

        Returns:
            Tuple of (query_embeddings, positive_embeddings)
        """
        query_emb = self.base_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        pos_emb = self.base_model.encode(
            positives,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return query_emb, pos_emb

    def loss(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Multiple Negatives Ranking Loss.

        All other samples in the batch serve as negatives.

        Args:
            query_emb: Query embeddings [batch_size, dim]
            pos_emb: Positive embeddings [batch_size, dim]

        Returns:
            Loss scalar
        """
        # Similarity matrix [batch_size, batch_size]
        scores = torch.mm(query_emb, pos_emb.t()) * self.scale

        # Labels: diagonal is positive (index i matches index i)
        labels = torch.arange(scores.size(0), device=scores.device)

        # Cross-entropy loss
        loss = F.cross_entropy(scores, labels)
        return loss


class DirectionalFineTuneModel(nn.Module):
    """
    Fine-tuning model that learns direction-specific projections.

    This model learns a separate projection head for each RGDB angle bin,
    allowing the same text to have different embeddings depending on the
    relationship context.
    """

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        output_dim: int = 256,
        num_angles: int = 16,
        shared_base: bool = True,
    ):
        """
        Initialize the directional model.

        Args:
            base_model_name: Base sentence-transformers model
            output_dim: Output embedding dimension
            num_angles: Number of RGDB angle bins
            shared_base: Whether to share the base model across angles
        """
        super().__init__()

        from sentence_transformers import SentenceTransformer

        self.base_model = SentenceTransformer(base_model_name)
        self.base_dim = self.base_model.get_sentence_embedding_dimension()
        self.output_dim = output_dim
        self.num_angles = num_angles

        # Per-angle projection heads
        self.angle_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.base_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
            for _ in range(num_angles)
        ])

    def forward(
        self,
        texts: List[str],
        angle_bin: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode texts with optional angle-specific projection.

        Args:
            texts: List of texts
            angle_bin: Angle bin for projection (None = base embedding)

        Returns:
            Embeddings [batch_size, output_dim]
        """
        # Get base embeddings
        base_emb = self.base_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        if angle_bin is not None:
            # Apply angle-specific projection
            emb = self.angle_projections[angle_bin](base_emb)
        else:
            # Average all projections (or just return base)
            emb = base_emb

        # Normalize
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def encode_with_angle(
        self,
        texts: List[str],
        angle_bins: List[int],
    ) -> torch.Tensor:
        """
        Encode each text with its corresponding angle bin.

        Args:
            texts: List of texts
            angle_bins: Angle bin for each text

        Returns:
            Embeddings [batch_size, output_dim]
        """
        base_emb = self.base_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        # Apply per-sample angle projections
        embeddings = []
        for i, angle in enumerate(angle_bins):
            emb = self.angle_projections[angle](base_emb[i:i+1])
            embeddings.append(emb)

        emb = torch.cat(embeddings, dim=0)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
