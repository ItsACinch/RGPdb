"""RotatE-style directional embeddings for RGDB."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatEForRGDB(nn.Module):
    """
    RotatE-style embeddings that respect RGDB's 16 angle bins.

    In RotatE, relations are modeled as rotations in complex space:
    tail = head * rotation(relation)

    We adapt this for RGDB by learning a rotation for each angle bin,
    so embeddings can be "rotated" to different relationship directions.
    """

    def __init__(
        self,
        num_nodes: int,
        dim: int,
        num_angles: int = 16,
        margin: float = 1.0,
        init_scale: float = 0.1,
    ):
        """
        Initialize the RotatE model.

        Args:
            num_nodes: Number of entities/nodes
            dim: Embedding dimension (will be split into real/imag pairs)
            num_angles: Number of RGDB angle bins (default 16)
            margin: Margin for ranking loss
            init_scale: Scale for initialization
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.dim = dim
        self.num_angles = num_angles
        self.margin = margin

        # Entity embeddings
        self.node_embeddings = nn.Embedding(num_nodes, dim)

        # Rotation phases for each angle bin
        # Each angle bin has dim//2 rotation phases (one per complex dimension pair)
        self.angle_phases = nn.Parameter(torch.randn(num_angles, dim // 2) * init_scale)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings uniformly on a unit sphere."""
        nn.init.uniform_(
            self.node_embeddings.weight,
            -1.0 / math.sqrt(self.dim),
            1.0 / math.sqrt(self.dim),
        )

    def forward(
        self,
        node_ids: torch.Tensor,
        angle_bin: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get embeddings, optionally rotated by an angle bin.

        Args:
            node_ids: Node IDs [batch_size] or [batch_size, seq_len]
            angle_bin: Optional angle bin to apply rotation

        Returns:
            Embeddings [batch_size, dim] or [batch_size, seq_len, dim]
        """
        emb = self.node_embeddings(node_ids)

        if angle_bin is not None:
            emb = self._rotate(emb, angle_bin)

        return emb

    def _rotate(self, emb: torch.Tensor, angle_bin: int) -> torch.Tensor:
        """
        Apply rotation for the given angle bin.

        Uses complex multiplication: (re + i*im) * (cos + i*sin)
        Result: (re*cos - im*sin) + i*(re*sin + im*cos)

        Args:
            emb: Embeddings [..., dim]
            angle_bin: Angle bin index

        Returns:
            Rotated embeddings [..., dim]
        """
        # Split into real and imaginary parts
        re, im = emb.chunk(2, dim=-1)

        # Get rotation phases for this angle bin
        phases = self.angle_phases[angle_bin]
        cos_phases = torch.cos(phases)
        sin_phases = torch.sin(phases)

        # Complex rotation
        new_re = re * cos_phases - im * sin_phases
        new_im = re * sin_phases + im * cos_phases

        return torch.cat([new_re, new_im], dim=-1)

    def rotate_batch(
        self,
        emb: torch.Tensor,
        angle_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply different rotations to each sample in a batch.

        Args:
            emb: Embeddings [batch_size, dim]
            angle_bins: Angle bin indices [batch_size]

        Returns:
            Rotated embeddings [batch_size, dim]
        """
        re, im = emb.chunk(2, dim=-1)

        # Get phases for each sample
        phases = self.angle_phases[angle_bins]  # [batch_size, dim//2]
        cos_phases = torch.cos(phases)
        sin_phases = torch.sin(phases)

        new_re = re * cos_phases - im * sin_phases
        new_im = re * sin_phases + im * cos_phases

        return torch.cat([new_re, new_im], dim=-1)

    def score_triple(
        self,
        head_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        angle_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using RotatE distance.

        In RotatE, the score is: ||head * rotation - tail||

        Args:
            head_ids: Head entity IDs [batch_size]
            tail_ids: Tail entity IDs [batch_size]
            angle_bins: Relation angle bins [batch_size]

        Returns:
            Scores [batch_size] (lower is better)
        """
        head_emb = self.node_embeddings(head_ids)
        tail_emb = self.node_embeddings(tail_ids)

        # Rotate head embeddings
        rotated_head = self.rotate_batch(head_emb, angle_bins)

        # L2 distance between rotated head and tail
        diff = rotated_head - tail_emb
        re, im = diff.chunk(2, dim=-1)

        # Use L2 norm of complex number: sqrt(re^2 + im^2)
        score = torch.sqrt(re ** 2 + im ** 2 + 1e-8).sum(dim=-1)

        return score

    def loss(
        self,
        head_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        angle_bins: torch.Tensor,
        negative_tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.

        Args:
            head_ids: Head entity IDs [batch_size]
            tail_ids: Positive tail IDs [batch_size]
            angle_bins: Relation angle bins [batch_size]
            negative_tail_ids: Negative tail IDs [batch_size, num_negatives]

        Returns:
            Loss scalar
        """
        batch_size = head_ids.size(0)
        num_negatives = negative_tail_ids.size(1)

        # Positive scores
        pos_scores = self.score_triple(head_ids, tail_ids, angle_bins)

        # Negative scores
        head_expanded = head_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        neg_tails_flat = negative_tail_ids.reshape(-1)
        bins_expanded = angle_bins.unsqueeze(1).expand(-1, num_negatives).reshape(-1)

        neg_scores = self.score_triple(head_expanded, neg_tails_flat, bins_expanded)
        neg_scores = neg_scores.view(batch_size, num_negatives)

        # Margin loss: max(0, margin + pos_score - neg_score)
        pos_scores = pos_scores.unsqueeze(1)  # [batch_size, 1]
        loss = F.relu(self.margin + pos_scores - neg_scores).mean()

        return loss

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all node embeddings."""
        return self.node_embeddings.weight

    def get_rotated_embeddings(self, angle_bin: int) -> torch.Tensor:
        """Get all node embeddings rotated by a specific angle bin."""
        all_ids = torch.arange(self.num_nodes, device=self.node_embeddings.weight.device)
        return self._rotate(self.node_embeddings(all_ids), angle_bin)


class RotatEWithText(nn.Module):
    """
    RotatE model that incorporates text embeddings.

    Combines a pretrained text encoder with RotatE rotations.
    """

    def __init__(
        self,
        text_encoder,
        dim: int = 256,
        num_angles: int = 16,
        project_text: bool = True,
    ):
        """
        Initialize the hybrid model.

        Args:
            text_encoder: A text encoder (e.g., PretrainedEmbedder)
            dim: Output embedding dimension
            num_angles: Number of RGDB angle bins
            project_text: Whether to project text embeddings to dim
        """
        super().__init__()

        self.text_encoder = text_encoder
        self.dim = dim
        self.num_angles = num_angles

        # Text projection (if text encoder has different dimension)
        text_dim = text_encoder.get_embedding_dimension()
        if project_text and text_dim != dim:
            self.text_projection = nn.Linear(text_dim, dim)
        else:
            self.text_projection = None

        # Rotation phases
        self.angle_phases = nn.Parameter(torch.randn(num_angles, dim // 2) * 0.1)

    def encode_text(self, texts):
        """Encode texts and optionally project."""
        import numpy as np

        # Get embeddings from text encoder
        emb = self.text_encoder.encode(texts)

        # Convert to tensor
        emb = torch.tensor(emb, dtype=torch.float32)

        # Project if needed
        if self.text_projection is not None:
            emb = self.text_projection(emb)

        return emb

    def forward(self, texts, angle_bin: Optional[int] = None):
        """
        Encode texts and optionally rotate.

        Args:
            texts: List of texts
            angle_bin: Optional angle bin for rotation

        Returns:
            Embeddings
        """
        emb = self.encode_text(texts)

        if angle_bin is not None:
            re, im = emb.chunk(2, dim=-1)
            phases = self.angle_phases[angle_bin]
            cos_phases = torch.cos(phases)
            sin_phases = torch.sin(phases)
            new_re = re * cos_phases - im * sin_phases
            new_im = re * sin_phases + im * cos_phases
            emb = torch.cat([new_re, new_im], dim=-1)

        return emb
