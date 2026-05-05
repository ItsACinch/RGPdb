"""Loss functions for embedding training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet margin loss for contrastive learning.

    Ensures that anchor-positive distance < anchor-negative distance by a margin.
    """

    def __init__(self, margin: float = 0.5, distance: str = "cosine"):
        """
        Initialize triplet loss.

        Args:
            margin: Minimum margin between positive and negative distances
            distance: Distance metric (cosine or euclidean)
        """
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim]

        Returns:
            Loss scalar
        """
        if self.distance == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        else:  # euclidean
            pos_dist = torch.norm(anchor - positive, p=2, dim=-1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=-1)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class RotatELoss(nn.Module):
    """
    RotatE margin-based ranking loss.

    Score function: ||head * rotation - tail||

    Loss: margin + positive_score - negative_score
    """

    def __init__(self, margin: float = 1.0, gamma: float = 12.0):
        """
        Initialize RotatE loss.

        Args:
            margin: Margin for ranking loss
            gamma: Fixed margin / temperature (from RotatE paper)
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RotatE ranking loss.

        Args:
            positive_scores: Scores for positive triples [batch_size]
                            (lower is better for distance-based scores)
            negative_scores: Scores for negative triples [batch_size, num_negatives]

        Returns:
            Loss scalar
        """
        # Convert distance to score (higher is better)
        pos_score = self.gamma - positive_scores
        neg_score = self.gamma - negative_scores

        # Softmax over negatives
        neg_log_prob = F.logsigmoid(-neg_score).mean(dim=-1)
        pos_log_prob = F.logsigmoid(pos_score)

        loss = -(pos_log_prob + neg_log_prob).mean()
        return loss


class DirectionalContrastiveLoss(nn.Module):
    """
    Contrastive loss that respects RGDB relationship directions.

    For a triple (head, relation, tail):
    - Rotated head should be close to tail
    - Rotated head should be far from negative samples
    """

    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.07,
        hard_negative_weight: float = 1.0,
    ):
        """
        Initialize directional contrastive loss.

        Args:
            margin: Margin for margin-based loss
            temperature: Temperature for InfoNCE-style loss
            hard_negative_weight: Extra weight for hard negatives
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        rotated_anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute directional contrastive loss.

        Args:
            rotated_anchors: Anchor embeddings rotated by relation [batch_size, dim]
            positives: Positive (tail) embeddings [batch_size, dim]
            negatives: Negative embeddings [batch_size, num_negatives, dim]

        Returns:
            Loss scalar
        """
        batch_size = rotated_anchors.size(0)

        # Positive similarity
        pos_sim = F.cosine_similarity(rotated_anchors, positives, dim=-1)
        pos_sim = pos_sim / self.temperature

        # Negative similarities
        # rotated_anchors: [batch_size, dim]
        # negatives: [batch_size, num_negatives, dim]
        neg_sim = torch.bmm(
            negatives,
            rotated_anchors.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, num_negatives]
        neg_sim = neg_sim / self.temperature

        # InfoNCE loss
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch, 1+num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=rotated_anchors.device)
        loss = F.cross_entropy(all_sim, labels)

        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (MNRL).

    Uses all other samples in the batch as negatives.
    Very efficient for batch training.
    """

    def __init__(self, scale: float = 20.0):
        """
        Initialize MNRL.

        Args:
            scale: Temperature scaling (higher = sharper distribution)
        """
        super().__init__()
        self.scale = scale

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MNRL loss.

        Args:
            queries: Query embeddings [batch_size, dim]
            positives: Positive document embeddings [batch_size, dim]

        Returns:
            Loss scalar
        """
        # Similarity matrix [batch_size, batch_size]
        similarity = torch.mm(queries, positives.t()) * self.scale

        # Labels: diagonal elements are positives
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Cross-entropy: correct positive should have highest score
        loss = F.cross_entropy(similarity, labels)

        return loss


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weights.

    Useful for hybrid training with multiple objectives.
    """

    def __init__(self, losses: dict, weights: dict = None):
        """
        Initialize combined loss.

        Args:
            losses: Dictionary of {name: loss_module}
            weights: Dictionary of {name: weight} (default: all 1.0)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses}

    def forward(self, **kwargs) -> tuple:
        """
        Compute combined loss.

        Args:
            **kwargs: Arguments for each loss function
                     Format: {loss_name}_args (e.g., triplet_args={'anchor': ..., 'positive': ...})

        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        total_loss = 0.0
        individual_losses = {}

        for name, loss_fn in self.losses.items():
            args_key = f"{name}_args"
            if args_key in kwargs:
                loss_val = loss_fn(**kwargs[args_key])
                weighted_loss = self.weights.get(name, 1.0) * loss_val
                total_loss = total_loss + weighted_loss
                individual_losses[name] = loss_val.item()

        return total_loss, individual_losses
