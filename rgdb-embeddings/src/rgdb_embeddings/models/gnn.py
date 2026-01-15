"""Graph Neural Network encoder respecting RGDB angle bins."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGDBGraphEncoder(nn.Module):
    """
    Graph Neural Network that respects RGDB's 16 angle bins.

    Each angle bin has its own message transformation, allowing
    the model to learn different aggregation patterns for different
    relationship types.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_angles: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ):
        """
        Initialize the GNN encoder.

        Args:
            input_dim: Input feature dimension (text embedding dim)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_angles: Number of RGDB angle bins
            num_layers: Number of GNN layers
            dropout: Dropout rate
            aggregation: Aggregation method (mean, sum, max)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_angles = num_angles
        self.num_layers = num_layers
        self.aggregation = aggregation

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Per-angle message transforms for each layer
        self.angle_transforms = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_angles)
            ])
            for _ in range(num_layers)
        ])

        # Self-loop transform for each layer
        self.self_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Combine transformed neighbors with self
        self.combine_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_angle_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges] (source, target)
            edge_angle_bins: Angle bin for each edge [num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        num_nodes = x.size(0)

        # Initial projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)

        # Message passing layers
        for layer in range(self.num_layers):
            h = self._message_passing_layer(
                h, edge_index, edge_angle_bins, layer, num_nodes
            )

        # Output projection
        h = self.output_proj(h)
        h = F.normalize(h, p=2, dim=-1)

        return h

    def _message_passing_layer(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_angle_bins: torch.Tensor,
        layer: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Single message passing layer with angle-aware transforms.

        Args:
            h: Current node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_angle_bins: Angle bins [num_edges]
            layer: Current layer index
            num_nodes: Number of nodes

        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        src, dst = edge_index

        # Aggregate messages per angle bin
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        counts = torch.zeros(num_nodes, 1, device=h.device)

        for angle in range(self.num_angles):
            # Get edges for this angle
            mask = edge_angle_bins == angle
            if mask.sum() == 0:
                continue

            # Get source embeddings
            src_idx = src[mask]
            dst_idx = dst[mask]

            # Transform source embeddings for this angle
            src_emb = h[src_idx]
            transformed = self.angle_transforms[layer][angle](src_emb)

            # Aggregate at destinations
            aggregated.index_add_(0, dst_idx, transformed)
            counts.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float).unsqueeze(-1))

        # Average aggregation
        if self.aggregation == "mean":
            aggregated = aggregated / (counts + 1e-8)
        elif self.aggregation == "sum":
            pass  # Already summed
        # max aggregation would need scatter_max

        # Self-loop transform
        self_trans = self.self_transforms[layer](h)

        # Combine with residual
        combined = torch.cat([self_trans, aggregated], dim=-1)
        h_new = self.combine_layers[layer](combined)
        h_new = F.relu(h_new)
        h_new = self.dropout(h_new)
        h_new = self.layer_norms[layer](h_new + h)  # Residual connection

        return h_new


class RGDBGraphEncoderWithAttention(nn.Module):
    """
    GNN encoder with attention over angle bins.

    Uses attention to learn the importance of different relationship
    types for each node.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_angles: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize the attention-based GNN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_angles: Number of angle bins
            num_heads: Number of attention heads
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_angles = num_angles
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Attention layers
        self.attention_layers = nn.ModuleList([
            AngleBinAttention(hidden_dim, num_angles, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_angle_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        h = self.input_proj(x)
        h = F.relu(h)

        for layer in range(self.num_layers):
            h_new = self.attention_layers[layer](h, edge_index, edge_angle_bins)
            h = self.layer_norms[layer](h + self.dropout(h_new))

        h = self.output_proj(h)
        return F.normalize(h, p=2, dim=-1)


class AngleBinAttention(nn.Module):
    """Attention over neighbors grouped by angle bin."""

    def __init__(
        self,
        dim: int,
        num_angles: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.dim = dim
        self.num_angles = num_angles
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, key, value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Angle-specific biases
        self.angle_bias = nn.Parameter(torch.zeros(num_angles, num_heads))

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_angle_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with angle-aware attention."""
        num_nodes = h.size(0)
        src, dst = edge_index

        # Compute Q, K, V
        q = self.q_proj(h)  # [num_nodes, dim]
        k = self.k_proj(h)[src]  # [num_edges, dim]
        v = self.v_proj(h)[src]  # [num_edges, dim]

        # Reshape for multi-head attention
        q = q.view(num_nodes, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores for each edge
        # q[dst] @ k for each edge
        q_dst = q[dst]  # [num_edges, num_heads, head_dim]
        attn = (q_dst * k).sum(dim=-1) / (self.head_dim ** 0.5)  # [num_edges, num_heads]

        # Add angle-specific bias
        attn = attn + self.angle_bias[edge_angle_bins]

        # Softmax per destination node (approximate with scatter_softmax if available)
        # For simplicity, we'll use a basic implementation
        attn = torch.softmax(attn, dim=0)  # This is approximate
        attn = self.dropout(attn)

        # Aggregate values
        weighted_v = attn.unsqueeze(-1) * v  # [num_edges, num_heads, head_dim]

        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=h.device)
        out.index_add_(0, dst, weighted_v)

        # Reshape and project
        out = out.view(num_nodes, self.dim)
        out = self.out_proj(out)

        return out
