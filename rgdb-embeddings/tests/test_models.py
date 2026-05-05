"""Tests for model modules."""

import pytest
import torch
import numpy as np


class TestRotatEForRGDB:
    """Tests for RotatE model."""

    def test_model_creation(self):
        """Test model initialization."""
        from rgdb_embeddings.models import RotatEForRGDB

        model = RotatEForRGDB(num_nodes=100, dim=64, num_angles=16)

        assert model.num_nodes == 100
        assert model.dim == 64
        assert model.num_angles == 16

    def test_forward_without_rotation(self):
        """Test forward pass without rotation."""
        from rgdb_embeddings.models import RotatEForRGDB

        model = RotatEForRGDB(num_nodes=100, dim=64)
        node_ids = torch.tensor([0, 1, 2])

        embeddings = model(node_ids)

        assert embeddings.shape == (3, 64)

    def test_forward_with_rotation(self):
        """Test forward pass with rotation."""
        from rgdb_embeddings.models import RotatEForRGDB

        model = RotatEForRGDB(num_nodes=100, dim=64)
        node_ids = torch.tensor([0, 1, 2])

        emb_no_rot = model(node_ids)
        emb_with_rot = model(node_ids, angle_bin=0)

        # Rotated should be different from non-rotated
        assert not torch.allclose(emb_no_rot, emb_with_rot)

    def test_score_triple(self):
        """Test triple scoring."""
        from rgdb_embeddings.models import RotatEForRGDB

        model = RotatEForRGDB(num_nodes=100, dim=64)

        head = torch.tensor([0, 1])
        tail = torch.tensor([2, 3])
        angle_bins = torch.tensor([0, 1])

        scores = model.score_triple(head, tail, angle_bins)

        assert scores.shape == (2,)
        assert torch.all(scores >= 0)  # Distance-based, should be non-negative

    def test_get_all_embeddings(self):
        """Test getting all embeddings."""
        from rgdb_embeddings.models import RotatEForRGDB

        model = RotatEForRGDB(num_nodes=50, dim=32)
        all_emb = model.get_all_embeddings()

        assert all_emb.shape == (50, 32)


class TestRGDBGraphEncoder:
    """Tests for GNN encoder."""

    def test_model_creation(self):
        """Test model initialization."""
        from rgdb_embeddings.models import RGDBGraphEncoder

        model = RGDBGraphEncoder(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_angles=16,
            num_layers=2,
        )

        assert model.input_dim == 64
        assert model.output_dim == 64
        assert model.num_layers == 2

    def test_forward(self):
        """Test forward pass."""
        from rgdb_embeddings.models import RGDBGraphEncoder

        model = RGDBGraphEncoder(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_angles=16,
        )

        # Create test data
        num_nodes = 10
        num_edges = 20

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_angle_bins = torch.randint(0, 16, (num_edges,))

        output = model(x, edge_index, edge_angle_bins)

        assert output.shape == (num_nodes, 64)

        # Output should be normalized
        norms = torch.norm(output, dim=1)
        torch.testing.assert_close(norms, torch.ones(num_nodes), atol=1e-5, rtol=1e-5)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_triplet_loss(self):
        """Test triplet loss computation."""
        from rgdb_embeddings.training import TripletLoss

        loss_fn = TripletLoss(margin=0.5)

        anchor = torch.randn(8, 64)
        positive = anchor + 0.1 * torch.randn(8, 64)  # Similar
        negative = torch.randn(8, 64)  # Random

        loss = loss_fn(anchor, positive, negative)

        assert loss.shape == ()
        assert loss >= 0

    def test_rotate_loss(self):
        """Test RotatE loss computation."""
        from rgdb_embeddings.training import RotatELoss

        loss_fn = RotatELoss(margin=1.0)

        pos_scores = torch.randn(8).abs()  # Positive distances
        neg_scores = torch.randn(8, 10).abs()  # Negative distances

        loss = loss_fn(pos_scores, neg_scores)

        assert loss.shape == ()

    def test_directional_contrastive_loss(self):
        """Test directional contrastive loss."""
        from rgdb_embeddings.training import DirectionalContrastiveLoss

        loss_fn = DirectionalContrastiveLoss(margin=1.0, temperature=0.07)

        rotated_anchors = torch.randn(8, 64)
        positives = torch.randn(8, 64)
        negatives = torch.randn(8, 5, 64)

        loss = loss_fn(rotated_anchors, positives, negatives)

        assert loss.shape == ()
        assert loss >= 0
