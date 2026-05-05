"""Tests for configuration module."""

import pytest

from rgdb_embeddings.config import (
    DataConfig,
    TrainingConfig,
    RELATION_TO_BIN,
    relation_to_bin,
    normalize_relation,
)


class TestRelationMapping:
    """Tests for relation to angle bin mapping."""

    def test_known_relations(self):
        """Test mapping of known relation types."""
        assert relation_to_bin("is_a") == 0
        assert relation_to_bin("requires") == 1
        assert relation_to_bin("related_to") == 2
        assert relation_to_bin("causes") == 4
        assert relation_to_bin("contains") == 6
        assert relation_to_bin("part_of") == 8
        assert relation_to_bin("similar_to") == 10
        assert relation_to_bin("opposite_of") == 12
        assert relation_to_bin("enables") == 14
        assert relation_to_bin("conflicts_with") == 15

    def test_case_insensitive(self):
        """Test that relation mapping is case insensitive."""
        assert relation_to_bin("IS_A") == 0
        assert relation_to_bin("IsA") == 0
        assert relation_to_bin("is_a") == 0

    def test_unknown_relation_default(self):
        """Test that unknown relations return default bin."""
        assert relation_to_bin("unknown_relation") == 2
        assert relation_to_bin("foo_bar") == 2

    def test_custom_default(self):
        """Test custom default value."""
        assert relation_to_bin("unknown", default=5) == 5

    def test_normalize_relation(self):
        """Test relation normalization."""
        assert normalize_relation("Part Of") == "part_of"
        assert normalize_relation("part-of") == "part_of"
        assert normalize_relation("PART_OF") == "part_of"


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.documents_path is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(chunk_size=256, chunk_overlap=25)
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.mode == "pretrained"
        assert config.embedding_dim == 384
        assert config.batch_size == 32
        assert config.epochs == 10

    def test_mode_options(self):
        """Test different training modes."""
        for mode in ["pretrained", "finetune", "rotate", "gnn"]:
            config = TrainingConfig(mode=mode)
            assert config.mode == mode
