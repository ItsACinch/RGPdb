"""Tests for export module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rgdb_embeddings.export import (
    export_to_rgdb,
    load_from_rgdb,
    validate_embeddings,
)
from rgdb_embeddings.export.rgdb_format import get_file_info


class TestExportToRGDB:
    """Tests for export_to_rgdb function."""

    def test_basic_export(self):
        """Test basic export functionality."""
        embeddings = np.random.randn(100, 64).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            export_to_rgdb(embeddings, f.name)

            # Verify file was created
            assert Path(f.name).exists()
            assert Path(f.name).stat().st_size > 0

    def test_export_normalize(self):
        """Test that normalization works."""
        embeddings = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            export_to_rgdb(embeddings, f.name, normalize=True)
            loaded = load_from_rgdb(f.name)

            # Check that vectors are normalized
            norms = np.linalg.norm(loaded, axis=1)
            np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_export_no_normalize(self):
        """Test export without normalization."""
        embeddings = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            export_to_rgdb(embeddings, f.name, normalize=False)
            loaded = load_from_rgdb(f.name)

            np.testing.assert_allclose(loaded, embeddings)

    def test_invalid_shape(self):
        """Test that invalid shapes raise error."""
        embeddings_1d = np.array([1.0, 2.0, 3.0])

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            with pytest.raises(ValueError, match="2D"):
                export_to_rgdb(embeddings_1d, f.name)

    def test_empty_embeddings(self):
        """Test that empty embeddings raise error."""
        embeddings = np.array([]).reshape(0, 64)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            with pytest.raises(ValueError, match="empty"):
                export_to_rgdb(embeddings, f.name)


class TestLoadFromRGDB:
    """Tests for load_from_rgdb function."""

    def test_roundtrip(self):
        """Test export and load roundtrip."""
        original = np.random.randn(50, 128).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            export_to_rgdb(original, f.name, normalize=False)
            loaded = load_from_rgdb(f.name)

            np.testing.assert_allclose(loaded, original, atol=1e-6)

    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_from_rgdb("/nonexistent/path/file.emb")

    def test_corrupted_file(self):
        """Test that corrupted file raises error."""
        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            f.write(b"corrupted")
            f.flush()

            # File is too small or has invalid header
            with pytest.raises(ValueError):
                load_from_rgdb(f.name)


class TestValidateEmbeddings:
    """Tests for validate_embeddings function."""

    def test_valid_embeddings(self):
        """Test validation of valid embeddings."""
        embeddings = np.random.randn(100, 64).astype(np.float32)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        results = validate_embeddings(embeddings)

        assert results["valid"] is True
        assert len(results["errors"]) == 0
        assert results["stats"]["num_nodes"] == 100
        assert results["stats"]["dim"] == 64
        assert results["stats"]["is_normalized"] is True

    def test_nan_detection(self):
        """Test detection of NaN values."""
        embeddings = np.array([[1.0, np.nan], [0.5, 0.5]], dtype=np.float32)

        results = validate_embeddings(embeddings)

        assert results["valid"] is False
        assert any("NaN" in e for e in results["errors"])

    def test_inf_detection(self):
        """Test detection of Inf values."""
        embeddings = np.array([[1.0, np.inf], [0.5, 0.5]], dtype=np.float32)

        results = validate_embeddings(embeddings)

        assert results["valid"] is False
        assert any("Inf" in e for e in results["errors"])

    def test_zero_vector_warning(self):
        """Test warning for zero vectors."""
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        results = validate_embeddings(embeddings)

        assert any("zero" in w for w in results["warnings"])


class TestGetFileInfo:
    """Tests for get_file_info function."""

    def test_file_info(self):
        """Test getting file information."""
        embeddings = np.random.randn(50, 64).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            export_to_rgdb(embeddings, f.name, normalize=False)

            info = get_file_info(f.name)

            assert info["exists"] is True
            assert info["num_nodes"] == 50
            assert info["dim"] == 64
            assert info["size_matches"] is True

    def test_nonexistent_file(self):
        """Test info for nonexistent file."""
        info = get_file_info("/nonexistent/path/file.emb")

        assert info["exists"] is False
