"""Tests for data loading modules."""

import tempfile
from pathlib import Path

import pytest

from rgdb_embeddings.data import (
    Chunk,
    TextChunker,
    DocumentLoader,
    Triple,
    TripleLoader,
)


class TestTextChunker:
    """Tests for TextChunker."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test sentence that should be chunked into pieces."
        chunks = chunker.chunk_text(text, document_id="test")

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.document_id == "test" for c in chunks)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", document_id="test")
        assert chunks == []

    def test_chunk_metadata(self):
        """Test that metadata is preserved in chunks."""
        chunker = TextChunker(chunk_size=100)
        metadata = {"source": "test_file.txt"}
        chunks = chunker.chunk_text(
            "Some text content here.",
            document_id="test",
            metadata=metadata,
        )

        assert len(chunks) > 0
        assert chunks[0].metadata.get("source") == "test_file.txt"


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self):
        """Test loading a plain text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, this is test content.")
            f.flush()

            loader = DocumentLoader()
            doc = loader.load_file(f.name)

            assert doc.content == "Hello, this is test content."
            assert doc.source_path == Path(f.name)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.txt")

    def test_unsupported_format(self):
        """Test that unsupported formats raise error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"content")

            loader = DocumentLoader()
            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load_file(f.name)

    def test_load_directory(self):
        """Test loading documents from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("Content 1")
            (Path(tmpdir) / "file2.txt").write_text("Content 2")
            (Path(tmpdir) / "ignored.xyz").write_text("Ignored")

            loader = DocumentLoader()
            docs = loader.load_directory(tmpdir)

            assert len(docs) == 2


class TestTripleLoader:
    """Tests for TripleLoader."""

    def test_load_csv(self):
        """Test loading triples from CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("head,relation,tail\n")
            f.write("a,is_a,b\n")
            f.write("c,requires,d\n")
            f.flush()

            loader = TripleLoader()
            triples = loader.load(f.name)

            assert len(triples) == 2
            assert triples[0] == Triple(head="a", relation="is_a", tail="b")

    def test_load_json(self):
        """Test loading triples from JSON."""
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            data = [
                {"head": "a", "relation": "is_a", "tail": "b"},
                {"head": "c", "relation": "requires", "tail": "d"},
            ]
            json.dump(data, f)
            f.flush()

            loader = TripleLoader()
            triples = loader.load(f.name)

            assert len(triples) == 2
            assert triples[0] == Triple(head="a", relation="is_a", tail="b")

    def test_build_vocab(self):
        """Test building entity vocabulary."""
        loader = TripleLoader()
        triples = [
            Triple("a", "rel", "b"),
            Triple("b", "rel", "c"),
        ]

        vocab = loader.build_entity_vocab(triples)

        assert len(vocab) == 3
        assert "a" in vocab
        assert "b" in vocab
        assert "c" in vocab

    def test_map_relations_to_bins(self):
        """Test mapping relations to angle bins."""
        loader = TripleLoader()
        triples = [
            Triple("a", "is_a", "b"),
            Triple("c", "causes", "d"),
        ]

        mapped = loader.map_relations_to_bins(triples)

        assert len(mapped) == 2
        assert mapped[0].angle_bin == 0  # is_a
        assert mapped[1].angle_bin == 4  # causes

    def test_split_train_test(self):
        """Test train/test splitting."""
        loader = TripleLoader()
        triples = [Triple(f"e{i}", "rel", f"e{i+1}") for i in range(100)]

        train, test = loader.split_train_test(triples, test_ratio=0.2)

        assert len(train) == 80
        assert len(test) == 20
