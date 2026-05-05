#!/usr/bin/env python3
"""
Example: Generate embeddings from documents using pretrained models.

This is the simplest workflow - no training required.
Just load documents, chunk them, and generate embeddings.
"""

from pathlib import Path

from rgdb_embeddings import (
    DocumentLoader,
    PretrainedEmbedder,
    TextChunker,
    export_to_rgdb,
)


def main():
    """Generate embeddings from documents."""
    # Configuration
    docs_path = Path("./sample_docs")  # Your documents directory
    output_path = Path("./embeddings.emb")
    chunk_size = 512
    model_name = "all-MiniLM-L6-v2"  # Fast, 384 dimensions

    # Step 1: Load documents
    print(f"Loading documents from: {docs_path}")
    loader = DocumentLoader()

    # Create sample documents if they don't exist
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        (docs_path / "sample1.txt").write_text(
            "Machine learning is a subset of artificial intelligence. "
            "It enables computers to learn from data without being explicitly programmed."
        )
        (docs_path / "sample2.txt").write_text(
            "Neural networks are computing systems inspired by biological neural networks. "
            "They consist of interconnected nodes that process information."
        )
        print("Created sample documents")

    # Load and chunk documents
    chunks = loader.load_and_chunk(docs_path, chunk_size=chunk_size)
    print(f"Created {len(chunks)} chunks from documents")

    # Step 2: Generate embeddings
    print(f"Loading model: {model_name}")
    embedder = PretrainedEmbedder(model_name)

    print("Generating embeddings...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress=True)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Step 3: Export to RGDB format
    export_to_rgdb(embeddings, output_path)
    print(f"Saved embeddings to: {output_path}")

    # Optional: Print some info
    print("\nChunk preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  {i}: {chunk.text[:80]}...")


if __name__ == "__main__":
    main()
