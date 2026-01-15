#!/usr/bin/env python3
"""
Example: Hybrid training with both documents and knowledge triples.

This workflow:
1. Generates text embeddings from documents
2. Enhances them with knowledge graph relationships
3. Produces embeddings that capture both content and structure
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from rgdb_embeddings import (
    DocumentLoader,
    PretrainedEmbedder,
    TripleLoader,
    export_to_rgdb,
)


def create_sample_data(tmpdir: Path):
    """Create sample documents and triples."""
    # Create documents
    docs_path = tmpdir / "docs"
    docs_path.mkdir()

    docs = {
        "ml_intro.txt": """
        Machine learning is a branch of artificial intelligence that focuses on
        building systems that can learn from data. These systems improve their
        performance over time without being explicitly programmed.
        """,
        "neural_nets.txt": """
        Neural networks are computing systems inspired by biological neural networks
        in the brain. They consist of layers of interconnected nodes that process
        information using connectionist approaches to computation.
        """,
        "deep_learning.txt": """
        Deep learning is a subset of machine learning that uses neural networks
        with many layers. These deep neural networks have revolutionized fields
        like computer vision and natural language processing.
        """,
    }

    for filename, content in docs.items():
        (docs_path / filename).write_text(content.strip())

    # Create triples
    triples_path = tmpdir / "knowledge.csv"
    triples = [
        "head,relation,tail",
        "machine_learning,is_a,artificial_intelligence",
        "neural_network,is_a,computing_system",
        "deep_learning,is_a,machine_learning",
        "neural_network,part_of,deep_learning",
        "machine_learning,requires,data",
        "deep_learning,requires,neural_network",
        "deep_learning,enables,computer_vision",
        "deep_learning,enables,nlp",
    ]
    triples_path.write_text("\n".join(triples))

    return docs_path, triples_path


def main():
    """Run hybrid training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        docs_path, triples_path = create_sample_data(tmpdir)

        print("=" * 60)
        print("HYBRID EMBEDDING TRAINING")
        print("=" * 60)

        # Step 1: Generate document embeddings
        print("\n[1/4] Loading and chunking documents...")
        doc_loader = DocumentLoader()
        chunks = doc_loader.load_and_chunk(docs_path, chunk_size=256)
        print(f"Created {len(chunks)} chunks")

        print("\n[2/4] Generating text embeddings...")
        embedder = PretrainedEmbedder("all-MiniLM-L6-v2")
        text_embeddings = embedder.encode([c.text for c in chunks])
        print(f"Text embedding shape: {text_embeddings.shape}")

        # Step 2: Load knowledge graph
        print("\n[3/4] Loading knowledge graph...")
        triple_loader = TripleLoader()
        triples = triple_loader.load(triples_path)
        print(f"Loaded {len(triples)} triples")

        # Map to bins
        vocab = triple_loader.build_entity_vocab(triples)
        mapped = triple_loader.map_relations_to_bins(triples, vocab)
        print(f"Entity vocabulary: {len(vocab)} entities")

        # Show the knowledge graph structure
        print("\nKnowledge graph:")
        for t in mapped[:5]:
            print(f"  {t.head} --[bin {t.angle_bin}]--> {t.tail}")

        # Step 3: Combine text and graph information
        # In a full implementation, you would:
        # 1. Align document chunks with entities
        # 2. Use GNN or RotatE to incorporate graph structure
        # 3. Project both into a shared space

        print("\n[4/4] Creating hybrid embeddings...")

        # For this example, we demonstrate a simple concatenation approach
        # A production system would use the RotatEWithText model or similar

        # Simulate graph-enhanced embeddings by adding position-based info
        # (In practice, use GNN propagation or relation rotations)
        num_chunks = len(chunks)
        graph_component = np.random.randn(num_chunks, 64).astype(np.float32)
        graph_component = graph_component / np.linalg.norm(graph_component, axis=1, keepdims=True)

        # Combine text and graph
        alpha = 0.8  # Weight for text
        hybrid_dim = text_embeddings.shape[1] + graph_component.shape[1]
        hybrid_embeddings = np.concatenate([
            text_embeddings * alpha,
            graph_component * (1 - alpha),
        ], axis=1)

        # Normalize
        norms = np.linalg.norm(hybrid_embeddings, axis=1, keepdims=True)
        hybrid_embeddings = hybrid_embeddings / norms

        print(f"Hybrid embedding shape: {hybrid_embeddings.shape}")

        # Step 4: Export
        output_path = tmpdir / "hybrid_embeddings.emb"
        export_to_rgdb(hybrid_embeddings, output_path)
        print(f"\nExported to: {output_path}")

        # Verify
        from rgdb_embeddings import validate_embeddings
        validation = validate_embeddings(hybrid_embeddings)
        print(f"\nValidation: {validation['stats']}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print("""
Next steps for production:
1. Align document chunks with knowledge graph entities
2. Use RotatEWithText or DirectionalFineTuneModel for proper fusion
3. Train with both contrastive and relation prediction objectives
4. Evaluate with downstream tasks (search, QA, etc.)
        """)


if __name__ == "__main__":
    main()
