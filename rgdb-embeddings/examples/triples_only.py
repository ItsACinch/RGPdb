#!/usr/bin/env python3
"""
Example: Train RotatE embeddings from knowledge triples.

This workflow trains relation-aware embeddings that understand
RGDB's 16 angle bins for different relationship types.
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgdb_embeddings import (
    RotatEForRGDB,
    RotatETrainer,
    TripleDataset,
    TripleLoader,
    TrainingConfig,
    export_to_rgdb,
)
from rgdb_embeddings.export.rgdb_format import export_vocab
from rgdb_embeddings.training import LoggingCallback


def create_sample_triples(path: Path):
    """Create sample knowledge triples."""
    triples = [
        # IsA relationships (bin 0)
        "machine_learning,is_a,artificial_intelligence",
        "deep_learning,is_a,machine_learning",
        "neural_network,is_a,model",
        "cnn,is_a,neural_network",
        "rnn,is_a,neural_network",
        # Requires relationships (bin 1)
        "machine_learning,requires,data",
        "neural_network,requires,training",
        "deep_learning,requires,gpu",
        # Contains relationships (bin 6)
        "neural_network,contains,layers",
        "cnn,contains,convolution",
        "rnn,contains,hidden_state",
        # SimilarTo relationships (bin 10)
        "cnn,similar_to,rnn",
        "deep_learning,similar_to,machine_learning",
        # Causes relationships (bin 4)
        "training,causes,learning",
        "overfitting,causes,poor_generalization",
    ]

    with open(path, "w") as f:
        f.write("head,relation,tail\n")
        f.write("\n".join(triples))


def main():
    """Train RotatE embeddings."""
    # Configuration
    embedding_dim = 128  # Smaller for demo
    epochs = 5
    batch_size = 8
    learning_rate = 1e-3

    # Create sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        triples_path = Path(tmpdir) / "triples.csv"
        output_path = Path(tmpdir) / "embeddings.emb"
        vocab_path = Path(tmpdir) / "vocab.json"

        create_sample_triples(triples_path)
        print(f"Created sample triples at: {triples_path}")

        # Step 1: Load triples
        loader = TripleLoader()
        triples = loader.load(triples_path)
        print(f"Loaded {len(triples)} triples")

        # Build vocabulary
        vocab = loader.build_entity_vocab(triples)
        num_entities = len(vocab)
        print(f"Vocabulary: {num_entities} entities")
        print(f"Entities: {list(vocab.keys())}")

        # Map relations to angle bins
        mapped = loader.map_relations_to_bins(triples, vocab)

        # Show angle bin distribution
        print("\nAngle bin distribution:")
        bin_stats = loader.get_angle_bin_statistics(mapped)
        for bin_idx, count in bin_stats.items():
            print(f"  Bin {bin_idx}: {count} triples")

        # Step 2: Create dataset and loader
        dataset = TripleDataset(mapped, num_entities, negative_samples=5)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Step 3: Create model
        model = RotatEForRGDB(
            num_nodes=num_entities,
            dim=embedding_dim,
            num_angles=16,
        )
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Step 4: Train
        config = TrainingConfig(
            mode="rotate",
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        trainer = RotatETrainer(model, config, callbacks=[LoggingCallback(log_every=10)])

        print("\nStarting training...")
        trainer.train(data_loader)

        # Step 5: Export
        embeddings = trainer.get_embeddings()
        export_to_rgdb(embeddings, output_path)
        export_vocab(vocab, vocab_path)

        print(f"\nExported embeddings to: {output_path}")
        print(f"Exported vocabulary to: {vocab_path}")
        print(f"Embedding shape: {embeddings.shape}")

        # Step 6: Test some similarities
        print("\nTesting learned embeddings:")
        test_pairs = [
            ("machine_learning", "deep_learning"),
            ("cnn", "rnn"),
            ("training", "data"),
        ]

        for e1, e2 in test_pairs:
            if e1 in vocab and e2 in vocab:
                idx1, idx2 = vocab[e1], vocab[e2]
                sim = torch.cosine_similarity(
                    torch.tensor(embeddings[idx1]).unsqueeze(0),
                    torch.tensor(embeddings[idx2]).unsqueeze(0),
                ).item()
                print(f"  {e1} <-> {e2}: {sim:.3f}")


if __name__ == "__main__":
    main()
