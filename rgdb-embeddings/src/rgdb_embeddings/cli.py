"""Command-line interface for RGDB embeddings."""

import sys
from pathlib import Path
from typing import Optional

import click

from .config import DataConfig, TrainingConfig


@click.group()
@click.version_option(version="0.1.0", prog_name="rgdb-embed")
def cli():
    """RGDB Embeddings - Train embeddings for RGDB graph database."""
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output JSON file for processed chunks",
)
@click.option(
    "--chunk-size",
    default=512,
    type=int,
    help="Target chunk size in tokens (default: 512)",
)
@click.option(
    "--overlap",
    default=50,
    type=int,
    help="Overlap between chunks in tokens (default: 50)",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recursively process subdirectories (default: True)",
)
def process(
    input_path: str,
    output: str,
    chunk_size: int,
    overlap: int,
    recursive: bool,
):
    """
    Process documents into chunks for embedding.

    INPUT_PATH: Path to a document file or directory.
    """
    from rich.console import Console
    from rich.progress import Progress

    from .data import DocumentLoader, save_chunks_json

    console = Console()
    console.print(f"[bold]Processing documents from:[/bold] {input_path}")

    try:
        loader = DocumentLoader()
        path = Path(input_path)

        with Progress() as progress:
            task = progress.add_task("Loading documents...", total=None)

            if path.is_file():
                doc = loader.load_file(path)
                docs = [doc]
            else:
                docs = loader.load_directory(path, recursive=recursive)

            progress.update(task, completed=True)
            console.print(f"Loaded {len(docs)} documents")

        # Chunk documents
        from .data import TextChunker

        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)

        all_chunks = []
        with Progress() as progress:
            task = progress.add_task("Chunking...", total=len(docs))

            for doc in docs:
                chunks = chunker.chunk_text(
                    doc.content,
                    document_id=doc.document_id,
                    metadata=doc.metadata,
                )
                all_chunks.extend(chunks)
                progress.advance(task)

        # Save chunks
        save_chunks_json(all_chunks, output)

        console.print(f"[bold green]Success![/bold green] Created {len(all_chunks)} chunks")
        console.print(f"Output saved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--mode", "-m",
    type=click.Choice(["pretrained", "finetune", "rotate", "gnn"]),
    default="pretrained",
    help="Training mode (default: pretrained)",
)
@click.option(
    "--input", "-i",
    "input_path",
    type=click.Path(exists=True),
    help="Input chunks JSON file (for pretrained/finetune modes)",
)
@click.option(
    "--triples", "-t",
    type=click.Path(exists=True),
    help="Knowledge triples CSV/JSON file (for rotate/gnn modes)",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output .emb file",
)
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    help="Pretrained model name (default: all-MiniLM-L6-v2)",
)
@click.option(
    "--dim",
    default=384,
    type=int,
    help="Embedding dimension (default: 384)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Number of training epochs (default: 10)",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size (default: 32)",
)
@click.option(
    "--lr",
    default=1e-4,
    type=float,
    help="Learning rate (default: 1e-4)",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use (default: cuda)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    help="Directory to save checkpoints",
)
def train(
    mode: str,
    input_path: Optional[str],
    triples: Optional[str],
    output: str,
    model_name: str,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    checkpoint_dir: Optional[str],
):
    """
    Train embeddings with the specified mode.

    Examples:
        # Use pretrained model directly
        rgdb-embed train --mode pretrained --input chunks.json --output embeddings.emb

        # Train RotatE on knowledge triples
        rgdb-embed train --mode rotate --triples knowledge.csv --output embeddings.emb
    """
    from rich.console import Console

    console = Console()

    try:
        import torch

        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
            device = "cpu"

        if mode == "pretrained":
            _train_pretrained(input_path, output, model_name, console)
        elif mode == "finetune":
            _train_finetune(
                input_path, triples, output, model_name, dim,
                epochs, batch_size, lr, device, checkpoint_dir, console
            )
        elif mode == "rotate":
            _train_rotate(
                triples, output, dim, epochs, batch_size, lr,
                device, checkpoint_dir, console
            )
        elif mode == "gnn":
            console.print("[yellow]GNN training not yet implemented[/yellow]")
            console.print("Use --mode rotate for knowledge graph embeddings")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _train_pretrained(input_path: Optional[str], output: str, model_name: str, console):
    """Generate embeddings using pretrained model."""
    if input_path is None:
        console.print("[bold red]Error:[/bold red] --input is required for pretrained mode")
        sys.exit(1)

    from .data import load_chunks_json
    from .export import export_to_rgdb
    from .models import PretrainedEmbedder

    console.print(f"[bold]Loading chunks from:[/bold] {input_path}")
    chunks = load_chunks_json(input_path)
    console.print(f"Loaded {len(chunks)} chunks")

    console.print(f"[bold]Loading model:[/bold] {model_name}")
    embedder = PretrainedEmbedder(model_name)

    console.print("[bold]Generating embeddings...[/bold]")
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress=True)

    console.print(f"[bold]Exporting to:[/bold] {output}")
    export_to_rgdb(embeddings, output)

    console.print(f"[bold green]Success![/bold green] Saved {len(embeddings)} embeddings")
    console.print(f"  Dimensions: {embeddings.shape[1]}")


def _train_finetune(
    input_path, triples, output, model_name, dim,
    epochs, batch_size, lr, device, checkpoint_dir, console
):
    """Fine-tune embeddings with contrastive learning."""
    if input_path is None:
        console.print("[bold red]Error:[/bold red] --input is required for finetune mode")
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader

    from .data import ContrastiveDataset, load_chunks_json
    from .export import export_to_rgdb
    from .models import FineTuneModel, PretrainedEmbedder
    from .training import EmbeddingTrainer, LoggingCallback, CheckpointCallback

    # Load data
    console.print(f"[bold]Loading chunks from:[/bold] {input_path}")
    chunks = load_chunks_json(input_path)

    # Generate base embeddings
    console.print(f"[bold]Generating base embeddings with:[/bold] {model_name}")
    base_embedder = PretrainedEmbedder(model_name)
    base_embeddings = base_embedder.encode([c.text for c in chunks], show_progress=True)

    # Create dataset
    dataset = ContrastiveDataset(chunks, precomputed_embeddings=base_embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = FineTuneModel(model_name, output_dim=dim)

    # Setup callbacks
    callbacks = [LoggingCallback()]
    if checkpoint_dir:
        callbacks.append(CheckpointCallback(Path(checkpoint_dir)))

    # Train
    config = TrainingConfig(
        mode="finetune",
        embedding_dim=dim,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        device=device,
    )

    trainer = EmbeddingTrainer(model, config, callbacks=callbacks)
    console.print("[bold]Starting training...[/bold]")
    trainer.train(loader)

    # Generate final embeddings
    console.print("[bold]Generating final embeddings...[/bold]")
    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode([c.text for c in chunks])
        final_embeddings = final_embeddings.cpu().numpy()

    # Export
    export_to_rgdb(final_embeddings, output)
    console.print(f"[bold green]Success![/bold green] Saved to {output}")


def _train_rotate(
    triples_path, output, dim, epochs, batch_size, lr,
    device, checkpoint_dir, console
):
    """Train RotatE embeddings."""
    if triples_path is None:
        console.print("[bold red]Error:[/bold red] --triples is required for rotate mode")
        sys.exit(1)

    from torch.utils.data import DataLoader

    from .data import TripleDataset, TripleLoader
    from .export import export_to_rgdb
    from .models import RotatEForRGDB
    from .training import RotatETrainer, LoggingCallback, CheckpointCallback

    # Load triples
    console.print(f"[bold]Loading triples from:[/bold] {triples_path}")
    loader = TripleLoader()
    triples = loader.load(triples_path)
    console.print(f"Loaded {len(triples)} triples")

    # Build vocabulary
    vocab = loader.build_entity_vocab(triples)
    num_entities = len(vocab)
    console.print(f"Vocabulary size: {num_entities} entities")

    # Map to angle bins
    mapped = loader.map_relations_to_bins(triples, vocab)

    # Show statistics
    bin_stats = loader.get_angle_bin_statistics(mapped)
    console.print("Angle bin distribution:")
    for bin_idx, count in bin_stats.items():
        console.print(f"  Bin {bin_idx}: {count} triples")

    # Create dataset and loader
    dataset = TripleDataset(mapped, num_entities)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = RotatEForRGDB(num_entities, dim)

    # Setup callbacks
    callbacks = [LoggingCallback()]
    if checkpoint_dir:
        callbacks.append(CheckpointCallback(Path(checkpoint_dir)))

    # Train
    config = TrainingConfig(
        mode="rotate",
        embedding_dim=dim,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        device=device,
    )

    trainer = RotatETrainer(model, config, callbacks=callbacks)
    console.print("[bold]Starting training...[/bold]")
    trainer.train(data_loader)

    # Export embeddings
    embeddings = trainer.get_embeddings()
    export_to_rgdb(embeddings, output)

    # Save vocabulary
    vocab_path = Path(output).with_suffix(".vocab.json")
    from .export.rgdb_format import export_vocab
    export_vocab(vocab, vocab_path)

    console.print(f"[bold green]Success![/bold green]")
    console.print(f"  Embeddings: {output}")
    console.print(f"  Vocabulary: {vocab_path}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output .emb file",
)
def export(model_path: str, output: str):
    """
    Export a trained model checkpoint to RGDB format.

    MODEL_PATH: Path to .pt checkpoint file.
    """
    from rich.console import Console

    import torch

    from .export import export_to_rgdb

    console = Console()

    try:
        console.print(f"[bold]Loading checkpoint:[/bold] {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        # Try to find embeddings in checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        embeddings = None
        for key in ["node_embeddings.weight", "embeddings.weight", "embedding.weight"]:
            if key in state_dict:
                embeddings = state_dict[key].numpy()
                console.print(f"Found embeddings at key: {key}")
                break

        if embeddings is None:
            console.print("[bold red]Error:[/bold red] Could not find embeddings in checkpoint")
            console.print("Expected keys: node_embeddings.weight, embeddings.weight, embedding.weight")
            sys.exit(1)

        console.print(f"Embedding shape: {embeddings.shape}")
        export_to_rgdb(embeddings, output)
        console.print(f"[bold green]Success![/bold green] Exported to {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("embeddings_path", type=click.Path(exists=True))
@click.option(
    "--triples", "-t",
    type=click.Path(exists=True),
    help="Test triples for evaluation",
)
@click.option(
    "--vocab", "-v",
    type=click.Path(exists=True),
    help="Vocabulary JSON file",
)
def evaluate(
    embeddings_path: str,
    triples: Optional[str],
    vocab: Optional[str],
):
    """
    Evaluate embedding quality.

    EMBEDDINGS_PATH: Path to .emb file.
    """
    from rich.console import Console
    from rich.table import Table

    from .export import load_from_rgdb, validate_embeddings

    console = Console()

    try:
        console.print(f"[bold]Loading embeddings:[/bold] {embeddings_path}")
        embeddings = load_from_rgdb(embeddings_path)
        console.print(f"Shape: {embeddings.shape}")

        # Validate
        results = validate_embeddings(embeddings)

        # Display results
        table = Table(title="Embedding Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in results["stats"].items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))

        console.print(table)

        if results["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                console.print(f"  - {error}")

        if results["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results["warnings"]:
                console.print(f"  - {warning}")

        # Evaluate on triples if provided
        if triples:
            console.print(f"\n[bold]Evaluating on triples:[/bold] {triples}")
            # TODO: Implement triple evaluation
            console.print("[yellow]Triple evaluation not yet implemented[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str):
    """
    Show information about an embedding file.

    PATH: Path to .emb file.
    """
    from rich.console import Console
    from rich.table import Table

    from .export.rgdb_format import get_file_info

    console = Console()

    try:
        info = get_file_info(path)

        table = Table(title="File Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in info.items():
            table.add_row(key, str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
