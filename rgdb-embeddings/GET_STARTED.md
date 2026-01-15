# Getting Started with RGDB Embeddings

This guide covers how to install the package, create input data, and generate embeddings for RGDB.

## Installation

```bash
cd rgdb-embeddings
pip install -e .
```

For GPU support, ensure you have PyTorch with CUDA installed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Option A: Generate embeddings from documents

```bash
# Process documents into chunks
rgdb-embed process ./my_docs/ --output chunks.json

# Generate embeddings with pretrained model
rgdb-embed train --mode pretrained --input chunks.json --output embeddings.emb
```

### Option B: Train on knowledge triples

```bash
# Train RotatE embeddings
rgdb-embed train --mode rotate --triples knowledge.csv --output embeddings.emb --epochs 20
```

---

## Creating Knowledge Triples Files

The package supports two formats for knowledge triples:

### CSV Format (Recommended)

Create a file with three columns: `head`, `relation`, `tail`

**Example: `knowledge.csv`**
```csv
head,relation,tail
machine_learning,is_a,artificial_intelligence
deep_learning,is_a,machine_learning
neural_network,requires,training_data
cnn,part_of,deep_learning
pytorch,similar_to,tensorflow
overfitting,causes,poor_generalization
regularization,enables,generalization
dropout,is_a,regularization
gpu,enables,fast_training
```

### JSON Format

**Example: `knowledge.json`**
```json
[
  {"head": "machine_learning", "relation": "is_a", "tail": "artificial_intelligence"},
  {"head": "deep_learning", "relation": "is_a", "tail": "machine_learning"},
  {"head": "neural_network", "relation": "requires", "tail": "training_data"},
  {"head": "cnn", "relation": "part_of", "tail": "deep_learning"}
]
```

---

## Supported Relations → RGDB Angle Bins

Use these relation names to map to RGDB's 16 angle bins:

| Relation | Angle Bin | Meaning |
|----------|-----------|---------|
| `is_a`, `type_of`, `instance_of` | 0 | Type hierarchy |
| `requires`, `depends_on`, `needs` | 1 | Dependencies |
| `related_to`, `associated_with` | 2 | General association |
| `causes`, `leads_to`, `results_in` | 4 | Causation |
| `contains`, `has`, `includes` | 6 | Containment |
| `part_of`, `belongs_to`, `member_of` | 8 | Part-whole |
| `similar_to`, `like`, `resembles` | 10 | Similarity |
| `opposite_of`, `contrasts_with` | 12 | Opposition |
| `enables`, `allows`, `supports` | 14 | Enablement |
| `conflicts_with`, `incompatible` | 15 | Conflict |

**Note:** Unknown relations default to bin 2 (`related_to`).

---

## CLI Commands Reference

### Process Documents

Chunk documents for embedding generation:

```bash
rgdb-embed process <input_dir> --output <chunks.json> [options]

Options:
  --chunk-size    Target chunk size in tokens (default: 512)
  --overlap       Overlap between chunks (default: 50)
  --recursive     Search subdirectories (default: true)
```

### Train Embeddings

```bash
rgdb-embed train --mode <mode> [options]

Modes:
  pretrained   Use pretrained model directly (fastest)
  finetune     Fine-tune on your data with contrastive learning
  rotate       Train RotatE embeddings from knowledge triples
  gnn          Train with graph neural network (coming soon)

Options:
  --input, -i       Input chunks JSON file
  --triples, -t     Knowledge triples CSV/JSON file
  --output, -o      Output .emb file (required)
  --model-name      Pretrained model (default: all-MiniLM-L6-v2)
  --dim             Embedding dimension (default: 384)
  --epochs          Training epochs (default: 10)
  --batch-size      Batch size (default: 32)
  --lr              Learning rate (default: 1e-4)
  --device          Device: cuda or cpu (default: cuda)
```

### Evaluate Embeddings

```bash
rgdb-embed evaluate <embeddings.emb> [options]

Options:
  --triples, -t    Test triples for evaluation
  --vocab, -v      Vocabulary JSON file
```

### Get File Info

```bash
rgdb-embed info <embeddings.emb>
```

---

## Python API Examples

### Generate embeddings from documents

```python
from rgdb_embeddings import DocumentLoader, PretrainedEmbedder, export_to_rgdb

# Load and chunk documents
loader = DocumentLoader()
chunks = loader.load_and_chunk("./docs/", chunk_size=512)

# Generate embeddings
embedder = PretrainedEmbedder("all-MiniLM-L6-v2")
embeddings = embedder.encode([c.text for c in chunks])

# Export for RGDB
export_to_rgdb(embeddings, "embeddings.emb")
```

### Train RotatE on knowledge triples

```python
import torch
from torch.utils.data import DataLoader
from rgdb_embeddings import (
    TripleLoader, TripleDataset, RotatEForRGDB,
    RotatETrainer, TrainingConfig, export_to_rgdb
)

# Load triples
loader = TripleLoader()
triples = loader.load("knowledge.csv")
vocab = loader.build_entity_vocab(triples)
mapped = loader.map_relations_to_bins(triples, vocab)

# Create dataset
dataset = TripleDataset(mapped, num_entities=len(vocab))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create and train model
model = RotatEForRGDB(num_nodes=len(vocab), dim=256)
config = TrainingConfig(mode="rotate", epochs=20, device="cuda")
trainer = RotatETrainer(model, config)
trainer.train(data_loader)

# Export
embeddings = trainer.get_embeddings()
export_to_rgdb(embeddings, "embeddings.emb")
```

---

## Tips for Good Results

### Entity Names
- Use lowercase with underscores: `machine_learning` not `Machine Learning`
- Be consistent: don't mix `ML` and `machine_learning` for the same concept

### Relations
- Case-insensitive: `Is_A`, `is_a`, `IS_A` all work
- Spaces/hyphens auto-converted: `part of` → `part_of`

### Data Size
- **Minimum**: 100+ triples for meaningful training
- **Recommended**: 1,000+ triples for good quality
- **Production**: 10,000+ triples for best results

### Training
- Start with `--epochs 10` and increase if loss is still decreasing
- Use `--device cuda` for faster training (10-100x speedup)
- Monitor loss: it should decrease over epochs

---

## Output Format

The `.emb` file uses RGDB's binary format:

```
[num_nodes: u32][dim: u32][embeddings: f32 * num_nodes * dim]
```

All values are little-endian. Load in Rust with:

```rust
let store = EmbeddingStore::load("embeddings.emb")?;
```

Or in Python:

```python
from rgdb_embeddings import load_from_rgdb
embeddings = load_from_rgdb("embeddings.emb")
```
