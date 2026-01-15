# RGDB Embeddings

Train embeddings for [RGDB](https://github.com/rgdb/rgdb) - the Refractive Graph Database.

## Features

- **Multiple Training Modes**: Pretrained, fine-tuning, RotatE, and GNN-based embeddings
- **Document Support**: Load and chunk PDF, TXT, and Markdown files
- **Knowledge Triples**: Train relation-aware embeddings from CSV/JSON triples
- **RGDB Integration**: Export directly to RGDB's binary `.emb` format
- **CLI Interface**: Easy command-line workflow

## Installation

```bash
pip install rgdb-embeddings
```

For GNN support:
```bash
pip install rgdb-embeddings[gnn]
```

## Quick Start

### Generate embeddings from documents

```bash
# Process documents into chunks
rgdb-embed process ./docs/ --output chunks.json

# Generate embeddings with pretrained model
rgdb-embed train --mode pretrained --input chunks.json --output embeddings.emb
```

### Train relation-aware embeddings from knowledge triples

```bash
# Train RotatE embeddings respecting RGDB's 16 angle bins
rgdb-embed train --mode rotate --triples knowledge.csv --output embeddings.emb --dim 256 --epochs 20
```

### Python API

```python
from rgdb_embeddings import PretrainedEmbedder, DocumentLoader, export_to_rgdb

# Load and chunk documents
loader = DocumentLoader()
chunks = loader.load_and_chunk("./docs/", chunk_size=512)

# Generate embeddings
embedder = PretrainedEmbedder("all-MiniLM-L6-v2")
embeddings = embedder.encode([c.text for c in chunks])

# Export for RGDB
export_to_rgdb(embeddings, "embeddings.emb")
```

## RGDB Angle Bins

RGDB uses 16 angle bins to represent different relationship types:

| Bin | Relationship | Examples |
|-----|--------------|----------|
| 0 | IsA | is_a, type_of, instance_of |
| 1 | Requires | requires, depends_on, needs |
| 2 | RelatedTo | related_to, associated_with |
| 4 | Causes | causes, leads_to, results_in |
| 6 | Contains | contains, has, includes |
| 8 | PartOf | part_of, belongs_to, member_of |
| 10 | SimilarTo | similar_to, like, resembles |
| 12 | OppositeOf | opposite_of, contrasts_with |
| 14 | Enables | enables, allows, supports |
| 15 | ConflictsWith | conflicts_with, incompatible |

The RotatE training mode learns rotation matrices for each angle bin, producing embeddings that capture directional semantics.

## Training Modes

### Pretrained (Quickstart)
Use a pretrained sentence-transformers model directly:
```bash
rgdb-embed train --mode pretrained --input chunks.json --output embeddings.emb
```

### Fine-tune
Fine-tune on your data with contrastive learning:
```bash
rgdb-embed train --mode finetune --input chunks.json --triples relations.csv --output embeddings.emb
```

### RotatE (Relation-Aware)
Train RotatE-style embeddings from knowledge triples:
```bash
rgdb-embed train --mode rotate --triples knowledge.csv --output embeddings.emb
```

### GNN (Graph-Aware)
Train with graph neural network respecting angle bins:
```bash
rgdb-embed train --mode gnn --triples knowledge.csv --output embeddings.emb
```

## File Format

### Knowledge Triples (CSV)
```csv
head,relation,tail
machine_learning,is_a,artificial_intelligence
neural_network,requires,training_data
deep_learning,similar_to,machine_learning
```

### Knowledge Triples (JSON)
```json
[
  {"head": "machine_learning", "relation": "is_a", "tail": "artificial_intelligence"},
  {"head": "neural_network", "relation": "requires", "tail": "training_data"}
]
```

### RGDB Embedding Format (.emb)
Binary format: `[num_nodes:u32][dim:u32][embeddings:f32*]` (little-endian)

## CLI Commands

```bash
# Process documents
rgdb-embed process <input_dir> --output <chunks.json> [--chunk-size 512] [--overlap 50]

# Train embeddings
rgdb-embed train --mode <mode> [--input <chunks.json>] [--triples <triples.csv>] --output <embeddings.emb>

# Export model to RGDB format
rgdb-embed export <model.pt> --output <embeddings.emb>

# Evaluate embeddings
rgdb-embed evaluate <embeddings.emb> --triples <test_triples.csv>
```

## License

MIT
