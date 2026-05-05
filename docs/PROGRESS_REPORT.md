# RGP/RGDB Project Progress Report

**Date:** January 2025
**Branch:** `directional_luminance`

---

## Project Overview

RGDB (Refractive Graph Database) is a novel database combining characteristics of vector databases and graph databases. It uses a physics-inspired "light propagation" model where information flows through the graph like light through a medium, with 16 "angle bins" representing different semantic relationship types.

**Primary Use Cases:**
- Enterprise search with hybrid retrieval + reasoning
- Personalized activity feeds
- Knowledge exploration with multi-hop reasoning

---

## Completed Work

### Phase 1: Core RAG Module (Rust)

Created a complete RAG (Retrieval-Augmented Generation) module in `src/rag/`:

| File | Purpose |
|------|---------|
| `src/rag/mod.rs` | Module organization and exports |
| `src/rag/embedding_store.rs` | Vector storage with cosine similarity, top-K search, save/load |
| `src/rag/intent.rs` | Query intent classification (Definition, Causation, etc.) → angle bins |
| `src/rag/personalization.rs` | UserContext with topic affinities, interaction history, access control |
| `src/rag/query_engine.rs` | RAGQueryEngine combining graph propagation + vector similarity |

**Key Features:**
- Hybrid scoring: `score = α×graph_intensity + β×vector_similarity + γ×personalization_boost`
- Intent classification maps queries like "What is X?" → bin 0 (IsA), "Why does X?" → bin 4 (Causes)
- Multi-source propagation from top-K similar nodes
- Activity feed generation based on user preferences

**Updated:** `src/lib.rs` to export the RAG module.

### Phase 2: Documentation

| File | Purpose |
|------|---------|
| `docs/LLM_INTEGRATION_GUIDE.md` | How to use RGDB with LLMs, tool definitions for OpenAI/Claude |
| `docs/RAG_TEST_PLAN.md` | Comprehensive test plan with unit tests, integration tests, benchmarks |
| `docs/EMBEDDING_MODEL_GUIDE.md` | How to create embedding models for RGDB |
| `CLAUDE.md` | Project overview for AI assistants |

### Phase 3: Python Embedding Training Package

Created a complete pip-installable Python package: `rgdb-embeddings/`

```
rgdb-embeddings/
├── pyproject.toml              # Package config, dependencies
├── README.md                   # Package documentation
├── GET_STARTED.md              # Quick start guide
├── src/rgdb_embeddings/
│   ├── __init__.py             # Public API
│   ├── cli.py                  # CLI: rgdb-embed command
│   ├── config.py               # TrainingConfig, RELATION_TO_BIN mapping
│   ├── data/
│   │   ├── document_loader.py  # PDF, TXT, MD loading
│   │   ├── triple_loader.py    # CSV/JSON knowledge triples
│   │   ├── chunker.py          # Text chunking with tiktoken
│   │   └── dataset.py          # PyTorch datasets
│   ├── models/
│   │   ├── pretrained.py       # SentenceTransformer wrapper
│   │   ├── rotate.py           # RotatE for RGDB's 16 angle bins
│   │   ├── finetune.py         # Contrastive fine-tuning
│   │   └── gnn.py              # Graph Neural Network encoder
│   ├── training/
│   │   ├── trainer.py          # EmbeddingTrainer, RotatETrainer
│   │   ├── losses.py           # TripletLoss, RotatELoss, DirectionalContrastiveLoss
│   │   └── callbacks.py        # Logging, checkpointing, early stopping
│   ├── export/
│   │   └── rgdb_format.py      # Export to .emb binary format
│   └── utils/
│       └── logging.py          # Logging utilities
├── tests/
│   ├── test_config.py          # Config tests
│   ├── test_loaders.py         # Data loader tests
│   ├── test_export.py          # Export format tests
│   └── test_models.py          # Model tests
└── examples/
    ├── documents_only.py       # Pretrained embeddings from docs
    ├── triples_only.py         # RotatE training on knowledge graph
    └── hybrid_training.py      # Combined approach
```

**CLI Commands:**
```bash
rgdb-embed process ./docs/ --output chunks.json
rgdb-embed train --mode pretrained --input chunks.json --output embeddings.emb
rgdb-embed train --mode rotate --triples knowledge.csv --output embeddings.emb
rgdb-embed evaluate embeddings.emb
rgdb-embed info embeddings.emb
```

---

## Key Technical Decisions

### 1. RGDB Angle Bin Mapping

16 angle bins represent semantic relationship types:

| Bin | Relation | Examples |
|-----|----------|----------|
| 0 | IsA | is_a, type_of, instance_of |
| 1 | Requires | requires, depends_on, needs |
| 2 | RelatedTo | related_to, associated_with |
| 4 | Causes | causes, leads_to, results_in |
| 6 | Contains | contains, has, includes |
| 8 | PartOf | part_of, belongs_to |
| 10 | SimilarTo | similar_to, like, resembles |
| 12 | OppositeOf | opposite_of, contrasts_with |
| 14 | Enables | enables, allows, supports |
| 15 | ConflictsWith | conflicts_with, incompatible |

### 2. Embedding File Format

Binary format for `.emb` files:
```
[num_nodes: u32][dim: u32][embeddings: f32 * num_nodes * dim]
```
All values little-endian, row-major order.

### 3. RotatE Adaptation

Adapted RotatE (knowledge graph embedding) for RGDB:
- Each angle bin has learned rotation phases
- Embeddings can be "rotated" to different relationship directions
- Training: rotated head should be close to tail

### 4. Hybrid Scoring

Query results combine three signals:
```
score = α × graph_intensity + β × vector_similarity + γ × personalization_boost
```
Default weights: α=0.5, β=0.4, γ=0.1

---

## What Remains To Do

### High Priority

1. **Test the Python package**
   ```bash
   cd rgdb-embeddings
   pip install -e .
   pytest
   ```

2. **Integration test**: Verify Python-generated `.emb` files load correctly in Rust

3. **Run Rust tests**
   ```bash
   cargo test
   ```

### Medium Priority

4. **Implement GNN training mode** in Python package (currently placeholder)

5. **API server** (Phase 4 from plan): REST endpoints using Axum
   - `POST /api/v1/search`
   - `POST /api/v1/ingest`
   - `GET /api/v1/explore/:node_id`
   - `POST /api/v1/feed`

6. **Ingestion pipeline** (Phase 2 from plan):
   - `src/rag/ingestion.rs` - Document chunking, node creation
   - `src/rag/entity.rs` - Entity extraction (NER)
   - `src/rag/relation.rs` - Relationship extraction

### Lower Priority

7. **CUDA acceleration** for RAG queries
8. **Reasoning path tracking** (currently returns empty paths)
9. **Incremental graph updates**
10. **Monitoring and metrics**

---

## File Locations Summary

### Rust (Core RGDB)
- Main library: `src/lib.rs`
- RAG module: `src/rag/` (new)
- Graph core: `src/graph.rs`
- Light propagation: `src/propagation.rs`
- PVS optimization: `src/pvs.rs`

### Python (Embedding Training)
- Package root: `rgdb-embeddings/`
- CLI entry point: `rgdb-embeddings/src/rgdb_embeddings/cli.py`
- Install: `cd rgdb-embeddings && pip install -e .`

### Documentation
- LLM integration: `docs/LLM_INTEGRATION_GUIDE.md`
- Test plan: `docs/RAG_TEST_PLAN.md`
- Embedding guide: `docs/EMBEDDING_MODEL_GUIDE.md`
- Project overview: `CLAUDE.md`

### Plan File
- Full architecture plan: `C:\Users\alpha\.claude\plans\misty-hugging-kay.md`

---

## Environment Notes

- **CUDA Toolkit**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
- **Platform**: Windows
- **Git branch**: `directional_luminance`

---

## Quick Resume Commands

```bash
# Check current state
cd D:\repos\rgp
git status
git log --oneline -5

# Test Rust code
cargo test

# Test Python package
cd rgdb-embeddings
pip install -e .
pytest

# Run Python example
python examples/triples_only.py
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     API LAYER (Axum) [TODO]                 │
│   /search    /ingest    /explore    /feed    /reason        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY ENGINE [DONE]                      │
│  Intent Classification → Source Selection → Hybrid Scoring  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ EMBEDDING    │    │ GRAPH ENGINE │    │ PERSONALIZATION  │
│ STORE [DONE] │    │ (CSR + PVS)  │    │ [DONE]           │
│ (.emb file)  │    │ (.rgdb file) │    │                  │
└──────────────┘    └──────────────┘    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PYTHON EMBEDDING PIPELINE [DONE]               │
│    rgdb-embed: process → train → export                     │
└─────────────────────────────────────────────────────────────┘
```
