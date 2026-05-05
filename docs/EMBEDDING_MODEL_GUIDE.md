# Embedding Model Guide for RGDB

This guide explains how to create embedding models that work well with RGDB's hybrid graph+vector architecture.

## Embedding Design Principles for RGDB

### What Makes RGDB Different

| Aspect | Traditional VectorDB | RGDB |
|--------|---------------------|------|
| Similarity | Embedding distance only | Graph propagation + embedding |
| Relationships | Implicit in vectors | Explicit (16 angle bins) |
| Multi-hop | Requires multiple queries | Native via light propagation |
| Direction | None | IsA, Causes, Contains, etc. |

**Key insight**: Your embeddings should capture *content similarity*, while the graph captures *structural/relational similarity*. Don't try to encode everything in the embedding.

## Option 1: Use Standard Sentence Embeddings (Recommended Start)

For documents/text nodes, use off-the-shelf models:

```python
# Using sentence-transformers (Python)
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, fast
# or: 'all-mpnet-base-v2' (768 dims, better quality)

# Embed your documents/chunks
texts = ["Machine learning is...", "Neural networks are...", ...]
embeddings = model.encode(texts)

# Save for RGDB
np.save("embeddings.npy", embeddings.astype(np.float32))
```

**Why this works**: The graph handles relationship types (IsA, Causes), so embeddings just need to capture "is this content similar?" The hybrid scoring blends both signals.

## Option 2: Relation-Aware Embeddings

If you want embeddings that understand relationship types, consider knowledge graph embedding approaches:

### RotatE-style Directional Embeddings

```python
import torch
import torch.nn as nn
import math

class DirectionalEmbedding(nn.Module):
    """
    Embeddings that can be 'rotated' to different relationship directions.
    Inspired by RotatE but adapted for RGDB's 16 angle bins.
    """
    def __init__(self, num_nodes, dim, num_angles=16):
        super().__init__()
        self.dim = dim
        self.num_angles = num_angles

        # Base embedding for each node
        self.node_embeddings = nn.Embedding(num_nodes, dim)

        # Rotation matrices for each angle bin (relationship type)
        # Each angle bin rotates embeddings in a learned subspace
        self.angle_rotations = nn.Parameter(
            torch.randn(num_angles, dim // 2) * 0.1
        )

    def forward(self, node_ids, angle_bin=None):
        """Get embedding, optionally rotated to a specific angle."""
        emb = self.node_embeddings(node_ids)

        if angle_bin is not None:
            # Apply rotation for this relationship direction
            emb = self._rotate(emb, angle_bin)

        return emb

    def _rotate(self, emb, angle_bin):
        """Rotate embedding by angle (complex rotation in pairs)."""
        # Split into real/imaginary pairs
        re, im = emb[..., :self.dim//2], emb[..., self.dim//2:]

        # Get rotation angle for this bin
        theta = self.angle_rotations[angle_bin]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Complex rotation: (re + i*im) * (cos + i*sin)
        new_re = re * cos_t - im * sin_t
        new_im = re * sin_t + im * cos_t

        return torch.cat([new_re, new_im], dim=-1)

    def similarity(self, emb1, emb2):
        """Cosine similarity."""
        return torch.cosine_similarity(emb1, emb2, dim=-1)
```

**Training objective**: Given (source, relation, target) triples, the rotated source should be close to target:

```python
def relation_loss(model, source_ids, target_ids, angle_bins):
    """Contrastive loss for relation-aware embeddings."""
    source_emb = model(source_ids, angle_bins)  # Rotated by relation
    target_emb = model(target_ids)              # Base embedding

    # Positive pairs should be similar
    pos_sim = model.similarity(source_emb, target_emb)

    # Negative sampling (random targets)
    neg_ids = torch.randint(0, model.node_embeddings.num_embeddings, target_ids.shape)
    neg_emb = model(neg_ids)
    neg_sim = model.similarity(source_emb, neg_emb)

    # Contrastive loss
    loss = -torch.log(torch.sigmoid(pos_sim - neg_sim)).mean()
    return loss
```

## Option 3: Graph-Aware Embeddings (GNN)

Train embeddings that incorporate graph structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RGDBGraphEncoder(nn.Module):
    """
    Graph neural network that respects RGDB's angle bins.
    Each message pass is weighted by relationship type compatibility.
    """
    def __init__(self, input_dim, hidden_dim, num_angles=16):
        super().__init__()
        self.num_angles = num_angles

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Per-angle-bin message transforms
        self.angle_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_angles)
        ])

        # Aggregation
        self.aggregate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index, edge_angle_bins):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: [2, num_edges] source/target indices
        edge_angle_bins: [num_edges] angle bin for each edge
        """
        h = self.input_proj(x)

        # Message passing with angle-aware transforms
        src, dst = edge_index
        messages = []

        for angle in range(self.num_angles):
            # Get edges with this angle
            mask = edge_angle_bins == angle
            if mask.sum() == 0:
                continue

            # Transform source embeddings for this relationship type
            src_emb = self.angle_transforms[angle](h[src[mask]])

            # Aggregate at destinations
            for i, d in enumerate(dst[mask]):
                messages.append((d.item(), src_emb[i]))

        # Aggregate messages per node
        agg = torch.zeros_like(h)
        counts = torch.zeros(h.size(0), 1, device=h.device)
        for dst_idx, msg in messages:
            agg[dst_idx] += msg
            counts[dst_idx] += 1
        agg = agg / (counts + 1e-8)

        # Combine with self
        h = self.aggregate(torch.cat([h, agg], dim=-1))
        return F.normalize(h, dim=-1)
```

## Option 4: Hybrid Content + Structure

Combine text embeddings with structural embeddings:

```python
class HybridEmbedding:
    """Combine content embeddings with graph-learned embeddings."""

    def __init__(self, text_model, graph_dim=64):
        self.text_model = text_model  # e.g., SentenceTransformer
        self.graph_embeddings = None  # Learned from graph
        self.alpha = 0.7  # Weight for text vs graph

    def encode(self, texts, node_ids=None):
        # Get text embeddings
        text_emb = self.text_model.encode(texts)
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

        if node_ids is not None and self.graph_embeddings is not None:
            # Get graph-learned embeddings
            graph_emb = self.graph_embeddings[node_ids]
            graph_emb = graph_emb / np.linalg.norm(graph_emb, axis=1, keepdims=True)

            # Concatenate (or weighted average)
            combined = np.concatenate([
                text_emb * self.alpha,
                graph_emb * (1 - self.alpha)
            ], axis=1)
            return combined

        return text_emb
```

## Practical Recommendations

### For Getting Started
```python
# Simple and effective
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(your_texts)
```

### For Production Quality
```python
# Better quality, still fast
model = SentenceTransformer('all-mpnet-base-v2')

# Or domain-specific if you have one
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')  # Medical
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')  # Legal
```

### For Relation-Aware (Advanced)
1. Train RotatE-style embeddings on your (source, relation, target) triples
2. Use TransE/ComplEx if you have a knowledge graph
3. Fine-tune sentence transformers with relationship labels

## RGDB Integration

Once you have embeddings, save and load them:

```rust
// Rust side - loading embeddings
use rgdb::EmbeddingStore;
use ndarray::Array2;

// Load from numpy file (via intermediate format)
let embeddings = load_numpy_f32("embeddings.npy")?;
let store = EmbeddingStore::new(embeddings);

// Or load from RGDB format
let store = EmbeddingStore::load("embeddings.emb")?;
```

```python
# Python side - saving for RGDB
import struct

def save_for_rgdb(embeddings, path):
    """Save embeddings in RGDB's binary format."""
    num_nodes, dim = embeddings.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', num_nodes, dim))
        f.write(embeddings.astype('<f4').tobytes())

save_for_rgdb(embeddings, "embeddings.emb")
```

## Key Insight: Let RGDB Do the Heavy Lifting

The beauty of RGDB is that the graph already encodes:
- **Relationship types** (angle bins)
- **Multi-hop paths** (light propagation)
- **Structural similarity** (graph connectivity)

So your embeddings should focus on:
- **Content similarity** (what is this node about?)
- **Semantic matching** (does query match node content?)

The hybrid scoring (`α×graph + β×vector`) then combines both signals. Start simple (sentence-transformers), and only add complexity if needed.
