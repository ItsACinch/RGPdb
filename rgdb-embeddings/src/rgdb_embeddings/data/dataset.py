"""PyTorch datasets for embedding training."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .chunker import Chunk
from .triple_loader import MappedTriple


class TripleDataset(Dataset):
    """Dataset for training on knowledge triples."""

    def __init__(
        self,
        triples: List[MappedTriple],
        num_entities: int,
        negative_samples: int = 10,
        filter_triples: Optional[set] = None,
    ):
        """
        Initialize the triple dataset.

        Args:
            triples: List of mapped triples
            num_entities: Total number of entities in vocabulary
            negative_samples: Number of negative samples per positive
            filter_triples: Set of (head_id, angle_bin, tail_id) tuples to filter
                           (for filtered evaluation, usually train+valid triples)
        """
        self.triples = triples
        self.num_entities = num_entities
        self.negative_samples = negative_samples
        self.filter_triples = filter_triples or set()

        # Build head/tail lookup for faster negative sampling
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup structures for efficient negative sampling."""
        # For each (head, relation) -> set of valid tails
        self.hr_to_tails: Dict[Tuple[int, int], set] = {}
        # For each (tail, relation) -> set of valid heads
        self.tr_to_heads: Dict[Tuple[int, int], set] = {}

        for t in self.triples:
            hr = (t.head_id, t.angle_bin)
            tr = (t.tail_id, t.angle_bin)

            if hr not in self.hr_to_tails:
                self.hr_to_tails[hr] = set()
            self.hr_to_tails[hr].add(t.tail_id)

            if tr not in self.tr_to_heads:
                self.tr_to_heads[tr] = set()
            self.tr_to_heads[tr].add(t.head_id)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary with:
                - head_id: Head entity ID
                - tail_id: Tail entity ID
                - angle_bin: Relationship angle bin
                - negative_tails: IDs of negative tail samples
                - negative_heads: IDs of negative head samples
        """
        triple = self.triples[idx]

        # Sample negative tails (corrupt tail)
        negative_tails = self._sample_negatives(
            triple.head_id, triple.angle_bin, triple.tail_id, corrupt_tail=True
        )

        # Sample negative heads (corrupt head)
        negative_heads = self._sample_negatives(
            triple.head_id, triple.angle_bin, triple.tail_id, corrupt_tail=False
        )

        return {
            "head_id": torch.tensor(triple.head_id, dtype=torch.long),
            "tail_id": torch.tensor(triple.tail_id, dtype=torch.long),
            "angle_bin": torch.tensor(triple.angle_bin, dtype=torch.long),
            "negative_tails": torch.tensor(negative_tails, dtype=torch.long),
            "negative_heads": torch.tensor(negative_heads, dtype=torch.long),
        }

    def _sample_negatives(
        self, head_id: int, angle_bin: int, tail_id: int, corrupt_tail: bool
    ) -> List[int]:
        """Sample negative entities."""
        negatives = []

        if corrupt_tail:
            # Get valid tails to avoid
            valid_tails = self.hr_to_tails.get((head_id, angle_bin), set())
        else:
            # Get valid heads to avoid
            valid_heads = self.tr_to_heads.get((tail_id, angle_bin), set())

        attempts = 0
        max_attempts = self.negative_samples * 10

        while len(negatives) < self.negative_samples and attempts < max_attempts:
            neg = random.randint(0, self.num_entities - 1)
            attempts += 1

            if corrupt_tail:
                if neg not in valid_tails:
                    negatives.append(neg)
            else:
                if neg not in valid_heads:
                    negatives.append(neg)

        # Pad with random if we couldn't find enough
        while len(negatives) < self.negative_samples:
            negatives.append(random.randint(0, self.num_entities - 1))

        return negatives


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning on text chunks."""

    def __init__(
        self,
        chunks: List[Chunk],
        embedder=None,
        precomputed_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize the contrastive dataset.

        Args:
            chunks: List of text chunks
            embedder: Embedder to generate embeddings on-the-fly
            precomputed_embeddings: Optional precomputed embeddings
        """
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = precomputed_embeddings

        # Build document groups for positive pairs
        self._build_document_groups()

    def _build_document_groups(self):
        """Group chunks by document for positive pair sampling."""
        self.doc_to_chunks: Dict[str, List[int]] = {}

        for idx, chunk in enumerate(self.chunks):
            doc_id = chunk.document_id
            if doc_id not in self.doc_to_chunks:
                self.doc_to_chunks[doc_id] = []
            self.doc_to_chunks[doc_id].append(idx)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample for contrastive learning.

        Returns:
            Dictionary with:
                - anchor_idx: Index of anchor chunk
                - positive_idx: Index of positive (same document) chunk
                - negative_idx: Index of negative (different document) chunk
                - anchor_text: Text of anchor chunk (if embedder is used)
                - positive_text: Text of positive chunk
                - negative_text: Text of negative chunk
        """
        anchor_chunk = self.chunks[idx]
        doc_id = anchor_chunk.document_id

        # Sample positive from same document
        same_doc_indices = self.doc_to_chunks[doc_id]
        if len(same_doc_indices) > 1:
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = random.choice(same_doc_indices)
        else:
            pos_idx = idx  # Fallback to self

        # Sample negative from different document
        other_docs = [d for d in self.doc_to_chunks.keys() if d != doc_id]
        if other_docs:
            neg_doc = random.choice(other_docs)
            neg_idx = random.choice(self.doc_to_chunks[neg_doc])
        else:
            # Fallback: use random chunk
            neg_idx = random.randint(0, len(self.chunks) - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(self.chunks) - 1)

        result = {
            "anchor_idx": torch.tensor(idx, dtype=torch.long),
            "positive_idx": torch.tensor(pos_idx, dtype=torch.long),
            "negative_idx": torch.tensor(neg_idx, dtype=torch.long),
        }

        # Include text for on-the-fly embedding
        if self.embedder is not None:
            result["anchor_text"] = self.chunks[idx].text
            result["positive_text"] = self.chunks[pos_idx].text
            result["negative_text"] = self.chunks[neg_idx].text

        # Include precomputed embeddings if available
        if self.embeddings is not None:
            result["anchor_emb"] = torch.tensor(self.embeddings[idx], dtype=torch.float32)
            result["positive_emb"] = torch.tensor(self.embeddings[pos_idx], dtype=torch.float32)
            result["negative_emb"] = torch.tensor(self.embeddings[neg_idx], dtype=torch.float32)

        return result


class TextTripleDataset(Dataset):
    """Dataset combining text and triple information for hybrid training."""

    def __init__(
        self,
        triples: List[MappedTriple],
        entity_texts: Dict[str, str],
        embedder=None,
        negative_samples: int = 5,
    ):
        """
        Initialize the text-triple dataset.

        Args:
            triples: List of mapped triples
            entity_texts: Dictionary mapping entity names to their text descriptions
            embedder: Embedder for generating text embeddings
            negative_samples: Number of negative samples
        """
        self.triples = triples
        self.entity_texts = entity_texts
        self.embedder = embedder
        self.negative_samples = negative_samples

        # Get all entity names
        self.entities = list(entity_texts.keys())

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get a training sample with text and triple information."""
        triple = self.triples[idx]

        head_text = self.entity_texts.get(triple.head, triple.head)
        tail_text = self.entity_texts.get(triple.tail, triple.tail)

        # Sample negative entities
        neg_entities = random.sample(
            [e for e in self.entities if e != triple.head and e != triple.tail],
            min(self.negative_samples, len(self.entities) - 2),
        )
        neg_texts = [self.entity_texts.get(e, e) for e in neg_entities]

        return {
            "head_text": head_text,
            "tail_text": tail_text,
            "head_id": triple.head_id,
            "tail_id": triple.tail_id,
            "angle_bin": triple.angle_bin,
            "negative_texts": neg_texts,
        }
