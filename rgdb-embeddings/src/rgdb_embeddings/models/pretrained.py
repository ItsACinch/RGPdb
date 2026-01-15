"""Pretrained sentence embedding models."""

from typing import List, Optional, Union

import numpy as np


class PretrainedEmbedder:
    """
    Wrapper around sentence-transformers for direct embedding generation.

    This is the simplest approach: use a pretrained model to generate
    embeddings without any training.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize the pretrained embedder.

        Args:
            model_name: Name of the sentence-transformers model.
                        Common options:
                        - "all-MiniLM-L6-v2" (384 dims, fast)
                        - "all-mpnet-base-v2" (768 dims, better quality)
                        - "paraphrase-MiniLM-L6-v2" (good for paraphrase detection)
            device: Device to use (cuda/cpu). Auto-detected if None.
            normalize: Whether to L2-normalize embeddings.
        """
        self.model_name = model_name
        self.normalize = normalize

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array [num_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Encode query texts (alias for encode).

        Some models have different prefixes for queries vs documents.
        This method can be overridden for such models.
        """
        return self.encode(queries, **kwargs)

    def encode_documents(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Encode document texts (alias for encode).

        Some models have different prefixes for queries vs documents.
        This method can be overridden for such models.
        """
        return self.encode(documents, **kwargs)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings [n, dim]
            embeddings2: Second set of embeddings [m, dim]

        Returns:
            Similarity matrix [n, m]
        """
        # Normalize if not already normalized
        if not self.normalize:
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        return np.dot(embeddings1, embeddings2.T)

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class InstructEmbedder(PretrainedEmbedder):
    """
    Embedder with instruction prefix support.

    Some newer models (like instructor-xl) use task-specific instructions
    to improve embedding quality.
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        query_instruction: str = "Represent the question for retrieval:",
        document_instruction: str = "Represent the document for retrieval:",
        **kwargs,
    ):
        """
        Initialize with instruction prefixes.

        Args:
            model_name: Instructor model name
            query_instruction: Instruction prefix for queries
            document_instruction: Instruction prefix for documents
        """
        super().__init__(model_name, **kwargs)
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries with query instruction."""
        prefixed = [[self.query_instruction, q] for q in queries]
        return self.model.encode(prefixed, **kwargs)

    def encode_documents(self, documents: List[str], **kwargs) -> np.ndarray:
        """Encode documents with document instruction."""
        prefixed = [[self.document_instruction, d] for d in documents]
        return self.model.encode(prefixed, **kwargs)


# Model recommendations for different use cases
MODEL_RECOMMENDATIONS = {
    "general": {
        "model": "all-MiniLM-L6-v2",
        "description": "Good balance of speed and quality. 384 dimensions.",
    },
    "quality": {
        "model": "all-mpnet-base-v2",
        "description": "Best quality for general text. 768 dimensions.",
    },
    "speed": {
        "model": "paraphrase-MiniLM-L3-v2",
        "description": "Fastest option. 384 dimensions.",
    },
    "multilingual": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "50+ languages supported. 384 dimensions.",
    },
    "code": {
        "model": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        "description": "Optimized for code search. 768 dimensions.",
    },
    "medical": {
        "model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "Fine-tuned on medical literature. 768 dimensions.",
    },
    "legal": {
        "model": "nlpaueb/legal-bert-base-uncased",
        "description": "Fine-tuned on legal documents. 768 dimensions.",
    },
}


def get_recommended_model(domain: str = "general") -> dict:
    """Get model recommendation for a specific domain."""
    return MODEL_RECOMMENDATIONS.get(domain, MODEL_RECOMMENDATIONS["general"])
