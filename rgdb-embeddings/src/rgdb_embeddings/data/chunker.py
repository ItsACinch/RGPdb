"""Text chunking strategies for document processing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class Chunk:
    """A chunk of text from a document."""

    text: str
    chunk_id: int
    document_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Estimate token count (approximate)."""
        return len(self.text.split())


class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            min_chunk_size: Minimum chunk size (skip smaller chunks)
            encoding_name: Tiktoken encoding name (cl100k_base for GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        if TIKTOKEN_AVAILABLE:
            self.encoding = tiktoken.get_encoding(encoding_name)
        else:
            self.encoding = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback: approximate with word count
        return len(text.split())

    def _encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        if self.encoding:
            return self.encoding.encode(text)
        # Fallback: use characters (less accurate)
        return list(range(len(text.split())))

    def _decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        if self.encoding:
            return self.encoding.decode(tokens)
        # Fallback not possible without original text
        raise ValueError("Cannot decode without tiktoken")

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc",
        metadata: Optional[Dict[str, str]] = None,
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk
            document_id: ID of the source document
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        if self.encoding:
            # Token-based chunking (more accurate)
            chunks = self._chunk_by_tokens(text, document_id, metadata)
        else:
            # Word-based chunking (fallback)
            chunks = self._chunk_by_words(text, document_id, metadata)

        # Filter out chunks that are too small
        chunks = [c for c in chunks if len(c.text.split()) >= self.min_chunk_size]

        return chunks

    def _chunk_by_tokens(
        self, text: str, document_id: str, metadata: Dict[str, str]
    ) -> List[Chunk]:
        """Chunk text using token-based splitting."""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Find character positions (approximate)
            # This is a simplification; exact positions would require more work
            char_start = len(self.encoding.decode(tokens[:start])) if start > 0 else 0
            char_end = char_start + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    start_char=char_start,
                    end_char=char_end,
                    metadata=metadata.copy(),
                )
            )

            chunk_id += 1
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start >= end:
                break

        return chunks

    def _chunk_by_words(
        self, text: str, document_id: str, metadata: Dict[str, str]
    ) -> List[Chunk]:
        """Chunk text using word-based splitting (fallback)."""
        words = text.split()
        chunks = []

        start = 0
        chunk_id = 0
        char_pos = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                    metadata=metadata.copy(),
                )
            )

            char_pos += len(chunk_text) + 1  # +1 for space
            chunk_id += 1
            start = end - self.chunk_overlap

            if start >= end:
                break

        return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 512,
    document_id: str = "doc",
    metadata: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    """
    Chunk text by sentences, grouping sentences until max size is reached.

    This preserves sentence boundaries, which can be better for semantic coherence.
    """
    import re

    metadata = metadata or {}

    # Simple sentence splitting (could be improved with nltk/spacy)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_size = 0
    char_pos = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())

        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Start a new chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                    metadata=metadata.copy(),
                )
            )
            char_pos += len(chunk_text) + 1
            chunk_id += 1
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                metadata=metadata.copy(),
            )
        )

    return chunks
