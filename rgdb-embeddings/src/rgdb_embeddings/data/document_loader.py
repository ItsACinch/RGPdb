"""Document loading for various file formats."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .chunker import Chunk, TextChunker


@dataclass
class Document:
    """A loaded document with metadata."""

    content: str
    document_id: str
    source_path: Path
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".rst", ".html"}

    def __init__(self):
        """Initialize the document loader."""
        self._pdf_available = self._check_pdf_support()

    def _check_pdf_support(self) -> bool:
        """Check if PDF support is available."""
        try:
            import PyPDF2  # noqa: F401

            return True
        except ImportError:
            return False

    def load_file(self, path: Union[str, Path]) -> Document:
        """
        Load a single document file.

        Args:
            path: Path to the document file

        Returns:
            Document object with content and metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()

        if extension == ".pdf":
            content = self._load_pdf(path)
        elif extension in {".txt", ".md", ".markdown", ".rst"}:
            content = self._load_text(path)
        elif extension == ".html":
            content = self._load_html(path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # Generate document ID from content hash
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        return Document(
            content=content,
            document_id=doc_id,
            source_path=path,
            metadata={
                "filename": path.name,
                "extension": extension,
                "size_bytes": str(path.stat().st_size),
            },
        )

    def load_directory(
        self, path: Union[str, Path], recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            path: Path to the directory
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents

    def load_and_chunk(
        self,
        path: Union[str, Path],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        recursive: bool = True,
    ) -> List[Chunk]:
        """
        Load documents and chunk them in one step.

        Args:
            path: Path to file or directory
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            recursive: Search subdirectories if path is a directory

        Returns:
            List of Chunk objects
        """
        path = Path(path)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if path.is_file():
            documents = [self.load_file(path)]
        else:
            documents = self.load_directory(path, recursive=recursive)

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_text(
                text=doc.content,
                document_id=doc.document_id,
                metadata=doc.metadata,
            )
            all_chunks.extend(chunks)

        return all_chunks

    def _load_text(self, path: Path) -> str:
        """Load plain text file."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode {path} with any supported encoding")

    def _load_pdf(self, path: Path) -> str:
        """Load PDF file."""
        if not self._pdf_available:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")

        import PyPDF2

        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def _load_html(self, path: Path) -> str:
        """Load HTML file and extract text."""
        import html
        import re

        content = self._load_text(path)

        # Remove script and style elements
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Decode HTML entities
        content = html.unescape(content)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content


def save_chunks_json(chunks: List[Chunk], output_path: Union[str, Path]) -> None:
    """Save chunks to JSON file."""
    import json

    data = [
        {
            "text": c.text,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chunks_json(input_path: Union[str, Path]) -> List[Chunk]:
    """Load chunks from JSON file."""
    import json

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Chunk(
            text=item["text"],
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            start_char=item["start_char"],
            end_char=item["end_char"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]
