"""Export embeddings to RGDB binary format."""

import struct
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


# RGDB embedding file format:
# Header: [num_nodes: u32][dim: u32] (little-endian)
# Data: [embeddings: f32 * num_nodes * dim] (little-endian, row-major)

MAGIC_BYTES = b"RGDB"  # Optional magic bytes for format validation
FORMAT_VERSION = 1


def export_to_rgdb(
    embeddings: np.ndarray,
    output_path: Union[str, Path],
    normalize: bool = True,
    include_header: bool = False,
) -> None:
    """
    Export embeddings to RGDB binary format.

    Format: [num_nodes:u32][dim:u32][embeddings:f32*]
    All values are little-endian.

    Args:
        embeddings: Embeddings array [num_nodes, dim]
        output_path: Path to output .emb file
        normalize: Whether to L2-normalize embeddings before saving
        include_header: Whether to include magic bytes and version (not standard RGDB)

    Raises:
        ValueError: If embeddings have invalid shape
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

    num_nodes, dim = embeddings.shape

    # Validate dimensions
    if num_nodes == 0:
        raise ValueError("Cannot export empty embeddings")
    if dim == 0:
        raise ValueError("Embedding dimension cannot be 0")

    # Convert to float32 and ensure C-contiguous (row-major)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Optional: Write magic bytes and version
        if include_header:
            f.write(MAGIC_BYTES)
            f.write(struct.pack("<I", FORMAT_VERSION))

        # Write dimensions (little-endian unsigned 32-bit integers)
        f.write(struct.pack("<II", num_nodes, dim))

        # Write embeddings (little-endian float32)
        f.write(embeddings.tobytes())


def load_from_rgdb(
    input_path: Union[str, Path],
    check_header: bool = False,
) -> np.ndarray:
    """
    Load embeddings from RGDB binary format.

    Args:
        input_path: Path to .emb file
        check_header: Whether to check for magic bytes

    Returns:
        Embeddings array [num_nodes, dim]

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {input_path}")

    with open(input_path, "rb") as f:
        # Optional: Check magic bytes
        if check_header:
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError(f"Invalid magic bytes: {magic}")
            version = struct.unpack("<I", f.read(4))[0]
            if version > FORMAT_VERSION:
                raise ValueError(f"Unsupported format version: {version}")

        # Read dimensions
        header_data = f.read(8)
        if len(header_data) < 8:
            raise ValueError("File too small to contain valid header")

        num_nodes, dim = struct.unpack("<II", header_data)

        # Validate dimensions
        if num_nodes == 0 or dim == 0:
            raise ValueError(f"Invalid dimensions: {num_nodes}x{dim}")

        # Security check: prevent memory exhaustion from malformed headers
        max_allowed_size = 10 * 1024 * 1024 * 1024  # 10GB limit
        expected_bytes = num_nodes * dim * 4  # 4 bytes per float32
        if expected_bytes > max_allowed_size:
            raise ValueError(
                f"Embedding size exceeds safety limit: {expected_bytes / 1e9:.1f}GB > 10GB. "
                f"Dimensions: {num_nodes}x{dim}"
            )

        # Read embeddings
        data = f.read(expected_bytes)

        if len(data) < expected_bytes:
            raise ValueError(
                f"File too small: expected {expected_bytes} bytes for embeddings, "
                f"got {len(data)}"
            )

        # Convert to numpy array
        embeddings = np.frombuffer(data, dtype="<f4").reshape(num_nodes, dim)

        # Return a copy to avoid issues with buffer ownership
        return embeddings.copy()


def validate_embeddings(embeddings: np.ndarray) -> Dict[str, any]:
    """
    Validate embeddings and return statistics.

    Args:
        embeddings: Embeddings array

    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check shape
    if embeddings.ndim != 2:
        results["valid"] = False
        results["errors"].append(f"Expected 2D array, got {embeddings.ndim}D")
        return results

    num_nodes, dim = embeddings.shape
    results["stats"]["num_nodes"] = num_nodes
    results["stats"]["dim"] = dim

    # Check for NaN/Inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()

    if nan_count > 0:
        results["valid"] = False
        results["errors"].append(f"Contains {nan_count} NaN values")

    if inf_count > 0:
        results["valid"] = False
        results["errors"].append(f"Contains {inf_count} Inf values")

    # Check norms
    norms = np.linalg.norm(embeddings, axis=1)
    results["stats"]["min_norm"] = float(norms.min())
    results["stats"]["max_norm"] = float(norms.max())
    results["stats"]["mean_norm"] = float(norms.mean())

    # Check for zero vectors
    zero_count = np.sum(norms < 1e-10)
    if zero_count > 0:
        results["warnings"].append(f"Contains {zero_count} near-zero vectors")

    # Check if normalized
    is_normalized = np.allclose(norms, 1.0, atol=1e-5)
    results["stats"]["is_normalized"] = is_normalized
    if not is_normalized:
        results["warnings"].append("Embeddings are not L2-normalized")

    # Value statistics
    results["stats"]["min_value"] = float(embeddings.min())
    results["stats"]["max_value"] = float(embeddings.max())
    results["stats"]["mean_value"] = float(embeddings.mean())
    results["stats"]["std_value"] = float(embeddings.std())

    return results


def export_vocab(
    vocab: Dict[str, int],
    output_path: Union[str, Path],
) -> None:
    """
    Export entity vocabulary alongside embeddings.

    Saves as JSON for easy inspection.

    Args:
        vocab: Dictionary mapping entity names to IDs
        output_path: Path to output .vocab.json file
    """
    import json

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocab(input_path: Union[str, Path]) -> Dict[str, int]:
    """
    Load entity vocabulary.

    Args:
        input_path: Path to .vocab.json file

    Returns:
        Dictionary mapping entity names to IDs
    """
    import json

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_info(path: Union[str, Path]) -> Dict[str, any]:
    """
    Get information about an embedding file without loading all data.

    Args:
        path: Path to .emb file

    Returns:
        Dictionary with file information
    """
    path = Path(path)

    info = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return info

    info["size_bytes"] = path.stat().st_size

    with open(path, "rb") as f:
        header_data = f.read(8)
        if len(header_data) >= 8:
            num_nodes, dim = struct.unpack("<II", header_data)
            info["num_nodes"] = num_nodes
            info["dim"] = dim
            info["expected_size"] = 8 + num_nodes * dim * 4
            info["size_matches"] = info["size_bytes"] == info["expected_size"]

    return info
