"""Shared utilities for GraphRAG entity extraction.

This module provides helper functions for loading chunks and saving
extraction results. The actual extraction logic is in extraction.py,
which uses curated entity types from graphrag_types.yaml.
"""

from typing import Any, Optional
from pathlib import Path
import json

from src.config import DIR_FINAL_CHUNKS, DIR_GRAPH_DATA
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def load_chunks_for_extraction(
    strategy: str = "section",
    book_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Load chunks from Stage 4 for entity extraction.

    Reads chunk files from DIR_FINAL_CHUNKS/{strategy}/ directory.
    Optionally filters by book IDs.

    Args:
        strategy: Chunking strategy subfolder (default: "section").
        book_ids: Optional list of book IDs to include (None = all).

    Returns:
        List of chunk dicts with text, chunk_id, context, etc.

    Raises:
        FileNotFoundError: If chunk directory doesn't exist.
    """
    chunk_dir = DIR_FINAL_CHUNKS / strategy

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    all_chunks = []
    chunk_files = list(chunk_dir.glob("*.json"))

    for chunk_file in chunk_files:
        # Filter by book ID if specified
        if book_ids:
            book_id = chunk_file.stem  # filename without extension
            if book_id not in book_ids:
                continue

        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list format and dict with "chunks" key
            if isinstance(data, list):
                all_chunks.extend(data)
            elif isinstance(data, dict) and "chunks" in data:
                all_chunks.extend(data["chunks"])
            else:
                logger.warning(f"Unexpected format in {chunk_file}")

    logger.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")
    return all_chunks


def save_extraction_results(
    results: dict[str, Any],
    output_name: str = "extraction_results.json",
) -> Path:
    """Save extraction results to JSON file.

    Saves to DIR_GRAPH_DATA for later Neo4j upload.

    Args:
        results: Dict from extraction (entities, relationships, stats).
        output_name: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = DIR_GRAPH_DATA
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved extraction results to {output_path}")
    return output_path
