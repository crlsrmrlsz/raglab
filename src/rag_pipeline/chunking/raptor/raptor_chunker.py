"""RAPTOR strategy interface for the chunking pipeline.

## RAG Theory: RAPTOR as Post-Processing Strategy

Like contextual chunking, RAPTOR is a post-processing step on semantic chunks (std=2):
1. Semantic chunking creates the leaf nodes (topic-boundary splits)
2. RAPTOR builds a hierarchical tree on top
3. All nodes (leaves + summaries) go to Stage 5 for embedding

This approach reuses existing chunk structure while adding multi-level
abstraction for improved thematic retrieval.

## Library Usage

- raptor.tree_builder: Core tree construction
- raptor.schemas: RaptorNode, TreeMetadata dataclasses
- src.shared.files: File I/O utilities

## Data Flow

1. Load: DIR_FINAL_CHUNKS/semantic_std2/{book}.json
2. Process: build_raptor_tree() for each book
3. Save: DIR_FINAL_CHUNKS/raptor/{book}.json
"""

import json
from pathlib import Path
from typing import Optional

from src.config import (
    DIR_FINAL_CHUNKS,
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_CLUSTER_SIZE,
    RAPTOR_SUMMARY_MODEL,
)
from src.shared.files import setup_logging, OverwriteContext, OverwriteMode
from src.rag_pipeline.chunking.raptor.tree_builder import build_raptor_tree
from src.rag_pipeline.chunking.raptor.schemas import RaptorNode, TreeMetadata

logger = setup_logging(__name__)

# Output folder name
RAPTOR_FOLDER = "raptor"


def run_raptor_chunking(
    max_levels: int = RAPTOR_MAX_LEVELS,
    min_cluster_size: int = RAPTOR_MIN_CLUSTER_SIZE,
    summary_model: str = RAPTOR_SUMMARY_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, TreeMetadata]:
    """Build RAPTOR trees for all semantic chunks (std=2).

    Main entry point for RAPTOR strategy. Loads semantic_std2 chunks and builds
    hierarchical summary trees for each book.

    Note: This is a POST-PROCESSING step on semantic chunks (std=2). Run semantic
    chunking first if the semantic_std2/ folder is empty.

    Args:
        max_levels: Maximum tree depth (default: 4).
        min_cluster_size: Minimum nodes for clustering (default: 3).
        summary_model: OpenRouter model ID for summarization.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to TreeMetadata with full tree statistics.

    Raises:
        FileNotFoundError: If semantic_std2 chunks don't exist.
        Exception: Re-raises any error from processing (fail-fast).
    """
    # Input: semantic chunks (std=2)
    input_dir = Path(DIR_FINAL_CHUNKS) / "semantic_std2"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Semantic chunks not found at {input_dir}. "
            "Run semantic chunking first: python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0"
        )

    input_files = list(input_dir.glob("*.json"))

    if not input_files:
        raise FileNotFoundError(
            f"No chunk files found in {input_dir}. "
            "Run semantic chunking first."
        )

    # Output directory
    output_dir = Path(DIR_FINAL_CHUNKS) / RAPTOR_FOLDER

    if overwrite_context is None:
        overwrite_context = OverwriteContext(OverwriteMode.ALL)

    results = {}
    skipped_count = 0

    logger.info("Starting RAPTOR tree building...")
    logger.info(f"Processing {len(input_files)} books from semantic_std2/")
    logger.info(f"Output folder: {RAPTOR_FOLDER}/")
    logger.info(f"Max levels: {max_levels}, min cluster size: {min_cluster_size}")
    logger.info(f"Summary model: {summary_model}")

    for file_path in sorted(input_files):
        book_name = file_path.stem

        # Output path
        output_path = output_dir / f"{book_name}.json"

        # Check if we should process
        if not overwrite_context.should_overwrite(output_path, logger):
            skipped_count += 1
            continue

        try:
            metadata = _process_single_book(
                file_path,
                output_path,
                max_levels=max_levels,
                min_cluster_size=min_cluster_size,
                summary_model=summary_model,
            )
            results[book_name] = metadata
            logger.info(
                f"  {book_name}: {metadata.total_nodes} nodes "
                f"({metadata.leaf_count} leaves, {metadata.summary_count} summaries), "
                f"{metadata.max_level} levels"
            )

        except Exception as e:
            logger.error(f"Failed processing {book_name}: {e}")
            raise

    total_nodes = sum(m.total_nodes for m in results.values())
    logger.info(f"RAPTOR tree building complete: {total_nodes} total nodes")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files (already exist)")

    return results


def _process_single_book(
    input_path: Path,
    output_path: Path,
    max_levels: int,
    min_cluster_size: int,
    summary_model: str,
) -> TreeMetadata:
    """Process a single book's semantic chunks (std=2) into a RAPTOR tree.

    Args:
        input_path: Path to semantic_std2 chunk JSON.
        output_path: Path for output RAPTOR JSON.
        max_levels: Maximum tree depth.
        min_cluster_size: Minimum nodes for clustering.
        summary_model: Model for summarization.

    Returns:
        TreeMetadata with full tree statistics.
    """
    book_id = input_path.stem

    # Load semantic chunks
    with input_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Processing {book_id} ({len(chunks)} semantic chunks)")

    # Build tree
    all_nodes, metadata = build_raptor_tree(
        chunks=chunks,
        book_id=book_id,
        max_levels=max_levels,
        min_cluster_size=min_cluster_size,
        summary_model=summary_model,
    )

    # Convert to dicts for JSON serialization
    chunks_output = [node.to_dict() for node in all_nodes]

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "book_id": book_id,
                "tree_metadata": metadata.to_dict(),
                "chunks": chunks_output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return metadata


if __name__ == "__main__":
    # Standalone execution for testing
    logger.info("Starting RAPTOR tree building (standalone)...")
    stats = run_raptor_chunking()
    logger.info(f"Completed! Processed {len(stats)} books")
    total = sum(m.total_nodes for m in stats.values())
    logger.info(f"Total nodes: {total}")
