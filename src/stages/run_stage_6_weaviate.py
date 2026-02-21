"""
Stage 6: Upload embeddings to Weaviate vector database.

This stage:
- Loads embedded chunks from Stage 5
- Creates a new Weaviate collection (deletes if exists)
- Batch uploads all chunks with their embeddings

Design goals:
- Clean slate: always recreates collection
- Deterministic: same input produces same UUIDs
- Transparent: logs progress per book

Usage:
    python -m src.stages.run_stage_6_weaviate                    # Default: section
    python -m src.stages.run_stage_6_weaviate --strategy semantic  # Semantic collection
"""

import argparse
import json
from pathlib import Path

from src.config import (
    get_collection_name,
    get_embedding_folder_path,
    get_semantic_folder_name,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    DEFAULT_CHUNKING_STRATEGY,
    SEMANTIC_STD_COEFFICIENT,
)

from src.shared.files import setup_logging
from src.rag_pipeline.indexing import (
    get_client,
    create_collection,
    create_raptor_collection,
    delete_collection,
    upload_embeddings,
    get_collection_count,
)
from src.rag_pipeline.chunking.strategies import list_strategies

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def load_embedding_file(file_path: Path) -> list[dict]:
    """
    Load embedded chunks from a JSON file.

    Args:
        file_path: Path to the embedding JSON file.

    Returns:
        List of chunk dictionaries with embeddings.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def upload_book(
    client, collection_name: str, file_path: Path, is_raptor: bool = False
) -> int:
    """
    Upload all chunks from a single book to Weaviate.

    Args:
        client: Connected Weaviate client.
        collection_name: Target collection name.
        file_path: Path to the embedding JSON file.
        is_raptor: If True, include RAPTOR tree properties in upload.

    Returns:
        Number of chunks uploaded.
    """
    logger.info(f"Uploading: {file_path.stem}")

    chunks = load_embedding_file(file_path)

    if not chunks:
        logger.warning(f"  No chunks found in {file_path.name}")
        return 0

    count = upload_embeddings(client, collection_name, chunks, is_raptor=is_raptor)
    logger.info(f"  Uploaded {count} chunks")

    return count


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Upload embeddings to Weaviate vector database."""
    parser = argparse.ArgumentParser(
        description="Stage 6: Upload embeddings to Weaviate"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list_strategies(),
        help=f"Chunking strategy for collection naming (default: {DEFAULT_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--std-coefficient",
        type=float,
        default=None,
        help=(
            f"Std coefficient (for finding correct embedding folder). "
            f"Only used with semantic strategy. (default: {SEMANTIC_STD_COEFFICIENT})"
        ),
    )
    args = parser.parse_args()

    # Determine strategy key for paths (semantic uses coefficient-based naming)
    if args.strategy == "semantic":
        coef = args.std_coefficient if args.std_coefficient is not None else SEMANTIC_STD_COEFFICIENT
        strategy_key = get_semantic_folder_name(coef)
    else:
        strategy_key = args.strategy

    # Generate collection name from strategy key
    collection_name = get_collection_name(strategy_key)

    logger.info(f"Starting Stage 6: Weaviate Upload (strategy: {strategy_key})")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Weaviate: http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")

    # Get embedding files from strategy-scoped folder
    embedding_dir = get_embedding_folder_path(strategy_key)

    if not embedding_dir.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {embedding_dir}. "
            f"Run Stage 5 first: python -m src.stages.run_stage_5_embedding --strategy {args.strategy}"
        )

    files = list(embedding_dir.glob("*.json"))

    if not files:
        logger.warning(f"No embedding files found in {embedding_dir}. Run Stage 5 first.")
        return

    logger.info(f"Input: {embedding_dir}")
    logger.info(f"Found {len(files)} books to upload")

    # Connect to Weaviate
    client = get_client()

    try:
        # Clean slate: delete existing collection
        if delete_collection(client, collection_name):
            logger.info(f"Deleted existing collection: {collection_name}")

        # Create fresh collection (RAPTOR needs extended schema)
        is_raptor = args.strategy == "raptor"
        if is_raptor:
            create_raptor_collection(client, collection_name)
            logger.info(f"Created RAPTOR collection: {collection_name}")
        else:
            create_collection(client, collection_name)
            logger.info(f"Created collection: {collection_name}")

        # Upload each book
        total_chunks = 0
        for file_path in sorted(files):
            try:
                count = upload_book(client, collection_name, file_path, is_raptor=is_raptor)
                total_chunks += count
            except Exception as e:
                logger.error(f"Failed uploading {file_path.name}: {e}")
                raise

        # Verify final count
        final_count = get_collection_count(client, collection_name)
        logger.info(f"Stage 6 complete: {final_count} chunks in {collection_name}")

        if final_count != total_chunks:
            logger.warning(
                f"Count mismatch: uploaded {total_chunks}, found {final_count}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
