"""
Stage 5: Embed final text chunks for RAG.

This stage:
- Loads chunks from Stage 4 (strategy-specific directory)
- Calls embedding API (OpenAI-compatible)
- Saves embeddings to disk (no vector DB yet)

Design goals:
- Deterministic
- Restartable
- Transparent

Usage:
    python -m src.stages.run_stage_5_embedding                    # Default: section
    python -m src.stages.run_stage_5_embedding --strategy semantic  # Semantic chunks
    python -m src.stages.run_stage_5_embedding --strategy semantic --threshold 0.6
"""

import argparse
import json
from pathlib import Path

from src.config import (
    DIR_FINAL_CHUNKS,
    EMBEDDING_MODEL,
    DEFAULT_CHUNKING_STRATEGY,
    SEMANTIC_STD_COEFFICIENT,
    get_semantic_folder_name,
    get_embedding_folder_path,
)

from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.rag_pipeline.embedder import embed_texts
from src.rag_pipeline.chunking.strategies import list_strategies

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def load_chunks(file_path: Path) -> list[dict]:
    """Load chunk list from a JSON file.

    Handles both flat list format (section/contextual) and nested format (raptor).
    RAPTOR files have structure: {"book_id": ..., "tree_metadata": ..., "chunks": [...]}
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle nested format (raptor)
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]

    # Handle flat list format (section, contextual, semantic)
    return data


def embed_book(file_path: Path, output_dir: Path):
    """
    Embed all chunks for a single book.

    Args:
        file_path: Path to the input chunk JSON file.
        output_dir: Directory where embedding file will be saved.

    Note:
        Preserves all chunk fields from the input (including strategy-specific
        fields like 'original_text' and 'contextual_snippet' for contextual
        chunking). Adds embedding fields: 'embedding', 'embedding_model',
        'embedding_dim'.

        Batching is handled internally by embed_texts() to stay under
        MAX_BATCH_TOKENS per API call.
    """
    logger.info(f"Embedding book: {file_path.stem}")

    chunks = load_chunks(file_path)
    texts = [c["text"] for c in chunks]

    logger.info(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)  # Batching handled internally

    # Build embedded chunks preserving all existing fields
    embedded_chunks = []
    for chunk, vector in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": vector,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": len(vector)
        })

    # Save per-book embedding file
    output_path = output_dir / f"{file_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "book_id": file_path.stem,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": embedded_chunks
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved {len(embedded_chunks)} embeddings to {output_path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Embed chunks from specified chunking strategy."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Embed chunks from specified chunking strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list_strategies(),
        help=f"Chunking strategy to embed (default: {DEFAULT_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--std-coefficient",
        type=float,
        default=None,
        help=(
            f"Std coefficient (for finding correct input folder). "
            f"Only used with semantic strategy. (default: {SEMANTIC_STD_COEFFICIENT})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite behavior: prompt (default), skip, all",
    )
    args = parser.parse_args()

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    # Determine strategy key for paths (semantic uses coefficient-based naming)
    if args.strategy == "semantic":
        coef = args.std_coefficient if args.std_coefficient is not None else SEMANTIC_STD_COEFFICIENT
        strategy_key = get_semantic_folder_name(coef)
    else:
        strategy_key = args.strategy

    logger.info(f"Starting Stage 5: Embedding (strategy: {strategy_key})")

    # Determine input directory (chunks from Stage 4)
    strategy_dir = DIR_FINAL_CHUNKS / strategy_key
    files = list(strategy_dir.glob("*.json")) if strategy_dir.exists() else []

    if not files:
        logger.warning(
            f"No chunks found in {strategy_dir}. "
            f"Run Stage 4 with --strategy {args.strategy} first."
        )
        return

    # Determine output directory (strategy-scoped embeddings)
    output_dir = get_embedding_folder_path(strategy_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input:  {strategy_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Found {len(files)} books to embed")

    success_count = 0
    skipped_count = 0
    for file_path in files:
        # Check overwrite decision (use strategy-scoped output path)
        output_path = output_dir / f"{file_path.stem}.json"
        if not overwrite_context.should_overwrite(output_path, logger):
            skipped_count += 1
            continue

        try:
            embed_book(file_path, output_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed embedding {file_path.name}: {e}")
            raise

    logger.info(f"Stage 5 complete ({strategy_key}). {success_count} embedded, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
