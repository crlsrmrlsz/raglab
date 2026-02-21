"""
Stage 4b: Build RAPTOR hierarchical summary trees.

This stage runs between Stage 4 (chunking) and Stage 5 (embedding).
It takes semantic chunks (std=2) as input and builds a tree of summaries.

RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
Paper: arXiv:2401.18059 (ICLR 2024)

Usage:
    python -m src.stages.run_stage_4b_raptor
    python -m src.stages.run_stage_4b_raptor --max-levels 3
    python -m src.stages.run_stage_4b_raptor --overwrite all

Input:  data/processed/05_final_chunks/semantic_std2/{book}.json
Output: data/processed/05_final_chunks/raptor/{book}.json
"""

import argparse

from src.config import (
    DIR_FINAL_CHUNKS,
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_CLUSTER_SIZE,
    RAPTOR_SUMMARY_MODEL,
)
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.rag_pipeline.chunking.raptor.raptor_chunker import run_raptor_chunking

logger = setup_logging("Stage4b_RAPTOR")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4b: Build RAPTOR hierarchical summary trees"
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=RAPTOR_MAX_LEVELS,
        help=f"Maximum tree depth (default: {RAPTOR_MAX_LEVELS})",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=RAPTOR_MIN_CLUSTER_SIZE,
        help=f"Minimum nodes for clustering (default: {RAPTOR_MIN_CLUSTER_SIZE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=RAPTOR_SUMMARY_MODEL,
        help=f"OpenRouter model for summarization (default: {RAPTOR_SUMMARY_MODEL})",
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

    logger.info("=" * 60)
    logger.info("Stage 4b: RAPTOR Tree Building")
    logger.info("=" * 60)
    logger.info(f"Input:  {DIR_FINAL_CHUNKS}/semantic_std2/")
    logger.info(f"Output: {DIR_FINAL_CHUNKS}/raptor/")
    logger.info(f"Max levels: {args.max_levels}")
    logger.info(f"Min cluster size: {args.min_cluster_size}")
    logger.info(f"Summary model: {args.model}")
    logger.info("-" * 60)

    try:
        stats = run_raptor_chunking(
            max_levels=args.max_levels,
            min_cluster_size=args.min_cluster_size,
            summary_model=args.model,
            overwrite_context=overwrite_context,
        )

        logger.info("-" * 60)
        total_nodes = sum(m.total_nodes for m in stats.values())
        total_leaves = sum(m.leaf_count for m in stats.values())
        total_summaries = sum(m.summary_count for m in stats.values())
        max_depth = max(m.max_level for m in stats.values()) if stats else 0
        logger.info(f"Stage 4b complete. Processed {len(stats)} books.")
        logger.info(
            f"Total: {total_nodes} nodes ({total_leaves} leaves, {total_summaries} summaries)"
        )
        logger.info(f"Max tree depth across all books: {max_depth} levels")

        # Print per-book statistics
        if stats:
            logger.info("-" * 60)
            logger.info("Per-book statistics:")
            for book_name, metadata in sorted(stats.items()):
                logger.info(
                    f"  {book_name}: {metadata.total_nodes} nodes "
                    f"({metadata.leaf_count} leaves, {metadata.summary_count} summaries), "
                    f"{metadata.max_level} levels, {metadata.build_time_seconds:.1f}s"
                )

        logger.info("=" * 60)
        logger.info("Next step: python -m src.stages.run_stage_5_embedding --strategy raptor")

    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run semantic chunking first:")
        logger.error("  python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0")
        raise

    except Exception as e:
        logger.error(f"Stage 4b failed: {e}")
        raise


if __name__ == "__main__":
    main()
