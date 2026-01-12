"""Stage 4: Chunking with strategy selection.

Supports multiple chunking strategies:
- section: Sequential with overlap (baseline, fast)
- semantic: Embedding similarity-based (better coherence, uses API)
- contextual: LLM-generated chunk context (Anthropic-style, uses API)
- raptor: Hierarchical summarization tree (multi-level retrieval, uses API)

Usage:
    python -m src.stages.run_stage_4_chunking                    # Default: section
    python -m src.stages.run_stage_4_chunking --strategy semantic  # Semantic chunking
    python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0
"""

import argparse
from pathlib import Path

from src.config import (
    DIR_NLP_CHUNKS,
    DIR_FINAL_CHUNKS,
    DEFAULT_CHUNKING_STRATEGY,
    SEMANTIC_STD_COEFFICIENT,
    get_semantic_folder_name,
)
from src.shared import (
    setup_logging,
    get_file_list,
    OverwriteContext,
    parse_overwrite_arg,
)
from src.rag_pipeline.chunking.strategies import get_strategy, list_strategies
from src.rag_pipeline.chunking.raptor.schemas import TreeMetadata

logger = setup_logging("Stage4_Chunking")


def main():
    """Run chunking with selected strategy."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Chunking with strategy selection"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list_strategies(),
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--std-coefficient",
        type=float,
        default=None,
        help=(
            f"Standard deviation coefficient for semantic breakpoints. "
            f"Higher = fewer splits (only extreme drops). "
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

    # Build strategy kwargs
    strategy_kwargs = {"overwrite_context": overwrite_context}
    if args.std_coefficient is not None:
        if args.strategy != "semantic":
            logger.warning("--std-coefficient is only used with semantic strategy, ignoring")
        else:
            strategy_kwargs["std_coefficient"] = args.std_coefficient

    logger.info(f"Starting Stage 4: Chunking (strategy: {args.strategy})")

    # Check Stage 3 output exists
    nlp_chunk_files = get_file_list(DIR_NLP_CHUNKS, "json")
    logger.info(f"Found {len(nlp_chunk_files)} NLP chunk files from Stage 3.")

    if not nlp_chunk_files:
        logger.warning(f"No NLP chunk files found in {DIR_NLP_CHUNKS}. Run Stage 3 first.")
        return

    # Get strategy function and run
    strategy_fn = get_strategy(args.strategy, **strategy_kwargs)
    stats = strategy_fn()

    # Determine output directory (semantic uses coefficient-based folder name)
    if args.strategy == "semantic":
        coef = args.std_coefficient if args.std_coefficient is not None else SEMANTIC_STD_COEFFICIENT
        strategy_dir = DIR_FINAL_CHUNKS / get_semantic_folder_name(coef)
    else:
        strategy_dir = DIR_FINAL_CHUNKS / args.strategy
    strategy_files = list(strategy_dir.glob("*.json")) if strategy_dir.exists() else []

    logger.info(f"Stage 4 complete ({args.strategy}). {len(strategy_files)} files in output.")

    # Handle different return types: TreeMetadata (raptor) vs int (other strategies)
    if stats and isinstance(next(iter(stats.values())), TreeMetadata):
        total_chunks = sum(m.total_nodes for m in stats.values())
    else:
        total_chunks = sum(stats.values())
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Output: {strategy_dir}")


if __name__ == "__main__":
    main()
