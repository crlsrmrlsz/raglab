"""Comprehensive evaluation: grid search across collections, alphas, strategies.

## RAG Theory: Systematic Evaluation

Comprehensive evaluation tests all valid combinations of:
- Collections (chunking strategies)
- Alpha values (keyword vs vector balance)
- Reranking (on/off per strategy constraints)
- Preprocessing strategies (none, hyde, decomposition, graphrag)

Uses StrategyConfig constraints to filter invalid combinations.
Generates leaderboard reports with statistical analysis.

## Data Flow

1. Build valid grid combinations from StrategyConfig constraints
2. Run each combination with the standard evaluation pipeline
3. Save checkpoints after each combination (crash resilience)
4. Generate statistical breakdown and leaderboard report
"""

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import time

from src.config import (
    EVAL_DEFAULT_METRICS,
    EVAL_GRAPHRAG_METRICS,
    EVAL_LOGS_DIR,
    PROJECT_ROOT,
)
from src.evaluation import run_evaluation
from src.rag_pipeline.retrieval.query_strategy_config import (
    is_valid_combination,
    list_strategy_configs,
    EVAL_TOP_K,
    get_all_collections,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)

# Paths
COMPREHENSIVE_QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation" / "questions" / "comprehensive_questions.json"
EVAL_TEST_QUESTIONS_FILE = Path(PROJECT_ROOT / "data" / "evaluation" / "questions" / "test_questions.json")
RESULTS_DIR = Path(PROJECT_ROOT / "data" / "evaluation" / "results")

# Type alias for retrieval cache (used in comprehensive mode)
# Cache key: (question_id, collection, search_type, alpha, strategy) - top_k is NOT in key
RetrievalCacheKey = tuple[str, str, str, float, str]
RetrievalCache = dict[RetrievalCacheKey, list[str]]  # Maps cache key -> context strings


def setup_file_logging(timestamp: str) -> Path:
    """Set up file logging for comprehensive evaluation.

    Creates a log file that captures all logger output for later review.
    The file handler is added to the root logger so all modules' logs are captured.

    Args:
        timestamp: Timestamp string for the log filename.

    Returns:
        Path to the log file.
    """
    # Ensure logs directory exists
    EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = EVAL_LOGS_DIR / f"comprehensive_{timestamp}.log"

    # Create file handler with same format as console
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Use a detailed format for file logs
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Add to root logger so all modules' logs are captured
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info(f"Log file: {log_path}")

    return log_path


def resolve_questions_file(questions_file_arg: Optional[str]) -> Path:
    """Resolve questions file path from CLI argument.

    Args:
        questions_file_arg: CLI argument value (None, 'full', or path).

    Returns:
        Resolved path to questions file.
    """
    if questions_file_arg is None:
        return COMPREHENSIVE_QUESTIONS_FILE
    if questions_file_arg.lower() == "full":
        return EVAL_TEST_QUESTIONS_FILE
    return Path(questions_file_arg)


def load_test_questions(
    filepath: Path = COMPREHENSIVE_QUESTIONS_FILE,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load test questions from JSON file.

    Args:
        filepath: Path to test questions JSON.
        limit: Max number of questions to load.

    Returns:
        List of question dictionaries.

    Raises:
        FileNotFoundError: If test questions file doesn't exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {filepath}\n"
            "Create test questions first in data/evaluation/test_questions.json"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Limit number of questions
    if limit and limit < len(questions):
        questions = questions[:limit]
        logger.info(f"Limited to first {limit} questions")

    return questions


# ============================================================================
# COMPREHENSIVE EVALUATION - HELPER FUNCTIONS
# ============================================================================


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set[tuple]]:
    """Load checkpoint and extract completed combination keys.

    Each combination is identified by (collection, alpha, reranking, strategy).
    Results with errors are still considered "completed" to avoid re-running
    known failures (use --retry-failed for that).

    Args:
        checkpoint_path: Path to checkpoint JSON file.

    Returns:
        Tuple of (existing_results, completed_keys) where completed_keys is
        a set of (collection, alpha, reranking, strategy) tuples.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        json.JSONDecodeError: If checkpoint file is corrupted.
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_results = data.get("results", [])
    completed_keys = set()

    for result in existing_results:
        key = (
            result["collection"],
            result["alpha"],
            result["reranking"],
            result["strategy"],
        )
        completed_keys.add(key)

    logger.info(
        f"Loaded checkpoint: {len(completed_keys)} completed of {data.get('total', '?')} total"
    )

    return existing_results, completed_keys


def _build_valid_combinations(
    available_collections: list[str],
) -> tuple[list[tuple], list[tuple], list[str]]:
    """Build valid grid combinations using StrategyConfig constraints.

    Separates strategies into standard (grid search) and dedicated (single run),
    then builds valid combinations for each.

    Args:
        available_collections: Collections available in Weaviate.

    Returns:
        Tuple of (valid_combinations, skipped_combinations, all_strategy_ids).
        - valid_combinations: List of (collection, alpha, reranking, strategy) tuples.
        - skipped_combinations: List of (collection, alpha, reranking, strategy, reason) tuples.
        - all_strategy_ids: List of all strategy IDs for reporting.
    """
    from src.ui.services.search import extract_strategy_from_collection
    from src.rag_pipeline.retrieval.strategy_registry import list_strategies

    # Separate strategies into standard (grid) and dedicated (single run)
    standard_strategies = []
    dedicated_strategies = []
    for config in list_strategy_configs():
        if config.uses_dedicated_index() or config.has_internal_search():
            dedicated_strategies.append(config)
        else:
            standard_strategies.append(config)

    logger.info(f"Standard strategies (grid): {[s.strategy_id for s in standard_strategies]}")
    logger.info(f"Dedicated strategies (single run): {[s.strategy_id for s in dedicated_strategies]}")

    combinations = []
    skipped_combinations = []

    # Standard strategies: iterate over their allowed values
    for config in standard_strategies:
        strategy = config.strategy_id
        allowed_collections = config.get_allowed_collections()
        allowed_alphas = config.get_allowed_alphas()
        allowed_rerankings = config.get_allowed_rerankings()

        for collection_type in allowed_collections:
            # Find matching Weaviate collection
            matching_collection = None
            for coll in available_collections:
                coll_strategy = extract_strategy_from_collection(coll)
                if coll_strategy == collection_type:
                    matching_collection = coll
                    break

            if not matching_collection:
                skipped_combinations.append(
                    (collection_type, "N/A", "N/A", strategy, f"Collection '{collection_type}' not in Weaviate")
                )
                continue

            for alpha in allowed_alphas:
                for rerank in allowed_rerankings:
                    is_valid, error_msg = is_valid_combination(
                        strategy, collection_type, alpha, rerank
                    )
                    if is_valid:
                        combinations.append((matching_collection, alpha, rerank, strategy))
                    else:
                        skipped_combinations.append(
                            (matching_collection, alpha, rerank, strategy, error_msg)
                        )

    # Dedicated strategies: use their dedicated collection and defaults
    for config in dedicated_strategies:
        dedicated_collection = config.collection_constraint.dedicated_collection
        alpha = config.get_default_alpha()
        rerank = config.get_default_reranking()

        if dedicated_collection in available_collections:
            combinations.append((dedicated_collection, alpha, rerank, config.strategy_id))
            logger.info(
                f"Added dedicated strategy: {config.strategy_id} with "
                f"collection={dedicated_collection}, alpha={alpha}, rerank={rerank}"
            )
        else:
            logger.warning(
                f"Skipping {config.strategy_id}: dedicated collection "
                f"'{dedicated_collection}' not found in Weaviate"
            )

    return combinations, skipped_combinations, list_strategies()


def print_combination_table() -> None:
    """Print all valid evaluation combinations as a formatted table.

    Enumerates every combination that --comprehensive would run,
    grouped by strategy. Useful for verifying the grid before a long run.
    """
    from src.ui.services.search import list_collections

    available_collections = list_collections()
    if not available_collections:
        logger.error("No RAG collections found in Weaviate. Run stage 6 first.")
        return

    combinations, skipped, all_strategies = _build_valid_combinations(available_collections)

    logger.info("=" * 110)
    logger.info("COMPREHENSIVE EVALUATION: ALL VALID COMBINATIONS")
    logger.info("=" * 110)
    logger.info(f"Total: {len(combinations)} combinations")
    logger.info(f"Available collections: {available_collections}")
    logger.info(f"Strategies: {all_strategies}")
    logger.info(f"Skipped (invalid): {len(skipped)}")
    logger.info("")

    # Table header
    logger.info(f"{'#':<4} {'Strategy':<16} {'Collection':<40} {'Alpha':<7} {'Rerank':<8} {'Notes'}")
    logger.info("-" * 110)

    # Group notes by strategy for readability
    strategy_notes = {
        "none": {
            (0.0, False): "Pure BM25 keyword search",
            (0.0, True): "BM25 + reranking",
            (0.5, False): "Balanced hybrid",
            (0.5, True): "Balanced hybrid + reranking",
            (1.0, False): "Pure vector",
            (1.0, True): "Pure vector + reranking",
        },
        "hyde": {
            (1.0, False): "HyDE avg embedding",
            (1.0, True): "HyDE + reranking",
        },
        "decomposition": {
            (1.0, True): "Sub-questions + mandatory reranking",
        },
        "graphrag": {
            (1.0, False): "Graph traversal (local) + DRIFT (global)",
        },
    }

    for i, (collection, alpha, reranking, strategy) in enumerate(combinations, 1):
        rerank_str = "ON" if reranking else "off"
        notes = strategy_notes.get(strategy, {}).get((alpha, reranking), "")
        coll_short = collection[:40]
        logger.info(f"{i:<4} {strategy:<16} {coll_short:<40} {alpha:<7} {rerank_str:<8} {notes}")

    # Show skipped combinations
    if skipped:
        logger.info(f"{'SKIPPED COMBINATIONS':}")
        logger.info("-" * 110)
        for coll, alpha, rerank, strat, reason in skipped[:10]:
            logger.info(f"  {strat} | {coll} | alpha={alpha} | rerank={rerank}: {reason}")
        if len(skipped) > 10:
            logger.info(f"  ... and {len(skipped) - 10} more")

    logger.info("=" * 110)


def _log_grid_summary(
    combinations: list[tuple],
    skipped_combinations: list[tuple],
    available_collections: list[str],
    all_strategies: list[str],
    top_k: int,
) -> None:
    """Log summary of grid parameters and skipped combinations.

    Args:
        combinations: Valid combinations to test.
        skipped_combinations: Combinations skipped with reasons.
        available_collections: Collections available in Weaviate.
        all_strategies: All strategy IDs.
        top_k: Fixed top_k value.
    """
    logger.info(f"Testing {len(combinations)} valid combinations:")
    logger.info(f"  Available collections: {available_collections}")
    logger.info(f"  Defined collection types: {get_all_collections()}")
    logger.info(f"  Top-K: {top_k} (fixed)")
    logger.info(f"  Preprocessing strategies: {all_strategies}")
    logger.info(f"  Skipped invalid: {len(skipped_combinations)}")

    if skipped_combinations:
        for item in skipped_combinations[:5]:
            coll, alpha, rerank, strat, reason = item
            logger.info(f"    - {strat} + alpha={alpha} + rerank={rerank} on {coll}: {reason}")
        if len(skipped_combinations) > 5:
            logger.info(f"    ... and {len(skipped_combinations) - 5} more")


def _save_checkpoint(
    checkpoint_path: Path,
    count: int,
    total: int,
    all_results: list[dict],
    available_collections: list[str],
    top_k: int,
    all_strategies: list[str],
    combinations: Optional[list[tuple]] = None,
) -> None:
    """Save checkpoint after each combination for crash resilience.

    Args:
        checkpoint_path: Path to checkpoint file.
        count: Number of completed combinations.
        total: Total combinations to process.
        all_results: Results collected so far.
        available_collections: Collections in the grid.
        top_k: Fixed top_k value.
        all_strategies: All strategy IDs.
        combinations: Full list of (collection, alpha, reranking, strategy) tuples
            for validation on resume.
    """
    progress_pct = round(count / total * 100, 1)
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "completed": count,
        "total": total,
        "progress_pct": progress_pct,
        "results": all_results,
        "grid_params": {
            "collections": available_collections,
            "top_k": top_k,
            "strategies": all_strategies,
        },
    }
    if combinations is not None:
        checkpoint_data["combinations"] = [
            [c, a, r, s] for c, a, r, s in combinations
        ]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"  Checkpoint saved: {count}/{total} ({progress_pct}%)")


def _run_single_combination(
    collection: str,
    alpha: float,
    use_reranking: bool,
    strategy: str,
    questions: list[dict],
    top_k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run evaluation for a single combination and return result dict.

    Args:
        collection: Weaviate collection name.
        alpha: Hybrid search alpha value.
        use_reranking: Whether to use reranking.
        strategy: Preprocessing strategy ID.
        questions: Test questions.
        top_k: Number of chunks to retrieve.
        args: CLI arguments with model settings.

    Returns:
        Result dict with scores, or error information if failed.
    """
    eval_metrics = EVAL_GRAPHRAG_METRICS if strategy == "graphrag" else EVAL_DEFAULT_METRICS

    results = run_evaluation(
        test_questions=questions,
        metrics=eval_metrics,
        top_k=top_k,
        generation_model=args.generation_model,
        evaluation_model=args.evaluation_model,
        collection_name=collection,
        use_reranking=use_reranking,
        alpha=alpha,
        preprocessing_strategy=strategy,
        preprocessing_model=None,
        save_trace=True,
        search_type="hybrid",
    )

    trace_path = results.get("trace_path")
    questions_processed = results.get("questions_processed", len(questions))
    questions_failed = results.get("questions_failed", 0)
    failed_questions = results.get("failed_questions", [])

    return {
        "collection": collection,
        "alpha": alpha,
        "reranking": use_reranking,
        "top_k": top_k,
        "strategy": strategy,
        "scores": results["scores"],
        "difficulty_breakdown": results.get("difficulty_breakdown", {}),
        "num_questions": len(questions),
        "questions_processed": questions_processed,
        "questions_failed": questions_failed,
        "failed_questions": failed_questions,
        "trace_path": str(trace_path) if trace_path else None,
    }


def _determine_failure_stage(error: Exception) -> str:
    """Determine which stage failed based on error message.

    Args:
        error: The exception that was raised.

    Returns:
        Stage name: "preprocessing", "retrieval", "generation", or "ragas_evaluation".
    """
    error_str = str(error).lower()
    if "preprocessing" in error_str:
        return "preprocessing"
    elif "retrieval" in error_str or "weaviate" in error_str:
        return "retrieval"
    elif "generation" in error_str:
        return "generation"
    return "ragas_evaluation"


# ============================================================================
# COMPREHENSIVE EVALUATION - MAIN FUNCTION
# ============================================================================


def run_comprehensive_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive evaluation across all VALID combinations with failure tracking.

    Tests: collections x alphas x reranking x preprocessing_strategies
    - Uses StrategyConfig constraints to filter invalid combinations
    - Reranking is a grid dimension (optional for 'none', required/forbidden for others)
    - top_k is fixed at EVAL_TOP_K (not a grid dimension)
    - Uses curated question subset
    - Generates enhanced leaderboard report with statistical analysis
    - Outputs article-ready summary with key findings
    - Saves failed_combinations.json for later retry
    - No individual run logging to evaluation-history.md

    Args:
        args: Command-line arguments (uses generation_model, evaluation_model, output).
    """
    from src.ui.services.search import list_collections
    from src.evaluation.schemas import FailedCombinationsReport

    # Generate timestamp early for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_file_logging(timestamp)

    logger.info("Starting comprehensive evaluation mode...")
    start_time = time.time()

    # Load questions
    questions_filepath = resolve_questions_file(args.questions_file)
    questions = load_test_questions(filepath=questions_filepath)
    if not questions:
        logger.error(f"No questions found in {questions_filepath}")
        return
    logger.info(f"Loaded {len(questions)} questions from {questions_filepath.name}")

    # Get available collections from Weaviate
    available_collections = list_collections()
    if not available_collections:
        logger.error("No RAG collections found in Weaviate. Run stage 6 first.")
        return

    # Fixed parameters (not grid dimensions)
    top_k = EVAL_TOP_K
    logger.info(f"Fixed parameters: top_k={top_k}")

    # Build valid combinations
    combinations, skipped_combinations, all_strategies = _build_valid_combinations(
        available_collections
    )

    # Log grid summary
    _log_grid_summary(
        combinations, skipped_combinations, available_collections, all_strategies, top_k
    )

    # Initialize tracking
    total_combinations = len(combinations)
    all_results = []
    completed_keys: set[tuple] = set()
    comprehensive_run_id = f"comprehensive_{timestamp}"
    checkpoint_path = RESULTS_DIR / f"comprehensive_checkpoint_{timestamp}.json"

    # Resume from checkpoint if requested
    resume_path = getattr(args, "resume", None)
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            logger.error(f"Checkpoint file not found: {resume_file}")
            return
        all_results, completed_keys = _load_checkpoint(resume_file)
        # Reuse the original checkpoint path for continued saves
        checkpoint_path = resume_file
        logger.info(
            f"Resumed: {len(completed_keys)}/{total_combinations} already completed, "
            f"{total_combinations - len(completed_keys)} remaining"
        )

    failed_report = FailedCombinationsReport(
        comprehensive_run_id=comprehensive_run_id,
        timestamp=datetime.now().isoformat(),
    )

    # Run all combinations (skipping completed on resume)
    for count, (collection, alpha, use_reranking, strategy) in enumerate(combinations, 1):
        combo_key = (collection, alpha, use_reranking, strategy)
        if combo_key in completed_keys:
            logger.info(f"  [{count}/{total_combinations}] SKIP (already completed): "
                        f"{collection} | alpha={alpha} | strategy={strategy}")
            continue

        rerank_str = "ON" if use_reranking else "off"
        logger.info(
            f"\n[{count}/{total_combinations}] "
            f"{collection} | alpha={alpha} | rerank={rerank_str} | strategy={strategy}"
        )

        try:
            result = _run_single_combination(
                collection, alpha, use_reranking, strategy, questions, top_k, args
            )
            all_results.append(result)

            scores = result["scores"]
            questions_failed = result.get("questions_failed", 0)
            status_suffix = f" ({questions_failed} questions failed)" if questions_failed > 0 else ""
            logger.info(
                f"  -> faith={scores.get('faithfulness', 0):.3f} "
                f"relev={scores.get('relevancy', 0):.3f} "
                f"ctx_prec={scores.get('context_precision', 0):.3f} "
                f"ctx_rec={scores.get('context_recall', 0):.3f} "
                f"ans_corr={scores.get('answer_correctness', 0):.3f}{status_suffix}"
            )

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")

            failed_at_stage = _determine_failure_stage(e)
            failed_report.add_failure(
                collection=collection,
                alpha=alpha,
                top_k=top_k,
                strategy=strategy,
                error=e,
                failed_at_stage=failed_at_stage,
                reranking=use_reranking,
            )

            all_results.append({
                "collection": collection,
                "alpha": alpha,
                "reranking": use_reranking,
                "top_k": top_k,
                "strategy": strategy,
                "scores": {m: 0 for m in EVAL_DEFAULT_METRICS},
                "num_questions": len(questions),
                "error": str(e),
                "error_type": type(e).__name__,
            })

        # Save checkpoint after each combination
        _save_checkpoint(
            checkpoint_path, count, total_combinations, all_results,
            available_collections, top_k, all_strategies, combinations
        )

    # Generate report
    duration_seconds = time.time() - start_time
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"comprehensive_{timestamp}.json"

    grid_params = {
        "collections": available_collections,
        "top_k": top_k,
        "strategies": all_strategies,
        "valid_combinations_count": total_combinations,
    }

    generate_comprehensive_report(
        all_results, questions, output_path, duration_seconds, grid_params, log_path
    )

    # Save failed combinations if any
    if failed_report.total_failed > 0:
        failed_path = RESULTS_DIR / f"failed_combinations_{timestamp}.json"
        failed_report.save(failed_path)
        logger.info(f"Failed combinations saved to: {failed_path}")
        logger.info(f"  To retry: {failed_report.cli_command}")

    logger.info("Comprehensive evaluation complete!")


def retry_failed_combinations(run_id: str, args: argparse.Namespace) -> None:
    """Re-run failed combinations from a previous comprehensive evaluation run.

    Loads the failed_combinations file for the given run_id and re-runs each
    failed combination. Generates a new report with the retry results.

    Args:
        run_id: The comprehensive run ID (e.g., "comprehensive_20251231_120000").
        args: Command-line arguments (uses generation_model, evaluation_model).
    """
    from src.evaluation.schemas import FailedCombinationsReport, FailedCombination

    # Parse the timestamp from run_id if it contains it
    if run_id.startswith("comprehensive_"):
        timestamp_part = run_id.replace("comprehensive_", "")
    else:
        timestamp_part = run_id

    # Try to find the failed combinations file
    failed_path = RESULTS_DIR / f"failed_combinations_{timestamp_part}.json"
    if not failed_path.exists():
        # Also try with the full run_id as filename
        failed_path = RESULTS_DIR / f"failed_combinations_{run_id}.json"

    if not failed_path.exists():
        logger.error(f"Failed combinations file not found: {failed_path}")
        logger.info(f"Available files in {RESULTS_DIR}:")
        for f in sorted(RESULTS_DIR.glob("failed_combinations_*.json")):
            logger.info(f"  {f.name}")
        return

    logger.info(f"Loading failed combinations from: {failed_path}")

    # Load the failed combinations report
    failed_report = FailedCombinationsReport.load(failed_path)

    if failed_report.total_failed == 0:
        logger.info("No failed combinations to retry.")
        return

    logger.info(f"Found {failed_report.total_failed} failed combinations to retry")

    # Load questions (respect --questions-file if provided)
    questions_filepath = resolve_questions_file(args.questions_file)
    questions = load_test_questions(filepath=questions_filepath)
    logger.info(f"Loaded {len(questions)} questions from {questions_filepath.name}")

    start_time = time.time()
    retry_results = []
    new_failures = FailedCombinationsReport(
        comprehensive_run_id=f"retry_{run_id}",
        timestamp=datetime.now().isoformat(),
    )

    for i, fc in enumerate(failed_report.failed_combinations):
        rerank_str = "ON" if fc.reranking else "off"
        logger.info(
            f"\n[{i + 1}/{failed_report.total_failed}] Retrying: "
            f"{fc.collection} | alpha={fc.alpha} | rerank={rerank_str} | "
            f"top_k={fc.top_k} | strategy={fc.strategy}"
        )

        try:
            eval_metrics = EVAL_GRAPHRAG_METRICS if fc.strategy == "graphrag" else EVAL_DEFAULT_METRICS

            results = run_evaluation(
                test_questions=questions,
                metrics=eval_metrics,
                top_k=fc.top_k,
                generation_model=args.generation_model,
                evaluation_model=args.evaluation_model,
                collection_name=fc.collection,
                use_reranking=fc.reranking,
                alpha=fc.alpha,
                preprocessing_strategy=fc.strategy,
                preprocessing_model=None,
                save_trace=True,  # Save traces for retried runs (debugging)
            )

            retry_results.append({
                "collection": fc.collection,
                "alpha": fc.alpha,
                "reranking": fc.reranking,
                "top_k": fc.top_k,
                "strategy": fc.strategy,
                "scores": results["scores"],
                "difficulty_breakdown": results.get("difficulty_breakdown", {}),
                "num_questions": len(questions),
                "retry_success": True,
                "original_error": fc.error_message,
            })

            scores = results["scores"]
            logger.info(
                f"  -> SUCCESS: faith={scores.get('faithfulness', 0):.3f} "
                f"relev={scores.get('relevancy', 0):.3f} "
                f"ans_corr={scores.get('answer_correctness', 0):.3f}"
            )

        except Exception as e:
            logger.error(f"  -> FAILED AGAIN: {e}")

            # Track the new failure
            new_failures.add_failure(
                collection=fc.collection,
                alpha=fc.alpha,
                top_k=fc.top_k,
                strategy=fc.strategy,
                error=e,
                failed_at_stage=fc.failed_at_stage,
                reranking=fc.reranking,
            )

            retry_results.append({
                "collection": fc.collection,
                "alpha": fc.alpha,
                "reranking": fc.reranking,
                "top_k": fc.top_k,
                "strategy": fc.strategy,
                "scores": {m: 0 for m in EVAL_DEFAULT_METRICS},
                "num_questions": len(questions),
                "retry_success": False,
                "error": str(e),
                "original_error": fc.error_message,
            })

    # Calculate duration
    duration_seconds = time.time() - start_time

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"retry_results_{timestamp}.json"

    successful_retries = sum(1 for r in retry_results if r.get("retry_success"))
    failed_retries = len(retry_results) - successful_retries

    report = {
        "metadata": {
            "original_run_id": run_id,
            "retry_timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "total_retried": len(retry_results),
            "successful_retries": successful_retries,
            "failed_retries": failed_retries,
        },
        "retry_results": retry_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nRetry results saved to: {output_path}")
    logger.info(f"  Successful: {successful_retries}/{len(retry_results)}")
    logger.info(f"  Still failing: {failed_retries}/{len(retry_results)}")

    # Save new failures if any remain
    if new_failures.total_failed > 0:
        new_failed_path = RESULTS_DIR / f"failed_combinations_retry_{timestamp}.json"
        new_failures.save(new_failed_path)
        logger.info(f"  Remaining failures saved to: {new_failed_path}")


def compute_statistical_breakdown(
    results: list[dict[str, Any]],
    group_key: str,
    metrics: Optional[list[str]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute statistical analysis (mean, std, min, max) for each group and metric.

    Args:
        results: List of result dicts (only successful runs).
        group_key: Key to group by ("strategy", "alpha", "top_k", "collection").
        metrics: List of metric names to analyze. Defaults to EVAL_DEFAULT_METRICS.

    Returns:
        Dict mapping group values to per-metric stats.
        Structure: {group_value: {metric_name: {mean, std, min, max, n}}}.
    """
    if metrics is None:
        metrics = EVAL_DEFAULT_METRICS

    # Collect scores per group per metric
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" not in r:
            group_val = str(r[group_key])
            for metric in metrics:
                groups[group_val][metric].append(r["scores"].get(metric) or 0)

    analysis: dict[str, dict[str, dict[str, float]]] = {}
    for group, metric_scores in groups.items():
        analysis[group] = {}
        for metric, scores in metric_scores.items():
            analysis[group][metric] = {
                "mean": round(statistics.mean(scores), 4),
                "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "n": len(scores),
            }
    return analysis


def find_best_configurations(
    successful_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find best configurations per metric.

    Args:
        successful_runs: Only successful runs.

    Returns:
        Dict with per-metric best configurations.
    """
    best = {}

    # Find best per metric
    for metric in EVAL_DEFAULT_METRICS:
        sorted_by_metric = sorted(
            successful_runs,
            key=lambda x: x["scores"].get(metric) or 0,
            reverse=True,
        )
        if sorted_by_metric:
            top = sorted_by_metric[0]
            best[f"by_{metric}"] = {
                "collection": top["collection"],
                "alpha": top["alpha"],
                "top_k": top["top_k"],
                "strategy": top["strategy"],
                "score": round(top["scores"].get(metric) or 0, 4),
            }

    return best


def generate_comprehensive_report(
    all_results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    output_path: Path,
    duration_seconds: float = 0.0,
    grid_params: Optional[dict[str, Any]] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Generate enhanced leaderboard and statistical analysis for articles.

    Produces:
    - JSON report with experiment metadata, statistical analysis, best configs
    - Console leaderboard with rankings
    - Article-ready summary with key findings

    Args:
        all_results: List of result dicts with collection, alpha, strategy, scores.
        questions: Original test questions.
        output_path: Path for output JSON file.
        duration_seconds: Total experiment duration in seconds.
        grid_params: Grid search parameters (collections, alphas, strategies).
        log_path: Path to the execution log file.
    """
    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by faithfulness (primary metric for grounded answers)
    sorted_results = sorted(
        all_results,
        key=lambda x: x["scores"].get("faithfulness") or 0,
        reverse=True
    )

    # Separate successful and failed runs
    successful_runs = [r for r in all_results if "error" not in r]
    failed_runs = [r for r in all_results if "error" in r]

    # Compute statistical analysis
    strategy_analysis = compute_statistical_breakdown(successful_runs, "strategy")
    alpha_analysis = compute_statistical_breakdown(successful_runs, "alpha")
    reranking_analysis = compute_statistical_breakdown(successful_runs, "reranking")
    collection_analysis = compute_statistical_breakdown(successful_runs, "collection")

    # Find best configurations
    best_configs = find_best_configurations(successful_runs)

    # Format duration
    duration_minutes = duration_seconds / 60
    duration_str = f"{int(duration_minutes)} minutes {int(duration_seconds % 60)} seconds"

    # Build enhanced report
    report = {
        "experiment_metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "duration_formatted": duration_str,
            "total_combinations": len(all_results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "log_file": str(log_path) if log_path else None,
        },
        "grid_parameters": {
            "collections": grid_params.get("collections", []) if grid_params else [],
            "alphas": grid_params.get("alphas", []) if grid_params else [],
            "top_k_values": grid_params.get("top_k_values", []) if grid_params else [],
            "strategies": grid_params.get("strategies", []) if grid_params else [],
            "num_questions": len(questions),
            "reranking_enabled": False,
        },
        "leaderboard": sorted_results,
        "statistical_analysis": {
            "by_strategy": strategy_analysis,
            "by_alpha": alpha_analysis,
            "by_reranking": reranking_analysis,
            "by_collection": collection_analysis,
        },
        "best_configurations": best_configs,
        "failed_runs": [
            {
                "collection": r["collection"],
                "alpha": r["alpha"],
                "top_k": r["top_k"],
                "strategy": r["strategy"],
                "error": r.get("error", "Unknown"),
            }
            for r in failed_runs
        ],
    }

    # Save JSON report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comprehensive report saved to: {output_path}")

    # =========================================================================
    # CONSOLE OUTPUT: Leaderboard
    # =========================================================================
    logger.info("=" * 100)
    logger.info("COMPREHENSIVE EVALUATION LEADERBOARD")
    logger.info("=" * 100)
    logger.info(f"Tested {len(all_results)} combinations on {len(questions)} questions")
    logger.info(f"Duration: {duration_str}")
    logger.info(f"Successful: {len(successful_runs)} | Failed: {len(failed_runs)}")
    logger.info("-" * 127)
    logger.info(f"{'Rank':<5} {'Collection':<35} {'Alpha':<7} {'Rerank':<8} {'Strategy':<15} {'Faith':<8} {'Relev':<8} {'CtxPrec':<8} {'CtxRec':<8} {'AnsCorr':<8}")
    logger.info("-" * 127)

    for i, result in enumerate(sorted_results, 1):
        scores = result["scores"]
        faith = scores.get("faithfulness") or 0
        relev = scores.get("relevancy") or 0
        ctx_prec = scores.get("context_precision") or 0
        ctx_rec = scores.get("context_recall") or 0
        ans_corr = scores.get("answer_correctness") or 0
        collection_short = result["collection"][:35]
        rerank_str = "ON" if result.get("reranking") else "off"
        error_marker = " *" if "error" in result else ""

        logger.info(
            f"{i:<5} {collection_short:<35} {result['alpha']:<7} "
            f"{rerank_str:<8} {result['strategy']:<15} {faith:<8.3f} {relev:<8.3f} "
            f"{ctx_prec:<8.3f} {ctx_rec:<8.3f} {ans_corr:<8.3f}{error_marker}"
        )

    # =========================================================================
    # CONSOLE OUTPUT: Statistical Breakdowns
    # =========================================================================
    def _print_breakdown(title: str, analysis: dict[str, dict[str, dict[str, float]]], format_label=str):
        logger.info("=" * 100)
        logger.info(title)
        logger.info("=" * 100)
        # Sort groups by faithfulness mean (primary metric) for consistent ordering
        def _sort_key(group_name: str) -> float:
            return analysis[group_name].get("faithfulness", {}).get("mean", 0)

        for group in sorted(analysis.keys(), key=_sort_key, reverse=True):
            metric_stats = analysis[group]
            label = format_label(group)
            # Use n from the first metric (all metrics have the same n per group)
            first_metric = next(iter(metric_stats.values()))
            logger.info(f"{label} (n={first_metric['n']}):")
            for metric, stats in metric_stats.items():
                metric_short = metric[:16]
                logger.info(f"  {metric_short:<17} {stats['mean']:.3f}  +/-  {stats['std']:.3f}  [{stats['min']:.3f}, {stats['max']:.3f}]")

    _print_breakdown("BREAKDOWN BY PREPROCESSING STRATEGY", strategy_analysis, str.upper)
    _print_breakdown("BREAKDOWN BY ALPHA (0.0=keyword, 1.0=vector)", alpha_analysis, lambda a: f"ALPHA={a}")
    _print_breakdown("BREAKDOWN BY RERANKING", reranking_analysis, lambda r: f"RERANK={'ON' if r == 'True' else 'off'}")
    _print_breakdown("BREAKDOWN BY COLLECTION", collection_analysis)

    # =========================================================================
    # CONSOLE OUTPUT: Article Summary
    # =========================================================================
    logger.info("=" * 100)
    logger.info("ARTICLE SUMMARY")
    logger.info("=" * 100)

    logger.info(f"EXPERIMENT METADATA")
    logger.info(f"  Duration: {duration_str}")
    logger.info(f"  Combinations tested: {len(all_results)} ({len(successful_runs)} successful, {len(failed_runs)} failed)")
    logger.info(f"  Questions per run: {len(questions)}")

    logger.info(f"BEST CONFIGURATIONS")
    for metric in EVAL_DEFAULT_METRICS:
        key = f"by_{metric}"
        if best_configs.get(key):
            b = best_configs[key]
            logger.info(f"  {metric.title():<15} {b['collection'][:25]} + alpha={b['alpha']} + top_k={b['top_k']} + {b['strategy']} ({b['score']:.3f})")

    # Calculate improvement percentages vs baseline (none strategy) for each metric
    if "none" in strategy_analysis:
        logger.info(f"STRATEGY IMPROVEMENT VS BASELINE (none)")
        for metric in EVAL_DEFAULT_METRICS:
            baseline = strategy_analysis["none"].get(metric, {}).get("mean", 0)
            logger.info(f"  {metric}  (baseline={baseline:.3f}):")
            for strategy in sorted(strategy_analysis.keys()):
                stats = strategy_analysis[strategy].get(metric, {})
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                if strategy == "none":
                    logger.info(f"    {strategy.upper():<15} {mean:.3f} +/- {std:.3f}  (baseline)")
                else:
                    improvement = ((mean - baseline) / baseline) * 100 if baseline > 0 else 0
                    logger.info(f"    {strategy.upper():<15} {mean:.3f} +/- {std:.3f}  [{improvement:+.1f}% vs baseline]")

    # Show failed runs if any
    if failed_runs:
        logger.info("=" * 100)
        logger.info("FAILED RUNS")
        logger.info("=" * 100)
        for result in failed_runs:
            logger.info(f"  {result['collection']} | alpha={result['alpha']} | top_k={result['top_k']} | {result['strategy']}")
            logger.info(f"    Error: {result.get('error', 'Unknown')}")

    logger.info("=" * 100)
