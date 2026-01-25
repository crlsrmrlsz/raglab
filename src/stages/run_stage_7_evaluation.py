"""Stage 7: RAGAS Evaluation for RAGLab.

Runs RAGAS evaluation on test questions to measure RAG pipeline quality.
Evaluates retrieval (context precision/recall) and generation (faithfulness/relevancy).

Purpose:
    - Measure RAG pipeline quality with standardized RAGAS metrics
    - Compare different retrieval strategies (collections, alpha, reranking)
    - Auto-log results to evaluation-history.md for A/B testing
    - Track configurations in tracking.json for reproducibility

Usage Examples:
    # Run with defaults (hybrid search, alpha=0.5, reranking disabled)
    python -m src.stages.run_stage_7_evaluation

    # Test a specific collection (e.g., contextual embeddings)
    python -m src.stages.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1

    # Alpha tuning experiments
    python -m src.stages.run_stage_7_evaluation --alpha 0.3  # Keyword-heavy (philosophy)
    python -m src.stages.run_stage_7_evaluation --alpha 0.7  # Vector-heavy (conceptual)

    # Enable reranking for higher accuracy (slower)
    python -m src.stages.run_stage_7_evaluation --reranking

    # Run on subset of questions
    python -m src.stages.run_stage_7_evaluation --questions 5

    # Use different models
    python -m src.stages.run_stage_7_evaluation --generation-model openai/gpt-4o
    python -m src.stages.run_stage_7_evaluation --evaluation-model anthropic/claude-3-5-sonnet

    # Custom output path
    python -m src.stages.run_stage_7_evaluation -o data/evaluation/results/alpha_0.3.json

    # COMPREHENSIVE MODE: Test all combinations (collections x alphas x strategies)
    python -m src.stages.run_stage_7_evaluation --comprehensive
    # Uses curated 10-question subset, tests all collections, alphas (0.0-1.0), strategies
    # Generates leaderboard report with metric breakdowns

Arguments:
    -n, --questions N         Limit to first N questions
    -m, --metrics METRICS     Metrics to compute (default: faithfulness relevancy context_precision context_recall)
    -k, --top-k K             Chunks to retrieve per question (default: 10)
    -a, --alpha ALPHA         Hybrid search balance: 0.0=keyword, 0.5=balanced, 1.0=vector
    --collection NAME         Weaviate collection to evaluate (default: from config)
    --reranking/--no-reranking  Enable/disable cross-encoder reranking (default: disabled)
    --generation-model MODEL  Answer generation model (default: openai/gpt-5-mini)
    --evaluation-model MODEL  RAGAS judge model (default: anthropic/claude-haiku-4.5)
    -o, --output PATH         Output JSON file path (default: results/eval_TIMESTAMP.json)
    --no-log                  Skip auto-logging to evaluation-history.md
    --comprehensive           Run grid search across all collections, alphas, strategies

Output:
    1. JSON report: data/evaluation/results/eval_TIMESTAMP.json
    2. Markdown log: memory-bank/evaluation-history.md (auto-appended)
    3. Config tracking: data/evaluation/tracking.json (auto-updated)

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
    - OpenRouter API key must be set in .env

See Also:
    - memory-bank/evaluation-history.md - Historical run comparisons
    - memory-bank/rag-improvement-plan.md - Improvement roadmap
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import (
    DEFAULT_TOP_K,
    EVAL_GENERATION_MODEL,
    EVAL_EVALUATION_MODEL,
    EVAL_TEST_QUESTIONS_FILE,
    EVAL_RESULTS_DIR,
    EVAL_LOGS_DIR,
    EVAL_DEFAULT_METRICS,
    PROJECT_ROOT,
    MAX_CHUNK_TOKENS,
    OVERLAP_SENTENCES,
    EMBEDDING_MODEL,
    get_collection_name,
)
from src.evaluation import run_evaluation
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.rag_pipeline.retrieval.preprocessing.strategy_config import (
    get_strategy_config,
    is_valid_combination,
    list_strategy_configs,
    EVAL_TOP_K,
    get_all_collections,
)

# Comprehensive evaluation imports (lazy-loaded in function)
COMPREHENSIVE_QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "comprehensive_questions.json"

logger = setup_logging(__name__)


def setup_file_logging(timestamp: str) -> Path:
    """Set up file logging for comprehensive evaluation.

    Creates a log file that captures all logger output for later review.
    The file handler is added to the root logger so all modules' logs are captured.

    Args:
        timestamp: Timestamp string for the log filename.

    Returns:
        Path to the log file.
    """
    import logging

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


# Type alias for retrieval cache (used in comprehensive mode)
# Cache key: (question_id, collection, search_type, alpha, strategy) - top_k is NOT in key
RetrievalCacheKey = tuple[str, str, str, float, str]
RetrievalCache = dict[RetrievalCacheKey, list[str]]  # Maps cache key -> context strings


# ============================================================================
# PATHS (from config)
# ============================================================================

DEFAULT_QUESTIONS_FILE = COMPREHENSIVE_QUESTIONS_FILE  # 15 curated questions
FULL_QUESTIONS_FILE = EVAL_TEST_QUESTIONS_FILE  # All 45 questions
RESULTS_DIR = EVAL_RESULTS_DIR
EVALUATION_HISTORY_FILE = PROJECT_ROOT / "memory-bank" / "evaluation-history.md"


# ============================================================================
# LOADER
# ============================================================================


def resolve_questions_file(questions_file_arg: Optional[str]) -> Path:
    """Resolve questions file path from CLI argument.

    Args:
        questions_file_arg: CLI argument value (None, 'full', or path).

    Returns:
        Resolved path to questions file.
    """
    if questions_file_arg is None:
        return DEFAULT_QUESTIONS_FILE
    if questions_file_arg.lower() == "full":
        return FULL_QUESTIONS_FILE
    return Path(questions_file_arg)


def load_test_questions(
    filepath: Path = DEFAULT_QUESTIONS_FILE,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Load test questions from JSON file.

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

    with open(filepath, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Limit number of questions
    if limit and limit < len(questions):
        questions = questions[:limit]
        logger.info(f"Limited to first {limit} questions")

    return questions


# ============================================================================
# AUTO-LOGGING
# ============================================================================


def get_next_run_number() -> int:
    """
    Get the next run number from evaluation-history.md.

    Parses the markdown file to find the highest existing run number
    and returns the next sequential number.

    Returns:
        Next run number (starts at 1 if no runs exist).
    """
    if not EVALUATION_HISTORY_FILE.exists():
        return 1

    content = EVALUATION_HISTORY_FILE.read_text()
    # Match patterns like "## Run 4:" or "## Run 10:"
    matches = re.findall(r"## Run (\d+):", content)

    if not matches:
        return 1

    max_run = max(int(m) for m in matches)
    return max_run + 1


def append_to_evaluation_history(
    results: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    questions: list[dict[str, Any]],
) -> None:
    """
    Append evaluation run summary to memory-bank/evaluation-history.md.

    Auto-generates markdown entry with:
    - Run number (auto-incremented)
    - Configuration details
    - Aggregate scores
    - Category breakdown (if questions have categories)

    Args:
        results: RAGAS evaluation results with 'scores' key.
        config: Run configuration dict with collection, alpha, models, etc.
        output_path: Path where JSON results were saved.
        questions: List of test questions (for category breakdown).
    """
    run_number = get_next_run_number()

    # Calculate category breakdown
    category_breakdown = {}
    df = results.get("results")
    if df is not None:
        for i, q in enumerate(questions):
            category = q.get("category", "unknown")
            if category not in category_breakdown:
                category_breakdown[category] = {"count": 0, "relevancy_sum": 0, "faithfulness_sum": 0}
            category_breakdown[category]["count"] += 1
            # Try to get scores from DataFrame
            if i < len(df):
                relevancy = df.iloc[i].get("answer_relevancy", 0) or 0
                faithfulness = df.iloc[i].get("faithfulness", 0) or 0
                category_breakdown[category]["relevancy_sum"] += relevancy
                category_breakdown[category]["faithfulness_sum"] += faithfulness

    # Format category table
    category_table = ""
    if category_breakdown:
        category_table = "\n### Category Breakdown\n| Category | Relevancy | Faithfulness |\n|----------|-----------|--------------|\n"
        for cat, data in category_breakdown.items():
            avg_rel = data["relevancy_sum"] / data["count"] if data["count"] > 0 else 0
            avg_faith = data["faithfulness_sum"] / data["count"] if data["count"] > 0 else 0
            category_table += f"| {cat.title()} ({data['count']}) | {avg_rel:.2f} | {avg_faith:.2f} |\n"

    # Count failures (relevancy = 0)
    failures = 0
    if df is not None and "answer_relevancy" in df.columns:
        failures = len(df[df["answer_relevancy"] == 0])

    # Get relative path from project root
    try:
        relative_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = output_path

    # Format scores
    scores = results.get("scores", {})
    faithfulness = scores.get("faithfulness", "N/A")
    relevancy = scores.get("relevancy", "N/A")
    context_precision = scores.get("context_precision", "N/A")

    # Format score strings
    faith_str = f"{faithfulness:.3f}" if isinstance(faithfulness, (int, float)) else str(faithfulness)
    rel_str = f"{relevancy:.3f}" if isinstance(relevancy, (int, float)) else str(relevancy)
    cp_str = f"{context_precision:.3f}" if isinstance(context_precision, (int, float)) else str(context_precision)

    # Format preprocessing info
    prep_strategy = config.get('preprocessing_strategy', 'none')
    prep_model = config.get('preprocessing_model', 'default')
    prep_str = f"{prep_strategy}" + (f" ({prep_model})" if prep_model else "")

    entry = f"""
---

## Run {run_number}: {config.get('collection', 'Unknown')}

**Date:** {datetime.now().strftime('%B %d, %Y')}
**File:** `{relative_path}`

### Configuration
- **Collection:** {config.get('collection', 'auto')}
- **Search Type:** Hybrid
- **Alpha:** {config.get('alpha', 0.5)}
- **Top-K:** {config.get('top_k', 10)}
- **Reranking:** {'Yes' if config.get('reranking', False) else 'No'}
- **Preprocessing:** {prep_str}
- **Generation Model:** {config.get('generation_model', 'unknown')}
- **Evaluation Model:** {config.get('evaluation_model', 'unknown')}

### Results
| Metric | Score |
|--------|-------|
| Faithfulness | {faith_str} |
| Relevancy | {rel_str} |
| Context Precision | {cp_str} |
| Failures | {failures}/{len(questions)} ({100*failures/len(questions):.0f}%) |
{category_table}
### Key Learning
[Add notes about this run manually]

"""

    with open(EVALUATION_HISTORY_FILE, "a") as f:
        f.write(entry)

    logger.info(f"Appended to {EVALUATION_HISTORY_FILE} as Run {run_number}")


# ============================================================================
# REPORT
# ============================================================================


def generate_report(
    results: dict[str, Any],
    questions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate evaluation report as JSON and summary.

    Args:
        results: RAGAS evaluation results (includes difficulty_breakdown).
        questions: Original test questions.
        output_path: Path for output JSON file.
    """
    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "aggregate_scores": results["scores"],
        "difficulty_breakdown": results.get("difficulty_breakdown", {}),
        "trace_path": str(results.get("trace_path", "")) if results.get("trace_path") else None,
        "per_question_results": [],
    }

    # Add per-question details
    df = results["results"]
    for i, q in enumerate(questions):
        question_result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "difficulty": q["difficulty"],
        }

        # Add metric scores from DataFrame
        for col in df.columns:
            if col not in ["user_input", "retrieved_contexts", "response", "reference"]:
                question_result[col] = float(df.iloc[i][col]) if i < len(df) else None

        report["per_question_results"].append(question_result)

    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nQuestions evaluated: {len(questions)}")
    print(f"\nAggregate Scores:")
    for metric, score in results["scores"].items():
        print(f"  {metric}: {score:.4f}")

    # Print difficulty breakdown
    if results.get("difficulty_breakdown"):
        print("\nScores by Question Difficulty:")
        for difficulty, metrics in results["difficulty_breakdown"].items():
            print(f"\n  {difficulty}:")
            for metric, score in metrics.items():
                print(f"    {metric}: {score:.4f}")

    print("\nPer-Question Results:")
    for qr in report["per_question_results"]:
        print(f"\n  [{qr['id']}] {qr['question'][:50]}...")
        for key, val in qr.items():
            if key not in ["id", "question", "category", "difficulty"] and val is not None:
                print(f"    {key}: {val:.4f}")

    print("\n" + "=" * 60)


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================


def run_comprehensive_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive evaluation across all VALID combinations with failure tracking.

    Tests: collections × alphas × reranking × preprocessing_strategies
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
    import time
    from src.ui.services.search import list_collections, extract_strategy_from_collection
    from src.rag_pipeline.retrieval.preprocessing.strategies import list_strategies
    from src.evaluation.schemas import FailedCombinationsReport

    # Generate timestamp early for consistent naming across log, checkpoint, and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up file logging (captures all output to log file)
    log_path = setup_file_logging(timestamp)

    logger.info("Starting comprehensive evaluation mode...")
    start_time = time.time()

    # Load questions (respect --questions-file if provided)
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
    top_k = EVAL_TOP_K  # Fixed at 15
    logger.info(f"Fixed parameters: top_k={top_k}")

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

    # Build valid combinations using StrategyConfig constraints
    # Tuple: (collection, alpha, reranking, strategy)
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
                    # Validate via StrategyConfig
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

        # Check if dedicated collection exists in available collections
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

    total_combinations = len(combinations)
    all_strategies = list_strategies()

    # Log what we're testing
    logger.info(f"Testing {total_combinations} valid combinations:")
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

    # Run all valid combinations
    all_results = []
    count = 0

    # Define checkpoint and failure tracking paths
    comprehensive_run_id = f"comprehensive_{timestamp}"
    checkpoint_path = RESULTS_DIR / f"comprehensive_checkpoint_{timestamp}.json"

    # Initialize failed combinations report
    failed_report = FailedCombinationsReport(
        comprehensive_run_id=comprehensive_run_id,
        timestamp=datetime.now().isoformat(),
    )

    for collection, alpha, use_reranking, strategy in combinations:
        count += 1
        rerank_str = "ON" if use_reranking else "off"
        logger.info(
            f"\n[{count}/{total_combinations}] "
            f"{collection} | alpha={alpha} | rerank={rerank_str} | strategy={strategy}"
        )

        try:
            results = run_evaluation(
                test_questions=questions,
                metrics=EVAL_DEFAULT_METRICS,
                top_k=top_k,
                generation_model=args.generation_model,
                evaluation_model=args.evaluation_model,
                collection_name=collection,
                use_reranking=use_reranking,
                alpha=alpha,
                preprocessing_strategy=strategy,
                preprocessing_model=None,
                save_trace=True,
                search_type="hybrid",  # Alpha controls balance
            )

            # Store configuration + scores + difficulty breakdown + trace path
            trace_path = results.get("trace_path")
            questions_processed = results.get("questions_processed", len(questions))
            questions_failed = results.get("questions_failed", 0)
            failed_questions = results.get("failed_questions", [])

            all_results.append({
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
            })

            scores = results["scores"]
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

            # Determine failure stage from error type
            error_type = type(e).__name__
            if "preprocessing" in str(e).lower():
                failed_at_stage = "preprocessing"
            elif "retrieval" in str(e).lower() or "weaviate" in str(e).lower():
                failed_at_stage = "retrieval"
            elif "generation" in str(e).lower():
                failed_at_stage = "generation"
            else:
                failed_at_stage = "ragas_evaluation"

            # Add to failed report
            failed_report.add_failure(
                collection=collection,
                alpha=alpha,
                top_k=top_k,
                strategy=strategy,
                error=e,
                failed_at_stage=failed_at_stage,
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
                "error_type": error_type,
            })

        # Save checkpoint after every combination for crash resilience
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "completed": count,
            "total": total_combinations,
            "progress_pct": round(count / total_combinations * 100, 1),
            "results": all_results,
            "grid_params": {
                "collections": available_collections,
                "top_k": top_k,
                "strategies": all_strategies,
            },
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"  Checkpoint saved: {count}/{total_combinations} ({checkpoint_data['progress_pct']}%)")

    # Calculate total duration
    duration_seconds = time.time() - start_time

    # Generate comprehensive report
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"comprehensive_{timestamp}.json"

    # Build grid parameters for report
    grid_params = {
        "collections": available_collections,
        "top_k": top_k,
        "strategies": all_strategies,
        "valid_combinations_count": total_combinations,
    }

    generate_comprehensive_report(
        all_results, questions, output_path, duration_seconds, grid_params, log_path
    )

    # Save failed combinations report if there were any failures
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
    import time
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
        logger.info(
            f"\n[{i + 1}/{failed_report.total_failed}] Retrying: "
            f"{fc.collection} | alpha={fc.alpha} | top_k={fc.top_k} | strategy={fc.strategy}"
        )

        try:
            # Get reranking from StrategyConfig default
            strategy_config = get_strategy_config(fc.strategy)
            use_reranking = strategy_config.get_default_reranking()

            results = run_evaluation(
                test_questions=questions,
                metrics=EVAL_DEFAULT_METRICS,
                top_k=fc.top_k,
                generation_model=args.generation_model,
                evaluation_model=args.evaluation_model,
                collection_name=fc.collection,
                use_reranking=use_reranking,
                alpha=fc.alpha,
                preprocessing_strategy=fc.strategy,
                preprocessing_model=None,
                save_trace=True,  # Save traces for retried runs (debugging)
            )

            retry_results.append({
                "collection": fc.collection,
                "alpha": fc.alpha,
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
            )

            retry_results.append({
                "collection": fc.collection,
                "alpha": fc.alpha,
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
    with open(output_path, "w") as f:
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
) -> dict[str, dict[str, float]]:
    """Compute statistical analysis (mean, std, min, max) for each group.

    Args:
        results: List of result dicts (only successful runs).
        group_key: Key to group by ("strategy", "alpha", "top_k", "collection").

    Returns:
        Dict mapping group values to stats (mean, std, min, max, n).
    """
    import statistics
    from collections import defaultdict

    groups: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            groups[str(r[group_key])].append(r["scores"].get("faithfulness") or 0)

    analysis = {}
    for group, scores in groups.items():
        analysis[group] = {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "n": len(scores),
        }
    return analysis


def find_best_configurations(
    sorted_results: list[dict[str, Any]],
    successful_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find best configurations per metric.

    Args:
        sorted_results: Results sorted by faithfulness.
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
    best_configs = find_best_configurations(sorted_results, successful_runs)

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
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comprehensive report saved to: {output_path}")

    # =========================================================================
    # CONSOLE OUTPUT: Leaderboard
    # =========================================================================
    print("\n" + "=" * 100)
    print("COMPREHENSIVE EVALUATION LEADERBOARD")
    print("=" * 100)
    print(f"\nTested {len(all_results)} combinations on {len(questions)} questions")
    print(f"Duration: {duration_str}")
    print(f"Successful: {len(successful_runs)} | Failed: {len(failed_runs)}")
    print("\n" + "-" * 118)
    print(f"{'Rank':<5} {'Collection':<35} {'Alpha':<7} {'Rerank':<8} {'Strategy':<15} {'Faith':<8} {'Relev':<8} {'CtxPrec':<8} {'CtxRec':<8}")
    print("-" * 118)

    for i, result in enumerate(sorted_results, 1):
        scores = result["scores"]
        faith = scores.get("faithfulness") or 0
        relev = scores.get("relevancy") or 0
        ctx_prec = scores.get("context_precision") or 0
        ctx_rec = scores.get("context_recall") or 0
        collection_short = result["collection"][:35]
        rerank_str = "ON" if result.get("reranking") else "off"
        error_marker = " *" if "error" in result else ""

        print(
            f"{i:<5} {collection_short:<35} {result['alpha']:<7} "
            f"{rerank_str:<8} {result['strategy']:<15} {faith:<8.3f} {relev:<8.3f} "
            f"{ctx_prec:<8.3f} {ctx_rec:<8.3f}{error_marker}"
        )

    # =========================================================================
    # CONSOLE OUTPUT: Statistical Breakdowns
    # =========================================================================
    def _print_breakdown(title: str, analysis: dict[str, dict[str, float]], format_label=str):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        for group in sorted(analysis.keys(), key=lambda x: analysis[x]["mean"], reverse=True):
            stats = analysis[group]
            label = format_label(group)
            print(f"\n{label} (n={stats['n']}):")
            print(f"  Mean:  {stats['mean']:.3f}  +/-  {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

    _print_breakdown("BREAKDOWN BY PREPROCESSING STRATEGY", strategy_analysis, str.upper)
    _print_breakdown("BREAKDOWN BY ALPHA (0.0=keyword, 1.0=vector)", alpha_analysis, lambda a: f"ALPHA={a}")
    _print_breakdown("BREAKDOWN BY RERANKING", reranking_analysis, lambda r: f"RERANK={'ON' if r == 'True' else 'off'}")
    _print_breakdown("BREAKDOWN BY COLLECTION", collection_analysis)

    # =========================================================================
    # CONSOLE OUTPUT: Article Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("ARTICLE SUMMARY")
    print("=" * 100)

    print(f"\nEXPERIMENT METADATA")
    print(f"  Duration: {duration_str}")
    print(f"  Combinations tested: {len(all_results)} ({len(successful_runs)} successful, {len(failed_runs)} failed)")
    print(f"  Questions per run: {len(questions)}")

    print(f"\nBEST CONFIGURATIONS")
    for metric in EVAL_DEFAULT_METRICS:
        key = f"by_{metric}"
        if best_configs.get(key):
            b = best_configs[key]
            print(f"  {metric.title():<15} {b['collection'][:25]} + alpha={b['alpha']} + top_k={b['top_k']} + {b['strategy']} ({b['score']:.3f})")

    # Calculate improvement percentages vs baseline (none strategy)
    if "none" in strategy_analysis:
        baseline = strategy_analysis["none"]["mean"]
        print(f"\nSTRATEGY IMPROVEMENT VS BASELINE (none={baseline:.3f})")
        for strategy in sorted(strategy_analysis.keys()):
            stats = strategy_analysis[strategy]
            if strategy == "none":
                print(f"  {strategy.upper():<15} {stats['mean']:.3f} +/- {stats['std']:.3f}  (baseline)")
            else:
                improvement = ((stats["mean"] - baseline) / baseline) * 100 if baseline > 0 else 0
                print(f"  {strategy.upper():<15} {stats['mean']:.3f} +/- {stats['std']:.3f}  [{improvement:+.1f}% vs baseline]")

    # Show failed runs if any
    if failed_runs:
        print("\n" + "=" * 100)
        print("FAILED RUNS")
        print("=" * 100)
        for result in failed_runs:
            print(f"  {result['collection']} | alpha={result['alpha']} | top_k={result['top_k']} | {result['strategy']}")
            print(f"    Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 100)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run RAGAS evaluation on test questions."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on test questions"
    )
    parser.add_argument(
        "--questions",
        "-n",
        type=int,
        default=None,
        help="Limit to first N questions",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=EVAL_DEFAULT_METRICS,
        help=f"Metrics to compute (default: {' '.join(EVAL_DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=EVAL_GENERATION_MODEL,
        help=f"OpenRouter model for answer generation (default: {EVAL_GENERATION_MODEL})",
    )
    parser.add_argument(
        "--evaluation-model",
        type=str,
        default=EVAL_EVALUATION_MODEL,
        help=f"OpenRouter model for RAGAS evaluation (default: {EVAL_EVALUATION_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: results/eval_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--search-type",
        "-s",
        type=str,
        choices=["keyword", "hybrid"],
        default="hybrid",
        help="Search type: keyword (BM25 only) or hybrid (vector+BM25, default)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.5,
        help="Hybrid search alpha: 0.5=balanced, 1.0=vector (default: 0.5). Only used with --search-type=hybrid.",
    )
    parser.add_argument(
        "--reranking",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking (disabled by default for speed)",
    )
    parser.add_argument(
        "--no-reranking",
        dest="reranking",
        action="store_false",
        help="Disable cross-encoder reranking (this is the default)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Weaviate collection to evaluate (default: auto from config)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        default=False,
        help="Skip auto-logging to evaluation-history.md and tracking.json",
    )
    parser.add_argument(
        "--preprocessing",
        "-p",
        type=str,
        choices=["none", "hyde", "decomposition", "graphrag"],
        default="none",
        help="Query preprocessing strategy (default: none for clean baseline)",
    )
    parser.add_argument(
        "--preprocessing-model",
        type=str,
        default=None,
        help="Model for preprocessing (default: from config)",
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite behavior for custom -o path: prompt (default), skip, all",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        default=False,
        help="Run comprehensive evaluation across all collections, alphas (0.0-1.0), and strategies",
    )
    parser.add_argument(
        "--retry-failed",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Re-run failed combinations from a previous comprehensive run (e.g., comprehensive_20251231_120000)",
    )
    parser.add_argument(
        "--questions-file",
        "-q",
        type=str,
        default=None,
        help="Questions file: path or 'full' for all 45 questions (default: comprehensive 15)",
    )

    args = parser.parse_args()

    # Retry failed mode: re-run specific failed combinations
    if args.retry_failed:
        retry_failed_combinations(args.retry_failed, args)
        return

    # Comprehensive mode: different execution path
    if args.comprehensive:
        run_comprehensive_evaluation(args)
        return

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    # Check overwrite for custom output path before doing expensive work
    if args.output:
        output_path = Path(args.output)
        if output_path.exists():
            if not overwrite_context.should_overwrite(output_path, logger):
                logger.info("Skipping evaluation (output file exists)")
                return

    # Load test questions
    questions_filepath = resolve_questions_file(args.questions_file)
    logger.info(f"Loading test questions from {questions_filepath.name}...")
    questions = load_test_questions(
        filepath=questions_filepath,
        limit=args.questions,
    )

    if not questions:
        logger.error("No test questions found")
        return

    logger.info(f"Loaded {len(questions)} test questions")

    # Determine collection name
    collection_name = args.collection or get_collection_name()

    # Validate reranking against strategy constraints
    strategy_config = get_strategy_config(args.preprocessing)
    use_reranking = args.reranking

    if not strategy_config.is_valid_reranking(use_reranking):
        rerank_mode = strategy_config.reranking_constraint.mode
        if rerank_mode == "required":
            logger.warning(
                f"Strategy '{args.preprocessing}' requires reranking (per paper). "
                "Enabling reranking automatically."
            )
            use_reranking = True
        elif rerank_mode == "forbidden":
            logger.warning(
                f"Strategy '{args.preprocessing}' forbids reranking. "
                "Disabling reranking automatically."
            )
            use_reranking = False

    # Run evaluation
    logger.info("Starting RAGAS evaluation...")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Search type: {args.search_type}")
    alpha_display = f"{args.alpha}" if args.search_type == "hybrid" else "N/A (keyword)"
    logger.info(f"Alpha: {alpha_display}")
    logger.info(f"Reranking: {use_reranking}")
    logger.info(f"Preprocessing: {args.preprocessing}")
    logger.info(f"Generation model: {args.generation_model}")
    logger.info(f"Evaluation model: {args.evaluation_model}")

    try:
        results = run_evaluation(
            test_questions=questions,
            metrics=args.metrics,
            top_k=args.top_k,
            generation_model=args.generation_model,
            evaluation_model=args.evaluation_model,
            collection_name=collection_name,
            use_reranking=use_reranking,
            alpha=args.alpha,
            preprocessing_strategy=args.preprocessing,
            preprocessing_model=args.preprocessing_model,
            search_type=args.search_type,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{timestamp}.json"

    generate_report(results, questions, output_path)

    # Auto-log results to evaluation history and tracking JSON
    if not args.no_log:
        config = {
            "collection": collection_name,
            "search_type": args.search_type,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "reranking": args.reranking,
            "preprocessing_strategy": args.preprocessing,
            "preprocessing_model": args.preprocessing_model,
            "generation_model": args.generation_model,
            "evaluation_model": args.evaluation_model,
        }
        append_to_evaluation_history(results, config, output_path, questions)
    else:
        logger.info("Skipping auto-logging (--no-log specified)")

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
