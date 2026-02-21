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
    EVAL_DEFAULT_METRICS,
    PROJECT_ROOT,
    MAX_CHUNK_TOKENS,
    OVERLAP_SENTENCES,
    EMBEDDING_MODEL,
    get_collection_name,
)
from src.evaluation import run_evaluation
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.rag_pipeline.retrieval.query_strategy_config import get_strategy_config

from src.evaluation.comprehensive import (
    run_comprehensive_evaluation,
    retry_failed_combinations,
    resolve_questions_file,
    load_test_questions,
    print_combination_table,
    COMPREHENSIVE_QUESTIONS_FILE,
)

logger = setup_logging(__name__)


# ============================================================================
# PATHS (from config)
# ============================================================================

DEFAULT_QUESTIONS_FILE = COMPREHENSIVE_QUESTIONS_FILE  # 15 curated questions
FULL_QUESTIONS_FILE = EVAL_TEST_QUESTIONS_FILE  # All 45 questions
RESULTS_DIR = EVAL_RESULTS_DIR
EVALUATION_HISTORY_FILE = PROJECT_ROOT / "memory-bank" / "evaluation-history.md"


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

    with open(EVALUATION_HISTORY_FILE, "a", encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {output_path}")

    # Log summary
    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Questions evaluated: {len(questions)}")
    logger.info("Aggregate Scores:")
    for metric, score in results["scores"].items():
        logger.info(f"  {metric}: {score:.4f}")

    # Log difficulty breakdown
    if results.get("difficulty_breakdown"):
        logger.info("Scores by Question Difficulty:")
        for difficulty, metrics in results["difficulty_breakdown"].items():
            logger.info(f"  {difficulty}:")
            for metric, score in metrics.items():
                logger.info(f"    {metric}: {score:.4f}")

    logger.info("Per-Question Results:")
    for qr in report["per_question_results"]:
        logger.info(f"  [{qr['id']}] {qr['question'][:50]}...")
        for key, val in qr.items():
            if key not in ["id", "question", "category", "difficulty"] and val is not None:
                logger.info(f"    {key}: {val:.4f}")

    logger.info("=" * 60)


# NOTE: Comprehensive evaluation functions moved to src/evaluation/comprehensive.py
# Imported at module level: run_comprehensive_evaluation, retry_failed_combinations

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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Resume comprehensive evaluation from checkpoint file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print all valid combinations without running evaluation",
    )

    args = parser.parse_args()

    # Retry failed mode: re-run specific failed combinations
    if args.retry_failed:
        retry_failed_combinations(args.retry_failed, args)
        return

    # Dry-run mode: print combination table and exit
    if args.dry_run:
        print_combination_table()
        return

    # Comprehensive mode: different execution path (supports --resume)
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
            "reranking": use_reranking,
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
