"""RAGAS evaluation for RAGLab.

Provides:
- OpenRouter chat integration for answer generation
- RAGAS evaluation with LangChain wrapper
- Retrieval and generation functions for the evaluation pipeline
- Strategy-aware retrieval (decomposition union merge, graphrag hybrid)
"""

import time
import math
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

from src.rag_pipeline.retrieval.query_preprocessing import PreprocessedQuery

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL_ID,
    get_collection_name,
    DEFAULT_TOP_K,
    EVAL_TRACES_DIR,
)
from src.rag_pipeline.indexing import get_client
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_simple_prompt
from src.evaluation.schemas import QuestionTrace, EvaluationTrace

logger = setup_logging(__name__)


# Default model for answer generation (can be overridden)
DEFAULT_CHAT_MODEL = "anthropic/claude-haiku-4.5"

# Mapping from our metric names to RAGAS DataFrame column names
RAGAS_METRIC_COLUMN_MAP = {
    "faithfulness": "faithfulness",
    "relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_without_reference",
    "context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
}


def _is_valid_score(value: Any) -> bool:
    """Check if a value is a valid numeric score (not None, not NaN).

    Args:
        value: The value to check.

    Returns:
        True if the value is a valid numeric score.
    """
    if value is None:
        return False
    try:
        float_val = float(value)
        return not math.isnan(float_val)
    except (TypeError, ValueError):
        return False




# ============================================================================
# RAGAS LLM WRAPPER
# ============================================================================


def create_evaluator_llm(model: str = "anthropic/claude-haiku-4.5") -> LangchainLLMWrapper:
    """
    Create LLM wrapper for RAGAS evaluation via OpenRouter.

    Uses LangchainLLMWrapper (deprecated but functional). The newer llm_factory
    API is incompatible with RAGAS evaluate() - causes 'agenerate_prompt' errors.

    Args:
        model: OpenRouter model ID for evaluation.

    Returns:
        LangchainLLMWrapper configured for OpenRouter.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.0,
        max_tokens=4096,
        request_timeout=120,
    )
    return LangchainLLMWrapper(llm)


def create_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Create embeddings wrapper for RAGAS evaluation via OpenRouter.

    Uses LangchainEmbeddingsWrapper (deprecated but functional).

    Returns:
        LangchainEmbeddingsWrapper configured for OpenRouter.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_ID,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    return LangchainEmbeddingsWrapper(embeddings)


class RAGASEvaluationError(Exception):
    """Raised when RAGAS evaluation fails after all retries."""

    pass


def evaluate_with_retry(
    dataset: EvaluationDataset,
    metrics: list,
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    run_config: Optional[RunConfig] = None,
) -> Any:
    """Run RAGAS evaluation with exponential backoff retry.

    Wraps ragas.evaluate() to handle transient API failures.
    Uses the same retry pattern as openrouter_client.py.

    Args:
        dataset: RAGAS EvaluationDataset.
        metrics: List of RAGAS metric objects.
        llm: Wrapped LLM for evaluation.
        embeddings: Wrapped embeddings for evaluation.
        max_retries: Maximum retry attempts.
        backoff_base: Exponential backoff base.
        run_config: RAGAS RunConfig for controlling concurrency, timeouts, and retries.
                   If None, uses RAGAS defaults (max_workers=16).

    Returns:
        RAGAS evaluation results.

    Raises:
        RAGASEvaluationError: After all retries exhausted.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )
            return results

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if it's a retryable error (rate limit, server error, network)
            is_retryable = (
                "rate" in error_str
                or "limit" in error_str
                or "429" in error_str
                or "500" in error_str
                or "502" in error_str
                or "503" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )

            if is_retryable and attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"RAGAS evaluation failed ({type(e).__name__}), "
                    f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            # Non-retryable error or max retries exceeded
            break

    raise RAGASEvaluationError(
        f"RAGAS evaluation failed after {max_retries} retries: {last_error}"
    ) from last_error


def compute_difficulty_breakdown(
    results_df: Any,
    test_questions: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-difficulty metric averages.

    Groups questions by difficulty field ("single_concept" or "cross_domain")
    and computes average scores for each metric within each group.

    Args:
        results_df: RAGAS results DataFrame with metric columns.
        test_questions: Original test questions with difficulty field.
        metrics: List of metric names to include in breakdown.

    Returns:
        Dict mapping difficulty -> {metric: avg_score}.
        Example: {
            "single_concept": {"faithfulness": 0.95, "relevancy": 0.88},
            "cross_domain": {"faithfulness": 0.82, "relevancy": 0.75}
        }
    """
    breakdown: dict[str, dict[str, list[float]]] = {}

    for i, q in enumerate(test_questions):
        if i >= len(results_df):
            continue

        difficulty = q.get("difficulty", "unknown")
        if difficulty not in breakdown:
            breakdown[difficulty] = {m: [] for m in metrics}

        for metric in metrics:
            col_name = RAGAS_METRIC_COLUMN_MAP.get(metric, metric)
            if col_name in results_df.columns:
                value = results_df.iloc[i].get(col_name)
                if _is_valid_score(value):
                    breakdown[difficulty][metric].append(float(value))

    # Compute averages
    result = {}
    for difficulty, metric_scores in breakdown.items():
        result[difficulty] = {}
        for metric, scores in metric_scores.items():
            if scores:
                result[difficulty][metric] = round(sum(scores) / len(scores), 4)
            else:
                result[difficulty][metric] = 0.0

    return result


# ============================================================================
# RETRIEVAL AND GENERATION
# ============================================================================


def retrieve_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessed: Optional[PreprocessedQuery] = None,
    search_type: str = "hybrid",
) -> list[str]:
    """
    Retrieve relevant contexts from Weaviate using strategy-based retrieval.

    This function uses the RetrievalStrategy pattern to handle different
    preprocessing strategies, eliminating complex conditional logic.

    Args:
        question: The user's question.
        top_k: Number of chunks to return after reranking.
        collection_name: Override collection name.
        use_reranking: If True, apply cross-encoder reranking.
        alpha: Hybrid search balance (0.5=balanced, 1.0=vector).
        preprocessed: PreprocessedQuery from strategy (used to determine strategy_id).
        search_type: "keyword" for pure BM25, "hybrid" for vector+BM25 (default).

    Returns:
        List of context strings from retrieved chunks.

    Technical Notes:
        - Strategies encapsulate their own retrieval logic:
          - StandardRetrieval: Direct search
          - HyDERetrieval: Embedding averaging
          - DecompositionRetrieval: Multi-query union merge + rerank
          - GraphRAGRetrieval: Graph + vector hybrid
        - Cross-encoder reranking improves precision by 20-35%
    """
    from src.rag_pipeline.retrieval.strategy_registry import RetrievalContext, get_strategy

    collection_name = collection_name or get_collection_name()

    # Determine strategy from preprocessed query
    strategy_id = "none"
    if preprocessed:
        strategy_id = preprocessed.strategy_used

    logger.info(f"  [retrieve_contexts] strategy={strategy_id}, search_type={search_type}")

    # Get Neo4j driver if needed for GraphRAG
    neo4j_driver = None
    if strategy_id == "graphrag":
        try:
            from src.graph.neo4j_client import get_driver
            neo4j_driver = get_driver()
        except Exception as e:
            logger.warning(f"  [retrieve_contexts] Neo4j driver unavailable: {e}")

    # Build retrieval context
    client = get_client()
    initial_k = 50 if use_reranking else top_k

    context = RetrievalContext(
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        use_reranking=use_reranking,
        initial_k=initial_k,
        alpha=alpha,
        search_type=search_type,
        neo4j_driver=neo4j_driver,
    )

    try:
        # Get strategy instance and execute (polymorphic dispatch)
        retrieval_strategy = get_strategy(strategy_id)
        result = retrieval_strategy.execute(question, context)

        # Return just the text for evaluation
        return [r.text for r in result.results]

    finally:
        client.close()
        if neo4j_driver:
            neo4j_driver.close()


def retrieve_contexts_with_metadata(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessed: Optional[PreprocessedQuery] = None,
    search_type: str = "hybrid",
) -> tuple[list[str], dict]:
    """Retrieve contexts and strategy metadata (supports DRIFT answer detection).

    Same as retrieve_contexts() but also returns RetrievalResult.metadata,
    which may contain drift_final_answer for GraphRAG global queries.

    Args:
        question: The user's question.
        top_k: Number of chunks to return.
        collection_name: Override collection name.
        use_reranking: If True, apply cross-encoder reranking.
        alpha: Hybrid search balance.
        preprocessed: PreprocessedQuery from strategy.
        search_type: "keyword" or "hybrid".

    Returns:
        Tuple of (context_texts, metadata_dict).
    """
    from src.rag_pipeline.retrieval.strategy_registry import RetrievalContext, get_strategy

    collection_name = collection_name or get_collection_name()

    strategy_id = "none"
    if preprocessed:
        strategy_id = preprocessed.strategy_used

    logger.info(f"  [retrieve_contexts_with_metadata] strategy={strategy_id}, search_type={search_type}")

    neo4j_driver = None
    if strategy_id == "graphrag":
        try:
            from src.graph.neo4j_client import get_driver
            neo4j_driver = get_driver()
        except Exception as e:
            logger.warning(f"  [retrieve_contexts_with_metadata] Neo4j driver unavailable: {e}")

    client = get_client()
    initial_k = 50 if use_reranking else top_k

    context = RetrievalContext(
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        use_reranking=use_reranking,
        initial_k=initial_k,
        alpha=alpha,
        search_type=search_type,
        neo4j_driver=neo4j_driver,
    )

    try:
        retrieval_strategy = get_strategy(strategy_id)
        result = retrieval_strategy.execute(question, context)
        return [r.text for r in result.results], result.metadata
    finally:
        client.close()
        if neo4j_driver:
            neo4j_driver.close()


def generate_answer(
    question: str,
    contexts: list[str],
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Generate an answer using retrieved contexts.

    Args:
        question: The user's question.
        contexts: List of context strings from retrieval.
        model: OpenRouter model ID for generation.

    Returns:
        Generated answer string.
    """
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""Based on the following context, answer the question.
Only use information from the context. If the context doesn't contain
enough information to fully answer the question, say so explicitly.

Context:
{context_text}

Question: {question}

Answer:"""

    return call_simple_prompt(prompt, model=model)


# ============================================================================
# RETRIEVAL CACHING (for comprehensive mode top_k optimization)
# ============================================================================


def retrieve_contexts_with_cache(
    question: str,
    top_k: int,
    retrieval_k: int,
    collection_name: str,
    use_reranking: bool,
    alpha: float,
    preprocessed: Optional[PreprocessedQuery],
    cache: Optional[dict] = None,
    cache_key: Optional[tuple] = None,
    search_type: str = "hybrid",
) -> list[str]:
    """Retrieve contexts with optional caching for comprehensive mode.

    When caching is enabled (cache and cache_key provided), this function:
    - On cache HIT: slices cached results to top_k and returns
    - On cache MISS: retrieves retrieval_k results, caches them, slices to top_k

    This optimization halves Weaviate calls when testing multiple top_k values,
    since we retrieve max(top_k) once and slice for smaller values.

    Args:
        question: The search query (may be preprocessed).
        top_k: Number of contexts to return.
        retrieval_k: Number to actually retrieve from Weaviate (>= top_k).
        collection_name: Weaviate collection name.
        use_reranking: Whether to apply cross-encoder reranking.
        alpha: Hybrid search balance.
        preprocessed: Optional PreprocessedQuery for strategy-aware retrieval.
        cache: Optional cache dict (keys are cache_key tuples).
        cache_key: Optional tuple (question_id, collection, search_type, alpha, strategy).
        search_type: "keyword" for BM25, "hybrid" for vector+BM25 (default).

    Returns:
        List of context text strings (length = top_k).
    """
    # Check cache first
    if cache is not None and cache_key is not None:
        if cache_key in cache:
            cached_contexts = cache[cache_key]
            sliced = cached_contexts[:top_k]
            logger.debug(f"  [cache] HIT for {cache_key[0]}, sliced {len(cached_contexts)} -> {len(sliced)}")
            return sliced

    # Cache miss - do full retrieval with retrieval_k
    full_contexts = retrieve_contexts(
        question=question,
        top_k=retrieval_k,  # Retrieve the larger count
        collection_name=collection_name,
        use_reranking=use_reranking,
        alpha=alpha,
        preprocessed=preprocessed,
        search_type=search_type,
    )

    # Store in cache before slicing
    if cache is not None and cache_key is not None:
        cache[cache_key] = full_contexts
        logger.debug(f"  [cache] STORED {len(full_contexts)} contexts for {cache_key[0]}")

    # Slice to top_k for return
    return full_contexts[:top_k]


def retrieve_contexts_with_cache_and_metadata(
    question: str,
    top_k: int,
    retrieval_k: int,
    collection_name: str,
    use_reranking: bool,
    alpha: float,
    preprocessed: Optional[PreprocessedQuery],
    cache: Optional[dict] = None,
    cache_key: Optional[tuple] = None,
    search_type: str = "hybrid",
) -> tuple[list[str], dict]:
    """Retrieve contexts with caching, returning metadata for DRIFT detection.

    Like retrieve_contexts_with_cache() but returns (contexts, metadata).
    On cache HIT, metadata is empty (DRIFT answers are not cached — GraphRAG
    always runs fresh since it's a single dedicated combination).

    Args:
        question: The search query.
        top_k: Number of contexts to return.
        retrieval_k: Number to retrieve from Weaviate.
        collection_name: Weaviate collection name.
        use_reranking: Whether to apply cross-encoder reranking.
        alpha: Hybrid search balance.
        preprocessed: Optional PreprocessedQuery for strategy-aware retrieval.
        cache: Optional cache dict.
        cache_key: Optional tuple for cache lookup.
        search_type: "keyword" or "hybrid".

    Returns:
        Tuple of (context_texts, metadata_dict).
    """
    # Check cache first (metadata not cached — only standard retrieval is cached)
    if cache is not None and cache_key is not None:
        if cache_key in cache:
            cached_contexts = cache[cache_key]
            sliced = cached_contexts[:top_k]
            logger.debug(f"  [cache] HIT for {cache_key[0]}, sliced {len(cached_contexts)} -> {len(sliced)}")
            return sliced, {}

    # Cache miss — do full retrieval with metadata
    full_contexts, metadata = retrieve_contexts_with_metadata(
        question=question,
        top_k=retrieval_k,
        collection_name=collection_name,
        use_reranking=use_reranking,
        alpha=alpha,
        preprocessed=preprocessed,
        search_type=search_type,
    )

    # Store in cache before slicing (only contexts, not metadata)
    if cache is not None and cache_key is not None:
        cache[cache_key] = full_contexts
        logger.debug(f"  [cache] STORED {len(full_contexts)} contexts for {cache_key[0]}")

    return full_contexts[:top_k], metadata


# ============================================================================
# PER-QUESTION PROCESSING WITH RETRY
# ============================================================================


@dataclass
class QuestionProcessingResult:
    """Result of processing a single question through the RAG pipeline.

    Attributes:
        success: Whether processing completed successfully.
        sample: RAGAS sample dict (user_input, retrieved_contexts, response, reference).
        trace: QuestionTrace for debugging and recalculation.
        error: Error message if processing failed.
        failed_at_stage: Which stage failed (preprocessing, retrieval, generation).
    """

    success: bool
    sample: Optional[dict[str, Any]] = None
    trace: Optional[QuestionTrace] = None
    error: Optional[str] = None
    failed_at_stage: Optional[str] = None


def process_single_question(
    question_data: dict[str, Any],
    question_index: int,
    preprocessing_strategy: str,
    preprocessing_model: Optional[str],
    top_k: int,
    resolved_collection: str,
    use_reranking: bool,
    alpha: float,
    generation_model: str,
    retrieval_cache: Optional[dict] = None,
    max_retrieval_k: Optional[int] = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    search_type: str = "hybrid",
) -> QuestionProcessingResult:
    """Process a single question through preprocessing, retrieval, and generation with retry.

    Implements per-question retry logic with exponential backoff for resilience
    against transient LLM failures.

    Args:
        question_data: Question dict with 'question', optional 'reference', 'id', etc.
        question_index: Index of question in the list (for default ID).
        preprocessing_strategy: Strategy to use (none, hyde, decomposition, graphrag).
        preprocessing_model: Model for preprocessing LLM calls.
        top_k: Number of chunks to retrieve.
        resolved_collection: Weaviate collection name.
        use_reranking: Whether to apply cross-encoder reranking.
        alpha: Hybrid search balance.
        generation_model: Model for answer generation.
        retrieval_cache: Optional cache dict for comprehensive mode.
        max_retrieval_k: Max retrieval count for caching.
        max_retries: Maximum retry attempts per stage.
        backoff_base: Exponential backoff base.
        search_type: "keyword" for BM25, "hybrid" for vector+BM25 (default).

    Returns:
        QuestionProcessingResult with success status, sample, trace, or error details.
    """
    question = question_data["question"]
    reference = question_data.get("reference")
    question_id = question_data.get("id", f"q_{question_index}")
    difficulty = question_data.get("difficulty", "unknown")
    category = question_data.get("category", "unknown")

    last_error = None
    failed_at_stage = None

    for attempt in range(max_retries + 1):
        try:
            # =====================================================================
            # STAGE 1: Preprocessing
            # =====================================================================
            failed_at_stage = "preprocessing"  # Set early for proper error tracking
            search_query = question
            preprocessed = None
            generated_queries = None

            if preprocessing_strategy != "none":
                from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query

                try:
                    preprocessed = preprocess_query(
                        query=question,
                        strategy=preprocessing_strategy,
                        model=preprocessing_model,
                    )
                    search_query = preprocessed.search_query
                    generated_queries = preprocessed.generated_queries
                    logger.info(f"  Preprocessed ({preprocessing_strategy}): {search_query[:50]}...")
                except Exception as e:
                    # Preprocessing failures are warnings, not fatal
                    logger.warning(f"  Preprocessing failed: {e}. Using original query.")
                    search_query = question
                    preprocessed = None

            # =====================================================================
            # STAGE 2: Retrieval
            # =====================================================================
            failed_at_stage = "retrieval"

            cache_key = None
            if retrieval_cache is not None:
                # Cache key includes search_type to differentiate keyword vs hybrid results
                cache_key = (question_id, resolved_collection, search_type, alpha, preprocessing_strategy)

            retrieval_k = max_retrieval_k if max_retrieval_k else top_k

            contexts, retrieval_meta = retrieve_contexts_with_cache_and_metadata(
                question=question,  # Always use original question (preprocessed has strategy data)
                top_k=top_k,
                retrieval_k=retrieval_k,
                collection_name=resolved_collection,
                use_reranking=use_reranking,
                alpha=alpha,
                preprocessed=preprocessed,
                cache=retrieval_cache,
                cache_key=cache_key,
                search_type=search_type,
            )
            logger.info(f"  Retrieved {len(contexts)} contexts")

            # =====================================================================
            # STAGE 3: Generation (or DRIFT answer passthrough)
            # =====================================================================
            failed_at_stage = "generation"

            drift_answer = retrieval_meta.get("drift_final_answer")
            if drift_answer:
                answer = drift_answer
                logger.info("  Using DRIFT synthesized answer (skipping generation)")
            else:
                answer = generate_answer(
                    question=question,
                    contexts=contexts,
                    model=generation_model,
                )
                logger.info(f"  Generated answer: {answer[:100]}...")

            # =====================================================================
            # BUILD SAMPLE AND TRACE
            # =====================================================================
            sample = {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
            }
            if reference:
                sample["reference"] = reference

            trace = QuestionTrace(
                question_id=question_id,
                question=question,
                difficulty=difficulty,
                category=category,
                reference=reference,
                preprocessing_strategy=preprocessing_strategy,
                search_query=search_query,
                generated_queries=generated_queries,
                retrieved_contexts=contexts,
                retrieval_metadata={
                    "top_k": top_k,
                    "alpha": alpha,
                    "collection": resolved_collection,
                    "reranking": use_reranking,
                    "search_type": search_type,
                    "graphrag_query_type": retrieval_meta.get("query_type"),
                    "preprocessing_strategy": preprocessing_strategy,
                },
                generated_answer=answer,
                generation_model=generation_model,
            )

            return QuestionProcessingResult(success=True, sample=sample, trace=trace)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is retryable
            is_retryable = (
                "rate" in error_str
                or "limit" in error_str
                or "429" in error_str
                or "500" in error_str
                or "502" in error_str
                or "503" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )

            if is_retryable and attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"  Question processing failed ({type(e).__name__}), "
                    f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            # Non-retryable or max retries exceeded
            break

    # All retries exhausted
    return QuestionProcessingResult(
        success=False,
        error=str(last_error),
        failed_at_stage=failed_at_stage,
    )


# ============================================================================
# RAGAS EVALUATION
# ============================================================================


def run_evaluation(
    test_questions: list[dict[str, Any]],
    metrics: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    generation_model: str = DEFAULT_CHAT_MODEL,
    evaluation_model: str = "anthropic/claude-haiku-4.5",
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessing_strategy: str = "none",
    preprocessing_model: Optional[str] = None,
    save_trace: bool = True,
    trace_path: Optional[Path] = None,
    ragas_max_retries: int = 3,
    ragas_backoff_base: float = 2.0,
    # Caching parameters for comprehensive mode (halves Weaviate calls for top_k dimension)
    retrieval_cache: Optional[dict] = None,
    max_retrieval_k: Optional[int] = None,
    # RAGAS concurrency control (prevents rate limiting)
    ragas_max_workers: int = 4,
    ragas_max_wait: int = 90,
    ragas_log_tenacity: bool = True,
    # Search type dimension (orthogonal to preprocessing)
    search_type: str = "hybrid",
) -> dict[str, Any]:
    """
    Run RAGAS evaluation on test questions with traceability and resilience.

    This function:
    1. Optionally preprocesses each question (hyde, decomposition, etc.)
    2. Retrieves contexts for each question (with optional cross-encoder reranking)
    3. Generates answers using the RAG pipeline
    4. Evaluates using RAGAS metrics with retry logic
    5. Builds and saves trace file for recalculation

    Args:
        test_questions: List of dicts with 'question' and optionally 'reference' keys.
        metrics: Which metrics to compute. Options:
            - "faithfulness": Is the answer grounded in context?
            - "relevancy": Does the answer address the question?
            - "context_precision": Are retrieved chunks relevant?
            - "context_recall": Did retrieval capture needed info? (requires reference)
            - "answer_correctness": Is the answer factually correct? (requires reference)
        top_k: Number of chunks to retrieve per question.
        generation_model: Model for answer generation.
        evaluation_model: Model for RAGAS evaluation.
        collection_name: Override Weaviate collection.
        use_reranking: If True, apply cross-encoder reranking to improve retrieval.
                       Default: True (enabled for best accuracy).
        alpha: Hybrid search balance (0.5=balanced, 1.0=vector). Only used for hybrid search_type.
        preprocessing_strategy: Query preprocessing strategy ("none", "hyde", "decomposition", "graphrag").
                               Default: "none" for clean baseline evaluation.
        preprocessing_model: Model for preprocessing LLM calls (default from config).
        save_trace: If True, save trace file with all interactions for recalculation.
        trace_path: Custom path for trace file. If None, auto-generates path.
        ragas_max_retries: Maximum retries for RAGAS evaluation API calls.
        ragas_backoff_base: Exponential backoff base for RAGAS retries.
        retrieval_cache: Optional cache dict for comprehensive mode. When provided,
                        retrieval results are cached by (question_id, collection, search_type, alpha, strategy).
                        Enables top_k slicing optimization (retrieve max_retrieval_k once, slice for smaller top_k).
        max_retrieval_k: When caching, retrieve this many results and cache them.
                        Smaller top_k values are sliced from the cached results.
        ragas_max_workers: Maximum concurrent API calls for RAGAS evaluation.
                          Default 4 (vs RAGAS default 16) to prevent rate limiting.
        ragas_max_wait: Maximum wait time (seconds) between RAGAS retries. Default 90.
        ragas_log_tenacity: If True, log RAGAS retry attempts for visibility.
        search_type: "keyword" for pure BM25, "hybrid" for vector+BM25 (default).

    Returns:
        Dict with:
            - "scores": Dict of metric_name -> average score
            - "results": Per-question results DataFrame
            - "samples": List of evaluation samples
            - "difficulty_breakdown": Per-difficulty group metric averages
            - "trace_path": Path where trace was saved (if save_trace=True)

    Raises:
        RAGASEvaluationError: If RAGAS evaluation fails after all retries.
    """
    from src.config import EVAL_DEFAULT_METRICS

    # Default metrics from centralized config
    if metrics is None:
        metrics = EVAL_DEFAULT_METRICS.copy()

    # Generate run_id for trace
    run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resolved_collection = collection_name or get_collection_name()

    logger.info(f"Starting evaluation with {len(test_questions)} questions")
    logger.info(f"Metrics: {metrics}")

    # Build config dict for trace
    config = {
        "collection": resolved_collection,
        "search_type": search_type,
        "alpha": alpha,
        "top_k": top_k,
        "use_reranking": use_reranking,
        "preprocessing_strategy": preprocessing_strategy,
        "preprocessing_model": preprocessing_model,
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
    }

    # Build evaluation samples and question traces with per-question retry
    samples = []
    question_traces = []
    failed_questions = []

    for i, q in enumerate(test_questions):
        question_id = q.get("id", f"q_{i}")
        question_text = q["question"]

        logger.info(f"Processing question {i + 1}/{len(test_questions)}: {question_text[:50]}...")

        # Process question with retry logic
        result = process_single_question(
            question_data=q,
            question_index=i,
            preprocessing_strategy=preprocessing_strategy,
            preprocessing_model=preprocessing_model,
            top_k=top_k,
            resolved_collection=resolved_collection,
            use_reranking=use_reranking,
            alpha=alpha,
            generation_model=generation_model,
            retrieval_cache=retrieval_cache,
            max_retrieval_k=max_retrieval_k,
            max_retries=ragas_max_retries,
            backoff_base=ragas_backoff_base,
            search_type=search_type,
        )

        if result.success:
            samples.append(result.sample)
            question_traces.append(result.trace)
        else:
            # Track failed question but continue with others
            failed_questions.append({
                "question_id": question_id,
                "question": question_text,
                "error": result.error,
                "failed_at_stage": result.failed_at_stage,
            })
            logger.error(
                f"  FAILED after retries: {result.error} "
                f"(stage: {result.failed_at_stage})"
            )

    # Log summary of failures
    if failed_questions:
        logger.warning(
            f"Failed to process {len(failed_questions)}/{len(test_questions)} questions"
        )
        for fq in failed_questions:
            logger.warning(f"  - {fq['question_id']}: {fq['error'][:100]}...")

    # Check if we have any successful samples
    if not samples:
        raise RAGASEvaluationError(
            f"All {len(test_questions)} questions failed processing. "
            f"First error: {failed_questions[0]['error'] if failed_questions else 'Unknown'}"
        )

    # Create RAGAS dataset
    dataset = EvaluationDataset.from_list(samples)

    # Map metric names to objects
    metric_map = {
        "faithfulness": Faithfulness(),
        "relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithoutReference(),
        "context_recall": LLMContextRecall(),
        "answer_correctness": AnswerCorrectness(),
    }

    # Validate metrics
    selected_metrics = []
    selected_metric_names = []
    for m in metrics:
        if m not in metric_map:
            logger.warning(f"Unknown metric: {m}, skipping")
            continue

        # Check if metric requires reference
        if m in ["context_recall", "answer_correctness"]:
            has_references = all(q.get("reference") for q in test_questions)
            if not has_references:
                logger.warning(f"Metric {m} requires reference answers, skipping")
                continue

        selected_metrics.append(metric_map[m])
        selected_metric_names.append(m)

    if not selected_metrics:
        raise ValueError("No valid metrics selected")

    logger.info(f"Running RAGAS evaluation with {len(selected_metrics)} metrics...")

    # Create evaluator LLM and embeddings
    evaluator_llm = create_evaluator_llm(model=evaluation_model)
    evaluator_embeddings = create_evaluator_embeddings()

    # Create RunConfig for rate limiting control
    ragas_run_config = RunConfig(
        max_workers=ragas_max_workers,
        max_retries=ragas_max_retries,
        max_wait=ragas_max_wait,
        log_tenacity=ragas_log_tenacity,
    )
    logger.info(f"RAGAS RunConfig: max_workers={ragas_max_workers}, max_retries={ragas_max_retries}")

    # Run evaluation with retry logic
    results = evaluate_with_retry(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        max_retries=ragas_max_retries,
        backoff_base=ragas_backoff_base,
        run_config=ragas_run_config,
    )

    logger.info("Evaluation complete")

    # Convert to DataFrame for analysis
    results_df = results.to_pandas()

    # Extract aggregate scores from DataFrame
    scores = {}
    for metric_name in selected_metric_names:
        col_name = RAGAS_METRIC_COLUMN_MAP.get(metric_name, metric_name)
        if col_name in results_df.columns:
            scores[metric_name] = float(results_df[col_name].mean())

    # Populate per-question scores into traces (and track failed metrics)
    for i, trace in enumerate(question_traces):
        if i < len(results_df):
            for metric_name in selected_metric_names:
                col_name = RAGAS_METRIC_COLUMN_MAP.get(metric_name, metric_name)
                if col_name in results_df.columns:
                    value = results_df.iloc[i].get(col_name)
                    if _is_valid_score(value):
                        trace.scores[metric_name] = float(value)
                    else:
                        # Track failed metric (RAGAS returned NaN)
                        trace.failed_metrics[metric_name] = "RAGAS returned NaN (rate limit or parse error)"

    # Log failed metrics summary
    total_failed = sum(len(t.failed_metrics) for t in question_traces)
    if total_failed > 0:
        logger.warning(f"Total failed metrics across all questions: {total_failed}")
        for trace in question_traces:
            if trace.failed_metrics:
                logger.warning(f"  {trace.question_id}: {list(trace.failed_metrics.keys())}")

    # Compute difficulty breakdown (using successfully processed questions only)
    # Build list matching results_df indices from question_traces
    processed_questions = [
        {"difficulty": trace.difficulty, "category": trace.category}
        for trace in question_traces
    ]
    difficulty_breakdown = compute_difficulty_breakdown(
        results_df=results_df,
        test_questions=processed_questions,
        metrics=selected_metric_names,
    )

    # Build and save evaluation trace
    saved_trace_path = None
    if save_trace:
        evaluation_trace = EvaluationTrace(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            questions=question_traces,
            aggregate_scores=scores,
            difficulty_breakdown=difficulty_breakdown,
            ragas_metrics_used=selected_metric_names,
            evaluation_model=evaluation_model,
        )

        # Determine trace path
        if trace_path:
            saved_trace_path = trace_path
        else:
            saved_trace_path = EVAL_TRACES_DIR / f"trace_{run_id}.json"

        evaluation_trace.save(saved_trace_path)
        logger.info(f"Trace saved to: {saved_trace_path}")

    return {
        "scores": scores,
        "results": results_df,
        "samples": samples,
        "difficulty_breakdown": difficulty_breakdown,
        "trace_path": saved_trace_path,
        "failed_questions": failed_questions,
        "questions_processed": len(samples),
        "questions_failed": len(failed_questions),
    }
