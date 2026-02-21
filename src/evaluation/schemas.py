"""Evaluation trace schemas for RAGLab.

Provides data structures for persisting evaluation traces, enabling:
- Recalculation of RAGAS metrics without re-running retrieval/generation
- Debugging failed evaluations
- Historical comparison of evaluation runs

Trace files are stored at: data/evaluation/traces/trace_{run_id}.json
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class QuestionTrace:
    """Trace data for a single question evaluation.

    Stores all information needed to recalculate RAGAS metrics for one question
    without re-running the retrieval and generation pipeline.

    Attributes:
        question_id: Unique identifier from test questions (e.g., "neuro_behave_01").
        question: The original question text.
        difficulty: Question difficulty level ("single_concept" or "cross_domain").
        category: Question category ("neuroscience", "philosophy", "cross_domain").
        reference: Ground truth answer if available (required for some metrics).
        preprocessing_strategy: Strategy used ("none", "hyde", "decomposition", "graphrag").
        search_query: The query after preprocessing (may differ from original).
        generated_queries: For decomposition/hyde - list of generated sub-queries/passages.
        retrieved_contexts: List of chunk texts (no embeddings - text only for size).
        retrieval_metadata: Configuration used (top_k, alpha, collection, reranking).
        generated_answer: The LLM-generated answer.
        generation_model: Model used for answer generation.
        scores: RAGAS metric scores (populated after evaluation).
        failed_metrics: Dict of metric_name -> error_message for metrics that returned NaN.
                       Enables retry of just failed metrics without re-running the full pipeline.
    """

    question_id: str
    question: str
    difficulty: str
    category: str
    reference: Optional[str]

    # Preprocessing
    preprocessing_strategy: str
    search_query: str
    generated_queries: Optional[list[dict[str, str]]] = None

    # Retrieval
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    # Generation
    generated_answer: str = ""
    generation_model: str = ""

    # RAGAS scores (filled after evaluation)
    scores: dict[str, float] = field(default_factory=dict)

    # Failed metrics (RAGAS returned NaN - enables retry of just failed metrics)
    failed_metrics: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionTrace":
        """Create from dict (for loading from JSON)."""
        return cls(**data)


@dataclass
class EvaluationTrace:
    """Complete trace for one evaluation run.

    Stores all information needed to understand and potentially recalculate
    an entire evaluation run. Includes configuration, per-question traces,
    aggregate scores, and difficulty breakdown.

    Attributes:
        run_id: Unique identifier for this run (e.g., "eval_20251231_120000").
        timestamp: ISO format timestamp of when evaluation started.
        config: Full configuration dict (collection, alpha, strategy, models, etc.).
        questions: List of QuestionTrace objects for each evaluated question.
        aggregate_scores: Average scores across all questions.
        difficulty_breakdown: Per-difficulty group metric averages.
        ragas_metrics_used: List of RAGAS metrics that were calculated.
        evaluation_model: Model used for RAGAS evaluation.
        error: Error message if evaluation failed (None if successful).
    """

    run_id: str
    timestamp: str
    config: dict[str, Any]
    questions: list[QuestionTrace] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    difficulty_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    ragas_metrics_used: list[str] = field(default_factory=list)
    evaluation_model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "questions": [q.to_dict() for q in self.questions],
            "aggregate_scores": self.aggregate_scores,
            "difficulty_breakdown": self.difficulty_breakdown,
            "ragas_metrics_used": self.ragas_metrics_used,
            "evaluation_model": self.evaluation_model,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationTrace":
        """Create from dict (for loading from JSON)."""
        questions = [QuestionTrace.from_dict(q) for q in data.get("questions", [])]
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            config=data["config"],
            questions=questions,
            aggregate_scores=data.get("aggregate_scores", {}),
            difficulty_breakdown=data.get("difficulty_breakdown", {}),
            ragas_metrics_used=data.get("ragas_metrics_used", []),
            evaluation_model=data.get("evaluation_model", ""),
            error=data.get("error"),
        )

    def save(self, path: Path) -> None:
        """Save trace to JSON file.

        Args:
            path: Path to save the trace file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvaluationTrace":
        """Load trace from JSON file.

        Args:
            path: Path to the trace file.

        Returns:
            EvaluationTrace instance.

        Raises:
            FileNotFoundError: If trace file doesn't exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class FailedCombination:
    """Record of a failed evaluation combination in comprehensive mode.

    Attributes:
        collection: Weaviate collection name.
        alpha: Hybrid search alpha value.
        top_k: Number of chunks retrieved.
        strategy: Preprocessing strategy used.
        reranking: Whether reranking was enabled for this combination.
        error_type: Exception class name (e.g., "RateLimitError").
        error_message: Full error message.
        failed_at_stage: Where failure occurred (preprocessing, retrieval, generation, ragas_evaluation).
        attempt_count: Number of times this combination was attempted.
        last_attempt_timestamp: ISO timestamp of last attempt.
    """

    collection: str
    alpha: float
    top_k: int
    strategy: str
    reranking: bool
    error_type: str
    error_message: str
    failed_at_stage: str
    attempt_count: int = 1
    last_attempt_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedCombination":
        """Create from dict (for loading from JSON).

        Handles backward compatibility for files saved before the reranking
        field was added by defaulting to False.
        """
        if "reranking" not in data:
            data = {**data, "reranking": False}
        return cls(**data)


@dataclass
class FailedCombinationsReport:
    """Collection of failed combinations from a comprehensive evaluation run.

    Attributes:
        comprehensive_run_id: ID of the comprehensive evaluation run.
        timestamp: When the report was created.
        failed_combinations: List of FailedCombination records.
    """

    comprehensive_run_id: str
    timestamp: str
    failed_combinations: list[FailedCombination] = field(default_factory=list)

    @property
    def total_failed(self) -> int:
        """Get total count of failed combinations."""
        return len(self.failed_combinations)

    @property
    def cli_command(self) -> str:
        """Generate CLI command to retry failed combinations."""
        return f"python -m src.stages.run_stage_7_evaluation --retry-failed {self.comprehensive_run_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "metadata": {
                "comprehensive_run_id": self.comprehensive_run_id,
                "timestamp": self.timestamp,
                "total_failed": self.total_failed,
            },
            "failed_combinations": [fc.to_dict() for fc in self.failed_combinations],
            "cli_command": self.cli_command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedCombinationsReport":
        """Create from dict (for loading from JSON)."""
        metadata = data.get("metadata", {})
        failed = [FailedCombination.from_dict(fc) for fc in data.get("failed_combinations", [])]
        return cls(
            comprehensive_run_id=metadata.get("comprehensive_run_id", ""),
            timestamp=metadata.get("timestamp", ""),
            failed_combinations=failed,
        )

    def save(self, path: Path) -> None:
        """Save report to JSON file.

        Args:
            path: Path to save the report file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "FailedCombinationsReport":
        """Load report from JSON file.

        Args:
            path: Path to the report file.

        Returns:
            FailedCombinationsReport instance.

        Raises:
            FileNotFoundError: If report file doesn't exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_failure(
        self,
        collection: str,
        alpha: float,
        top_k: int,
        strategy: str,
        error: Exception,
        failed_at_stage: str,
        reranking: bool = False,
    ) -> None:
        """Add a failed combination to the report.

        Args:
            collection: Weaviate collection name.
            alpha: Hybrid search alpha value.
            top_k: Number of chunks retrieved.
            strategy: Preprocessing strategy.
            error: The exception that caused the failure.
            failed_at_stage: Where in the pipeline the failure occurred.
            reranking: Whether reranking was enabled for this combination.
        """
        self.failed_combinations.append(
            FailedCombination(
                collection=collection,
                alpha=alpha,
                top_k=top_k,
                strategy=strategy,
                reranking=reranking,
                error_type=type(error).__name__,
                error_message=str(error),
                failed_at_stage=failed_at_stage,
            )
        )
