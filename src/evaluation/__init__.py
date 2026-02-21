"""RAG evaluation module using RAGAS framework.

Provides:
- RAGAS metric evaluation for RAG responses
- Test question management
- Evaluation report generation
- Trace schemas for evaluation persistence
"""

from src.evaluation.ragas_evaluator import (
    run_evaluation,
    create_evaluator_llm,
    retrieve_contexts,
    generate_answer,
    RAGASEvaluationError,
    evaluate_with_retry,
    compute_difficulty_breakdown,
)
from src.evaluation.schemas import (
    QuestionTrace,
    EvaluationTrace,
    FailedCombination,
    FailedCombinationsReport,
)

__all__ = [
    "run_evaluation",
    "create_evaluator_llm",
    "retrieve_contexts",
    "generate_answer",
    "RAGASEvaluationError",
    "evaluate_with_retry",
    "compute_difficulty_breakdown",
    "QuestionTrace",
    "EvaluationTrace",
    "FailedCombination",
    "FailedCombinationsReport",
]
