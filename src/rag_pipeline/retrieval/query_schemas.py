"""Pydantic schemas for query preprocessing LLM responses.

## RAG Theory: Query Preprocessing Schemas

These schemas define the expected structure of LLM responses during
query preprocessing. Each schema corresponds to a specific preprocessing
function and ensures type-safe extraction of decomposition results.

Benefits of schema-based parsing:
1. **Guaranteed types** - No more isinstance() checks or .get() fallbacks
2. **Clear contracts** - Schema documents exactly what LLM should return
3. **Validation errors** - Descriptive errors instead of silent failures
4. **IDE support** - Autocomplete and type hints for response fields

## Library Usage

Uses Pydantic v2 BaseModel with:
- Field() for defaults and descriptions
- model_validate_json() for parsing

## Data Flow

1. System prompt instructs LLM to return specific JSON structure
2. OpenRouter enforces schema via response_format (if supported)
3. Pydantic validates and parses response into typed object
4. Calling code accesses fields with full type safety
"""

from pydantic import BaseModel, Field


class DecompositionResult(BaseModel):
    """Result of query decomposition for complex queries.

    Used by: decompose_query()

    Contains 2-4 sub-questions that can be answered independently.
    Each sub-question is used for separate retrieval, with results
    merged using RRF. This handles complex comparison or multi-aspect
    questions that span multiple domains.

    Example for "Compare Stoic and Buddhist views on suffering":
        {
            "sub_questions": [
                "What is the Stoic view on suffering and how to overcome it?",
                "What is the Buddhist teaching on suffering and its cessation?",
                "How do Stoic and Buddhist approaches to suffering differ?"
            ],
            "reasoning": "The question asks for comparison, so we need each
                          tradition's view separately, then a synthesis."
        }
    """

    sub_questions: list[str] = Field(
        description="Self-contained sub-questions for independent retrieval"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition approach",
    )
