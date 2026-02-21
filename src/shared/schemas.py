"""Pydantic schema utilities for structured LLM outputs.

## RAG Theory: Structured Outputs

LLMs can produce unreliable JSON without schema enforcement. Pydantic models:
1. Define expected response structure with type hints
2. Validate LLM outputs automatically
3. Provide IDE autocompletion and type safety
4. Generate JSON Schema for OpenRouter's strict mode

## Library Usage

Pydantic BaseModel generates JSON Schema via model_json_schema().
OpenRouter's response_format accepts this schema for guaranteed compliance.
With strict mode, the LLM is constrained to produce only valid JSON.

## Data Flow

1. Define Pydantic model with expected fields
2. Generate schema via model.model_json_schema()
3. Pass schema in API request's response_format parameter
4. LLM produces schema-compliant JSON
5. Validate with model.model_validate_json()
"""
