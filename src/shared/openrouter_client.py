"""Unified OpenRouter API client.

## RAG Theory: Centralized LLM Communication

All LLM calls in a RAG pipeline should go through a single client for:
- Consistent retry logic and error handling
- Centralized rate limit management
- Request/response logging for debugging
- Easy model switching for A/B testing

## Library Usage

Uses `requests` for HTTP calls. The retry pattern implements
exponential backoff, which is essential for handling:
- Rate limits (429 errors) from OpenRouter
- Temporary server errors (5xx)
- Network transients

## Data Flow

1. Module (preprocessing/generation/evaluation) needs LLM
2. Imports call_chat_completion() from here
3. Constructs messages in OpenAI format
4. Receives response string or raises exception
"""

import json
import os
import time
from typing import Optional, Any, Type, TypeVar

from json_repair import repair_json
from pydantic import BaseModel, ValidationError as PydanticValidationError

T = TypeVar("T", bound=BaseModel)

import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


from src.shared.files import setup_logging

logger = setup_logging(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded after all retries."""
    pass


class APIError(OpenRouterError):
    """Raised when API returns an error response."""
    pass


def call_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_p: Optional[float] = None,
    json_mode: bool = False,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> str:
    """Call OpenRouter chat completion API with retry logic.

    This is the unified LLM call function used by all modules:
    - preprocessing: Step-back, multi-query, decomposition
    - generation: Answer synthesis
    - evaluation: RAGAS answer generation

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello"}]
        model: OpenRouter model ID (e.g., "deepseek/deepseek-v3.2").
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold (0.0-1.0). If set, only tokens with
            cumulative probability <= top_p are considered.
        json_mode: If True, request JSON response format.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        backoff_base: Backoff multiplier for retries.

    Returns:
        The assistant's response content as a string.

    Raises:
        OpenRouterError: On API errors after all retries.
        RateLimitError: If rate limited after all retries.

    Example:
        >>> response = call_chat_completion(
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     model="deepseek/deepseek-v3.2",
        ... )
        >>> print(response)
        "4"
    """
    if not OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if top_p is not None:
        payload["top_p"] = top_p

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()

                # Handle error responses that return 200 but no choices
                # (OpenRouter sometimes does this for rate limits or overload)
                if "choices" not in result:
                    error_msg = result.get("error", {}).get("message", str(result))
                    if attempt < max_retries:
                        delay = backoff_base ** (attempt + 1)
                        logger.warning(
                            f"API returned 200 with error: {error_msg}, "
                            f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    raise APIError(f"API error after {max_retries} retries: {error_msg}")

                content = result["choices"][0]["message"]["content"]

                # Log successful LLM call
                chars_in = sum(len(m.get("content", "")) for m in messages)
                chars_out = len(content)
                logger.info(f"[LLM] model={model} chars_in={chars_in} chars_out={chars_out}")

                return content

            # Retryable errors: rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries:
                    delay = backoff_base ** (attempt + 1)
                    error_type = "Rate limit" if response.status_code == 429 else "Server error"
                    logger.warning(
                        f"{error_type} ({response.status_code}), "
                        f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    if response.status_code == 429:
                        raise RateLimitError(f"Rate limited after {max_retries} retries")
                    raise APIError(f"Server error {response.status_code} after {max_retries} retries")

            # Non-retryable client errors
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except (ValueError, KeyError):
                error_detail = response.text
            raise APIError(f"API error {response.status_code}: {error_detail}")

        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"Request failed ({exc}), retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise OpenRouterError(f"Request failed after {max_retries} retries: {exc}")

    raise OpenRouterError("Max retries exceeded")


def call_simple_prompt(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """Convenience wrapper for single-prompt calls.

    Converts a simple prompt string to the messages format.
    Useful for evaluation and simple LLM tasks.

    Args:
        prompt: The prompt text.
        model: OpenRouter model ID.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        **kwargs: Additional arguments passed to call_chat_completion.

    Returns:
        The model's response text.

    Example:
        >>> response = call_simple_prompt(
        ...     "Summarize: The quick brown fox...",
        ...     model="deepseek/deepseek-v3.2",
        ... )
    """
    messages = [{"role": "user", "content": prompt}]
    return call_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def call_structured_completion(
    messages: list[dict[str, str]],
    model: str,
    response_model: Type[T],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 60,
    max_retries: int = 5,
    backoff_base: float = 2.5,
) -> T:
    """Call OpenRouter with JSON Schema mode and Pydantic validation.

    This function provides structured outputs by:
    1. Generating a JSON Schema from the Pydantic model
    2. Requesting schema-constrained output from the model (json_schema mode)
    3. Parsing and validating response with Pydantic

    Uses json_schema mode to constrain LLM output at generation time.
    Pydantic validation provides a secondary check with JSON repair fallback.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: OpenRouter model ID (e.g., "deepseek/deepseek-v3.2").
        response_model: Pydantic BaseModel class defining expected response.
        temperature: Sampling temperature (0.0 recommended for structured).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        backoff_base: Backoff multiplier for retries.

    Returns:
        Validated Pydantic model instance of type response_model.

    Raises:
        OpenRouterError: On API errors after all retries.
        RateLimitError: If rate limited after all retries.
        PydanticValidationError: If response fails Pydantic validation.

    Example:
        >>> from pydantic import BaseModel
        >>> class Result(BaseModel):
        ...     answer: str
        >>> result = call_structured_completion(
        ...     messages=[{"role": "user", "content": "Say hello"}],
        ...     model="deepseek/deepseek-v3.2",
        ...     response_model=Result,
        ... )
        >>> result.answer
        "Hello!"
    """
    if not OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build payload with json_schema mode for schema-guided output.
    # strict=false allows defaults and constraints (minLength, min/max)
    # that strict mode would reject. Some models still wrap output in
    # markdown fences, which is handled by the fence-stripping below.
    json_schema = response_model.model_json_schema()

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": False,
                "schema": json_schema,
            },
        },
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()

                # Handle malformed response (missing 'choices' key)
                if "choices" not in result or not result["choices"]:
                    if attempt < max_retries:
                        logger.warning(
                            f"Malformed API response (missing 'choices'), "
                            f"retry {attempt + 1}/{max_retries}"
                        )
                        time.sleep(backoff_base)
                        continue
                    raise APIError(f"Malformed API response after retries: {result}")

                content = result["choices"][0]["message"]["content"]

                # Strip markdown code fences (```json ... ```) that some
                # models wrap around JSON output in json_object mode
                stripped = content.strip()
                if stripped.startswith("```"):
                    # Remove opening fence (```json or ```)
                    first_newline = stripped.index("\n")
                    stripped = stripped[first_newline + 1:]
                    # Remove closing fence
                    if stripped.endswith("```"):
                        stripped = stripped[:-3].rstrip()
                    content = stripped

                # Log successful LLM call
                chars_in = sum(len(m.get("content", "")) for m in messages)
                chars_out = len(content)
                logger.info(f"[LLM] model={model} chars_in={chars_in} chars_out={chars_out} (structured)")

                # Parse and validate with Pydantic (repair malformed JSON as fallback)
                try:
                    return response_model.model_validate_json(content)
                except PydanticValidationError:
                    repaired = repair_json(content, return_objects=False)
                    logger.warning(
                        f"Repaired malformed JSON from LLM "
                        f"({len(content)} -> {len(repaired)} chars)"
                    )
                    return response_model.model_validate_json(repaired)

            # Retryable errors: rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries:
                    # Use retry-after header if provided, otherwise exponential backoff
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = backoff_base ** (attempt + 1)
                    else:
                        delay = backoff_base ** (attempt + 1)
                    error_type = "Rate limit" if response.status_code == 429 else "Server error"
                    logger.warning(
                        f"{error_type} ({response.status_code}), "
                        f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    if response.status_code == 429:
                        raise RateLimitError(f"Rate limited after {max_retries} retries")
                    raise APIError(f"Server error {response.status_code} after {max_retries} retries")

            # Non-retryable client errors
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except (ValueError, KeyError):
                error_detail = response.text
            raise APIError(f"API error {response.status_code}: {error_detail}")

        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"Request failed ({exc}), retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise OpenRouterError(f"Request failed after {max_retries} retries: {exc}")

    raise OpenRouterError("Max retries exceeded")
