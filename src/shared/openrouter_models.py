"""Fetch available models from OpenRouter API with curated selection.

Provides dynamic model loading with category-based curation.
Each task (preprocessing, generation) gets 4-5 curated options:
- Budget: Cheapest viable option
- Value: Best quality/price ratio
- Quality: Higher quality, moderate cost
- Premium: Best available for those who want top results
"""

import os
from typing import Optional, Any

import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# =============================================================================
# CURATED MODEL PRIORITIES
# =============================================================================
# Each category has a list of model IDs in order of preference.
# The first available model in each category is selected.

PREPROCESSING_CURATED: dict[str, list[str]] = {
    # Simple tasks: hyde, decomposition
    # Prioritize speed and cost over deep reasoning
    "Budget": [
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-chat",
        "mistralai/ministral-8b-latest",
    ],
    "Value": [
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "google/gemini-2.0-flash-001",
    ],
    "Quality": [
        "anthropic/claude-haiku-4.5",
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
    ],
    "Premium": [
        "anthropic/claude-opus-4.5",
        "openai/gpt-5.2-chat",
        "anthropic/claude-haiku-4.5",
    ],
}

GENERATION_CURATED: dict[str, list[str]] = {
    # Complex tasks: answer synthesis, reasoning across sources
    # Quality matters more than speed
    "Budget": [
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-chat",
        "openai/gpt-5-mini",
    ],
    "Value": [
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "deepseek/deepseek-v3.2",
    ],
    "Quality": [
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.2-chat",
        "google/gemini-3-flash-preview",
    ],
    "Premium": [
        "anthropic/claude-opus-4.5",
        "google/gemini-3-pro-preview",
        "openai/gpt-5.2-chat",
    ],
}


# =============================================================================
# FALLBACK MODELS (used if API fetch fails)
# =============================================================================

FALLBACK_PREPROCESSING_MODELS: list[tuple[str, str]] = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("google/gemini-3-flash-preview", "Value: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Quality: Claude Haiku 4.5"),
    ("anthropic/claude-opus-4.5", "Premium: Claude Opus 4.5"),
]

FALLBACK_GENERATION_MODELS: list[tuple[str, str]] = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("google/gemini-3-flash-preview", "Value: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Quality: Claude Haiku 4.5"),
    ("anthropic/claude-opus-4.5", "Premium: Claude Opus 4.5"),
]


# =============================================================================
# API FETCHING
# =============================================================================


def fetch_available_models() -> Optional[list[dict[str, Any]]]:
    """Fetch all available models from OpenRouter API.

    Returns:
        List of model dictionaries with id, name, pricing, etc.
        None if API call fails.
    """
    if not OPENROUTER_API_KEY:
        return None

    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
    except requests.RequestException:
        pass
    return None


def _format_price(price_per_token: float) -> str:
    """Format price per token as price per 1M tokens."""
    if price_per_token == 0:
        return "free"
    price_per_million = price_per_token * 1_000_000
    if price_per_million < 0.01:
        return f"${price_per_million:.3f}"
    elif price_per_million < 1:
        return f"${price_per_million:.2f}"
    else:
        return f"${price_per_million:.1f}"


def _format_curated_label(category: str, model: dict[str, Any]) -> str:
    """Create a curated label: 'Category: Name ($in/$out)'."""
    name = model.get("name", model["id"].split("/")[-1])
    pricing = model.get("pricing", {})

    prompt_price = float(pricing.get("prompt", 0))
    completion_price = float(pricing.get("completion", 0))

    prompt_str = _format_price(prompt_price)
    completion_str = _format_price(completion_price)

    return f"{category}: {name} ({prompt_str}/{completion_str})"


def _get_curated_models(
    all_models: list[dict[str, Any]],
    curated_priorities: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Select one model per category from curated priorities.

    For each category (Budget, Value, Quality, Premium), finds the first
    available model from the priority list and includes it with live pricing.

    Args:
        all_models: List of all models from OpenRouter API.
        curated_priorities: Dict mapping category -> list of model IDs.

    Returns:
        List of (model_id, "Category: Name ($X/$Y)") tuples.
    """
    # Build lookup for quick access
    models_by_id = {m["id"]: m for m in all_models}
    result = []

    for category, candidates in curated_priorities.items():
        for model_id in candidates:
            if model_id in models_by_id:
                model = models_by_id[model_id]
                label = _format_curated_label(category, model)
                result.append((model_id, label))
                break  # Found one for this category, move to next

    return result


# =============================================================================
# PUBLIC API
# =============================================================================


def get_preprocessing_models(
    cached_models: Optional[list[dict[str, Any]]] = None
) -> list[tuple[str, str]]:
    """Get curated models for preprocessing (hyde, decomposition).

    Returns 4 options: Budget, Value, Quality, Premium.
    Each option is the best available model in that category.

    Args:
        cached_models: Optional pre-fetched models to avoid repeat API calls.

    Returns:
        List of (model_id, label) tuples.
    """
    models = cached_models or fetch_available_models()

    if not models:
        return FALLBACK_PREPROCESSING_MODELS

    curated = _get_curated_models(models, PREPROCESSING_CURATED)
    return curated if curated else FALLBACK_PREPROCESSING_MODELS


def get_generation_models(
    cached_models: Optional[list[dict[str, Any]]] = None
) -> list[tuple[str, str]]:
    """Get curated models for answer generation.

    Returns 4 options: Budget, Value, Quality, Premium.
    Generation benefits from higher quality models for reasoning.

    Args:
        cached_models: Optional pre-fetched models to avoid repeat API calls.

    Returns:
        List of (model_id, label) tuples.
    """
    models = cached_models or fetch_available_models()

    if not models:
        return FALLBACK_GENERATION_MODELS

    curated = _get_curated_models(models, GENERATION_CURATED)
    return curated if curated else FALLBACK_GENERATION_MODELS
