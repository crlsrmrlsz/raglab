"""OpenRouter embedding client for RAGLab.

Provides API integration for generating text embeddings via OpenRouter.
Includes automatic batching to prevent API timeouts with large inputs.
"""

import time
import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL_ID,
    MAX_BATCH_TOKENS,
)
from src.shared.files import setup_logging
from src.shared.tokens import count_tokens

logger = setup_logging(__name__)

# --------------------------------------------------------------------------------
# OPENROUTER EMBEDDING CLIENT
# --------------------------------------------------------------------------------

def call_openrouter_embeddings_api(
    inputs: list[str],
    max_retries: int = 3,
    backoff_base: float = 1.5
) -> list[list[float]]:
    """
    Calls the OpenRouter embeddings API.

    Args:
        inputs: List of text strings.
        max_retries: How many retries on failure (HTTP/network).
        backoff_base: Backoff multiplier for retry delays.

    Returns:
        A list of embedding vectors (one list per input).
    """
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL_ID,
        "input": inputs,
    }

    attempt = 0
    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # The API returns "data" as a list of embeddings
                embeddings = [item["embedding"] for item in result.get("data", [])]
                return embeddings

            # Rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    response.raise_for_status()
                delay = backoff_base**attempt
                logger.warning(f"Server error {response.status_code}, retry {attempt} after {delay:.1f}s")
                time.sleep(delay)
                continue

            # Hard failure
            response.raise_for_status()

        except requests.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = backoff_base**attempt
            logger.warning(f"Request failed ({exc}), retry {attempt} in {delay:.1f}s")
            time.sleep(delay)
            continue


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts with automatic batching to prevent API timeouts.

    Large embedding requests (500+ texts) can timeout or fail silently.
    This function automatically batches requests to stay under MAX_BATCH_TOKENS.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of vectors (float lists), one per input text.

    Raises:
        Exception: Re-raises API errors after max retries.
    """
    if not texts:
        return []

    # Build token-aware batches
    batches = _batch_texts_by_tokens(texts, MAX_BATCH_TOKENS)

    # Single batch: direct call (common case)
    if len(batches) == 1:
        return call_openrouter_embeddings_api(batches[0])

    # Multiple batches: collect results with inter-request delay
    logger.info(f"Batching {len(texts)} texts into {len(batches)} API calls")
    all_embeddings = []
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(0.1)  # 100ms delay between requests (rate limit gentleness)
        batch_embeddings = call_openrouter_embeddings_api(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _batch_texts_by_tokens(
    texts: list[str],
    max_tokens: int,
) -> list[list[str]]:
    """
    Group texts into batches that stay under max_tokens.

    Args:
        texts: List of texts to batch.
        max_tokens: Maximum tokens per batch.

    Returns:
        List of text batches.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        tokens = count_tokens(text)

        # Single text exceeds limit: put in own batch
        if tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([text])
            continue

        # Would exceed limit: start new batch
        if current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches
