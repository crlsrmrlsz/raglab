"""Query logging for evaluation and analysis.

Saves all query execution data to a JSON file for:
- RAGAS evaluation
- Performance tracking
- Query analysis
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import PROJECT_ROOT

QUERY_LOG_FILE = PROJECT_ROOT / "data" / "evaluation" / "ui_query_traces.json"


def log_query(
    query: str,
    preprocessed: Optional[Any],
    retrieval_settings: dict,
    search_results: list[dict],
    rerank_data: Optional[Any],
    generated_answer: Optional[Any],
    collection_name: str = "",
    rrf_data: Optional[Any] = None,
) -> str:
    """Log a query execution to JSON file.

    Args:
        query: Original user query.
        preprocessed: PreprocessedQuery object (or None if disabled).
        retrieval_settings: Dict with search_type, alpha, top_k.
        search_results: List of chunk dicts from search.
        rerank_data: RerankResult object (or None if disabled).
        generated_answer: GeneratedAnswer object (or None if disabled).
        collection_name: Weaviate collection name.
        rrf_data: RRFResult object (or None if multi-query not used).

    Returns:
        Query ID (UUID string).
    """
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {"query": query},
        "preprocessing": _build_preprocessing(preprocessed),
        "retrieval": _build_retrieval(retrieval_settings, search_results, collection_name),
        "rrf_merging": _build_rrf(rrf_data),
        "reranking": _build_reranking(rerank_data),
        "generation": _build_generation(generated_answer),
    }

    # Load existing, append, save
    QUERY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"queries": []}
    if QUERY_LOG_FILE.exists():
        try:
            data = json.loads(QUERY_LOG_FILE.read_text())
        except json.JSONDecodeError:
            data = {"queries": []}

    data["queries"].append(record)
    QUERY_LOG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return record["id"]


def _build_preprocessing(prep) -> dict:
    """Build preprocessing section from PreprocessedQuery."""
    if not prep:
        return {"enabled": False}

    result = {
        "enabled": True,
        "strategy": getattr(prep, "strategy_used", "unknown"),
        "model": getattr(prep, "model", ""),
        "search_query": prep.search_query,
        "hyde_passage": getattr(prep, "hyde_passage", None),
        "time_ms": round(prep.preprocessing_time_ms, 1),
    }

    # Add multi-query fields if present
    generated_queries = getattr(prep, "generated_queries", None)
    if generated_queries:
        result["generated_queries"] = generated_queries

    principle_extraction = getattr(prep, "principle_extraction", None)
    if principle_extraction:
        result["principle_extraction"] = principle_extraction

    # Add decomposition sub-queries if present
    sub_queries = getattr(prep, "sub_queries", None)
    if sub_queries:
        result["sub_queries"] = sub_queries

    return result


def _build_retrieval(settings: dict, results: list[dict], collection: str) -> dict:
    """Build retrieval section from settings and search results."""
    return {
        "search_type": settings.get("search_type", ""),
        "alpha": settings.get("alpha", 0.5),
        "top_k": settings.get("top_k", 10),
        "collection": collection,
        "chunks": [
            {
                "rank": i + 1,
                "chunk_id": r.get("chunk_id", ""),
                "book_id": r.get("book_id", ""),
                "section": r.get("section", ""),
                "text": r.get("text", ""),
                "score": round(r.get("similarity", 0), 4),
            }
            for i, r in enumerate(results)
        ],
    }


def _build_reranking(rerank) -> dict:
    """Build reranking section from RerankResult."""
    if not rerank:
        return {"enabled": False}
    return {
        "enabled": True,
        "model": rerank.model,
        "time_ms": round(rerank.rerank_time_ms, 1),
        "order_changes": rerank.order_changes,
    }


def _build_rrf(rrf_data) -> dict:
    """Build RRF merging section from RRFResult."""
    if not rrf_data:
        return {"enabled": False}

    # Get query contributions summary (top 10)
    contributions = {}
    if hasattr(rrf_data, "query_contributions") and rrf_data.query_contributions:
        contributions = dict(list(rrf_data.query_contributions.items())[:10])

    return {
        "enabled": True,
        "queries_merged": len(rrf_data.query_contributions) if hasattr(rrf_data, "query_contributions") else 0,
        "time_ms": round(getattr(rrf_data, "merge_time_ms", 0), 1),
        "query_contributions": contributions,
    }


def _build_generation(ans) -> dict:
    """Build generation section from GeneratedAnswer."""
    if not ans:
        return {"enabled": False}
    return {
        "enabled": True,
        "model": ans.model,
        "answer": ans.answer,
        "sources_cited": ans.sources_used,
        "time_ms": round(ans.generation_time_ms, 1),
    }
