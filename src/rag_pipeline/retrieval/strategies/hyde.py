"""HyDE retrieval strategy with embedding averaging.

Hypothetical Document Embeddings (HyDE) generates plausible answers
to the query, then averages their embeddings for a single semantic search.

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
       arXiv:2212.10496 (Gao, Ma, Lin, Callan)

Key algorithm:
    1. Generate K hypothetical answers to the query
    2. Embed original query + K hypotheticals (K+1 vectors)
    3. Average embeddings element-wise
    4. Single semantic search with averaged embedding (alpha=1.0)
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_registry import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query
from src.rag_pipeline.retrieval.query_strategy_config import get_strategy_config
from src.rag_pipeline.retrieval.query_helpers import execute_search
from src.rag_pipeline.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_query import query_hybrid
from src.rag_pipeline.retrieval.reranking import apply_reranking_if_enabled
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class HyDERetrieval:
    """HyDE strategy: Generate hypotheticals → Average embeddings → Single search.

    Paper-aligned implementation (arXiv:2212.10496):
    - Generates K hypothetical answers (default K=4)
    - Averages embeddings: original query + K hypotheticals
    - Single semantic search with averaged embedding (alpha=1.0)

    This encapsulates all the embedding averaging logic that was previously
    scattered between strategies.py and retrieval_helpers.py.

    Attributes:
        strategy_id: "hyde" - identifies this as the HyDE strategy.
    """

    strategy_id = "hyde"

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        preprocessing_model: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute HyDE retrieval with embedding averaging.

        Args:
            query: User's original query text.
            context: Retrieval context with Weaviate client and settings.
            preprocessing_model: Model for generating hypothetical answers.

        Returns:
            RetrievalResult with search results and HyDE metadata.
        """
        # Preprocessing: Generate K hypothetical answers
        preprocessed = preprocess_query(
            query=query,
            strategy="hyde",
            model=preprocessing_model,
        )

        # Extract all queries for embedding (original + hypotheticals)
        generated = preprocessed.generated_queries or []
        all_queries = [q.get("query", "") for q in generated if q.get("query")]

        # Get alpha constraint from strategy config (HyDE requires alpha=1.0)
        config = get_strategy_config("hyde")
        hyde_alpha = config.alpha_constraint.fixed_value or 1.0

        if len(all_queries) <= 1:
            # Fallback: if generation failed, use standard search
            logger.warning("[hyde] No hypotheticals generated, falling back to standard search")
            results = execute_search(context, query, context.initial_k, alpha=hyde_alpha)
        else:
            # Paper-aligned: Average embeddings
            logger.info(
                f"[hyde] Paper-aligned: averaging {len(all_queries)} embeddings, alpha={hyde_alpha}"
            )
            embeddings = embed_texts(all_queries)
            avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]

            # Single search with averaged embedding (pure semantic, no BM25)
            results = query_hybrid(
                client=context.client,
                query_text=query,  # For display/BM25 (ignored at alpha=1.0)
                top_k=context.initial_k,
                alpha=hyde_alpha,
                collection_name=context.collection_name,
                precomputed_embedding=avg_embedding,
            )

        # Apply reranking if enabled
        results = apply_reranking_if_enabled(
            results,
            query,
            context.top_k,
            context.use_reranking,
        )

        logger.info(f"[hyde] Retrieved {len(results)} results")

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata={
                "strategy": "hyde",
                "queries_averaged": len(all_queries),
                "alpha": hyde_alpha,
                "hypotheticals_generated": len(all_queries) - 1,  # Exclude original
            },
        )
