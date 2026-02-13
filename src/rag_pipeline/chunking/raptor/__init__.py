"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

This module implements the RAPTOR algorithm from arXiv:2401.18059 (ICLR 2024),
which builds a hierarchical tree of summaries from document chunks for
multi-level retrieval.

Key Components:
    - schemas: Dataclass definitions (RaptorNode, ClusterResult, TreeMetadata)
    - clustering: UMAP dimensionality reduction + GMM soft clustering
    - summarizer: LLM-based cluster summarization
    - tree_builder: Core tree construction orchestration
    - raptor_chunker: Strategy interface (run_raptor_chunking)

Usage:
    >>> from src.rag_pipeline.chunking.raptor import run_raptor_chunking
    >>> stats = run_raptor_chunking()  # Process all semantic_std2 chunks
    >>> print(stats)  # {"book_name": node_count, ...}

Pipeline Integration:
    Stage 4b: python -m src.stages.run_stage_4b_raptor
    Input:  data/processed/05_final_chunks/semantic_std2/{book}.json
    Output: data/processed/05_final_chunks/raptor/{book}.json
"""

from src.rag_pipeline.chunking.raptor.raptor_chunker import run_raptor_chunking
from src.rag_pipeline.chunking.raptor.schemas import (
    RaptorNode,
    ClusterResult,
    TreeMetadata,
)

__all__ = [
    "run_raptor_chunking",
    "RaptorNode",
    "ClusterResult",
    "TreeMetadata",
]
