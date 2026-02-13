"""Retrieval strategy implementations.

Each strategy encapsulates its own preprocessing + retrieval logic,
eliminating the confusing multi_queries dual semantics.

Strategies:
    StandardRetrieval: No preprocessing, direct Weaviate search
    HyDERetrieval: Hypothetical document embeddings (arXiv:2212.10496)
    DecompositionRetrieval: Query decomposition + union merge + rerank (arXiv:2507.00355)
    GraphRAGRetrieval: Pure graph retrieval with combined_degree ranking (arXiv:2404.16130)
"""

from src.rag_pipeline.retrieval.strategies.standard import StandardRetrieval
from src.rag_pipeline.retrieval.strategies.hyde import HyDERetrieval
from src.rag_pipeline.retrieval.strategies.decomposition import DecompositionRetrieval
from src.rag_pipeline.retrieval.strategies.graphrag import GraphRAGRetrieval

__all__ = [
    "StandardRetrieval",
    "HyDERetrieval",
    "DecompositionRetrieval",
    "GraphRAGRetrieval",
]
