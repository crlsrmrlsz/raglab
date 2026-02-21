"""UMAP dimensionality reduction and GMM clustering for RAPTOR.

## RAG Theory: Why UMAP + GMM?

RAPTOR uses a two-stage clustering approach:

1. **UMAP Reduction**: High-dimensional embeddings (3072 dims for text-embedding-3-large)
   are difficult for GMM to cluster effectively. UMAP reduces to ~10 dimensions while
   preserving local and global structure.

2. **GMM Clustering**: Gaussian Mixture Models allow soft clustering - a chunk can
   belong to multiple clusters with different probabilities. This is semantically
   meaningful: a chunk about "stress and cortisol" might belong to both a
   "neuroscience" cluster and a "health effects" cluster.

3. **BIC for K Selection**: Bayesian Information Criterion balances model fit with
   complexity. Lower BIC = better. We search K from 2 to min(50, n_nodes/2).

## Library Usage

- umap-learn: UMAP dimensionality reduction
- scikit-learn: GaussianMixture for GMM clustering

## Data Flow

1. Input: List of embedding vectors (3072-dim)
2. UMAP reduction: 3072 -> 10 dimensions
3. BIC search: Find optimal number of clusters K
4. GMM clustering: Soft cluster assignments
5. Output: ClusterResult with assignments and probabilities
"""

import numpy as np
from typing import Optional

from src.config import (
    RAPTOR_UMAP_N_NEIGHBORS,
    RAPTOR_UMAP_N_COMPONENTS,
    RAPTOR_MIN_CLUSTERS,
    RAPTOR_MAX_CLUSTERS,
    RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
)
from src.shared.files import setup_logging
from src.rag_pipeline.chunking.raptor.schemas import ClusterResult

logger = setup_logging(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int = RAPTOR_UMAP_N_NEIGHBORS,
    n_components: int = RAPTOR_UMAP_N_COMPONENTS,
    random_state: int = 42,
) -> np.ndarray:
    """Apply UMAP dimensionality reduction to embeddings.

    UMAP (Uniform Manifold Approximation and Projection) reduces high-dimensional
    embeddings while preserving both local and global structure. This makes
    subsequent GMM clustering more effective.

    Uses dynamic n_neighbors like the original RAPTOR implementation:
    n_neighbors = min(config_value, sqrt(n-1), n-1)

    This allows clustering to continue with fewer nodes, enabling deeper trees.
    For small sample sizes (< 15), uses random initialization instead of spectral
    to avoid scipy sparse matrix errors.

    Args:
        embeddings: Input embeddings array of shape (n_samples, embedding_dim).
        n_neighbors: Maximum neighbors for UMAP (actual value is dynamic).
            The effective value is min(n_neighbors, sqrt(n-1), n-1).
        n_components: Target dimensionality (typically 10 for GMM input).
        random_state: Random seed for reproducibility.

    Returns:
        Reduced embeddings of shape (n_samples, n_components).

    Raises:
        ValueError: If embeddings.shape[0] < 3 (minimum for meaningful UMAP).
    """
    # Lazy import to avoid loading UMAP unless needed
    from umap import UMAP

    n_samples = embeddings.shape[0]

    # Dynamic n_neighbors like original RAPTOR: sqrt(n-1)
    # This allows clustering to continue with fewer nodes for deeper trees
    dynamic_neighbors = int((n_samples - 1) ** 0.5)
    effective_neighbors = min(n_neighbors, dynamic_neighbors, n_samples - 1)

    # Minimum 2 neighbors needed for UMAP, which requires at least 3 samples
    if effective_neighbors < 2:
        raise ValueError(
            f"Too few samples ({n_samples}) for meaningful UMAP reduction. "
            f"Need at least 3 samples."
        )

    if effective_neighbors != n_neighbors:
        logger.info(
            f"Dynamic n_neighbors: {n_neighbors} -> {effective_neighbors} "
            f"(sqrt({n_samples}-1) = {dynamic_neighbors})"
        )

    # Use random init for small samples to avoid spectral layout crash
    # Spectral layout fails with scipy sparse matrix when k >= N
    init_method = "random" if n_samples < 15 else "spectral"

    logger.info(
        f"UMAP: {embeddings.shape[1]} dims -> {n_components} dims "
        f"(n_neighbors={effective_neighbors}, init={init_method})"
    )

    reducer = UMAP(
        n_neighbors=effective_neighbors,
        n_components=n_components,
        min_dist=0.0,  # Tight clusters for GMM
        metric="cosine",  # Match embedding distance metric
        random_state=random_state,
        init=init_method,
    )

    reduced = reducer.fit_transform(embeddings)
    return reduced


def find_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int = RAPTOR_MIN_CLUSTERS,
    max_k: int = RAPTOR_MAX_CLUSTERS,
    random_state: int = 42,
) -> tuple[int, float]:
    """Find optimal cluster count using Bayesian Information Criterion (BIC).

    Tests GMM with different K values and selects the one with lowest BIC.
    BIC penalizes model complexity, preventing overfitting.

    Args:
        embeddings: Reduced embeddings from UMAP (n_samples, n_components).
        min_k: Minimum number of clusters to test.
        max_k: Maximum number of clusters to test.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (optimal_k, best_bic_score).

    Raises:
        ValueError: If embeddings.shape[0] < min_k.
    """
    from sklearn.mixture import GaussianMixture

    n_samples = embeddings.shape[0]

    if n_samples < min_k:
        raise ValueError(
            f"Not enough samples ({n_samples}) for clustering with min_k={min_k}."
        )

    # Limit max_k to reasonable range based on sample count
    effective_max_k = min(max_k, n_samples // 2, n_samples - 1)
    effective_max_k = max(effective_max_k, min_k)

    logger.info(f"BIC search: K from {min_k} to {effective_max_k}")

    best_k = min_k
    best_bic = float("inf")
    bic_scores = []

    for k in range(min_k, effective_max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                n_init=3,  # Multiple initializations for robustness
                max_iter=200,
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            bic_scores.append((k, bic))

            if bic < best_bic:
                best_bic = bic
                best_k = k

        except Exception as e:
            logger.warning(f"GMM failed for K={k}: {e}")
            continue

    logger.info(f"Optimal K={best_k} (BIC={best_bic:.2f})")
    return best_k, best_bic


def cluster_nodes(
    embeddings: np.ndarray,
    n_clusters: int,
    probability_threshold: float = RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
    random_state: int = 42,
) -> ClusterResult:
    """Perform soft GMM clustering on embeddings.

    Uses Gaussian Mixture Model for soft clustering, where each node has
    a probability of belonging to each cluster. Hard assignments are made
    to the highest probability cluster.

    Args:
        embeddings: Reduced embeddings from UMAP (n_samples, n_components).
        n_clusters: Number of clusters (from find_optimal_clusters).
        probability_threshold: Minimum probability for cluster membership.
            Nodes with P < threshold are still assigned to highest-prob cluster.
        random_state: Random seed for reproducibility.

    Returns:
        ClusterResult with assignments and probability matrix.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=3,
        max_iter=200,
    )
    gmm.fit(embeddings)

    # Get soft probabilities (n_samples x n_clusters)
    probabilities = gmm.predict_proba(embeddings)

    # Hard assignment: highest probability cluster
    hard_assignments = probabilities.argmax(axis=1).tolist()

    # Count nodes per cluster
    nodes_per_cluster = [0] * n_clusters
    for assignment in hard_assignments:
        nodes_per_cluster[assignment] += 1

    # Log cluster sizes
    logger.info(f"Cluster sizes: {nodes_per_cluster}")

    # Count nodes above threshold (for logging only with soft clustering)
    above_threshold = (probabilities >= probability_threshold).sum(axis=1)
    multi_cluster = (above_threshold > 1).sum()
    if multi_cluster > 0:
        logger.info(
            f"Soft clustering: {multi_cluster}/{len(hard_assignments)} nodes "
            f"belong to multiple clusters (P >= {probability_threshold})"
        )

    return ClusterResult(
        n_clusters=n_clusters,
        cluster_assignments=hard_assignments,
        cluster_probabilities=probabilities.tolist(),
        bic_score=gmm.bic(embeddings),
        nodes_per_cluster=nodes_per_cluster,
    )


def get_cluster_members(
    cluster_result: ClusterResult,
    cluster_id: int,
    probability_threshold: Optional[float] = None,
) -> list[int]:
    """Get indices of nodes belonging to a cluster.

    With hard assignment (default), returns nodes assigned to this cluster.
    With soft assignment (threshold provided), includes nodes with P >= threshold.

    Args:
        cluster_result: Result from cluster_nodes().
        cluster_id: Cluster ID to get members for.
        probability_threshold: If provided, use soft assignment with this threshold.

    Returns:
        List of node indices belonging to this cluster.
    """
    if probability_threshold is None:
        # Hard assignment
        return [
            i
            for i, assignment in enumerate(cluster_result.cluster_assignments)
            if assignment == cluster_id
        ]
    else:
        # Soft assignment
        return [
            i
            for i, probs in enumerate(cluster_result.cluster_probabilities)
            if probs[cluster_id] >= probability_threshold
        ]
