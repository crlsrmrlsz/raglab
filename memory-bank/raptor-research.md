# RAPTOR Research: Hierarchical Summarization for RAG

**Date:** 2025-12-25
**Status:** Research Complete, Ready for Implementation
**Paper:** [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059)
**Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (Stanford/Google)
**Published:** ICLR 2024

---

## 1. Executive Summary

RAPTOR is a retrieval technique that constructs a **hierarchical tree of summaries** from document chunks, enabling retrieval at multiple levels of abstraction. Unlike traditional RAG which retrieves only leaf-level chunks, RAPTOR allows the LLM to access both fine-grained details AND high-level themes in a single query.

**Key Results:**
- **+20% absolute accuracy** on QuALITY benchmark (complex multi-step reasoning)
- **55.7% F1** on QASPER (new SOTA), vs 53.0% for DPR
- **18.5-57%** of retrieved nodes come from non-leaf layers (summaries actually matter!)

**Why This Matters for RAGLab:**
- Current chunking loses document-level context (contextual chunking helps but doesn't create hierarchies)
- Questions about themes, arcs, or multi-section concepts fail because relevant info spans many chunks
- RAPTOR enables answering "What is the main argument of this book?" alongside "What did Sapolsky say about cortisol?"

---

## 2. RAPTOR Algorithm: Complete Technical Specification

### 2.1 Tree Construction Overview

```
+------------------------------------------------------------------+
|                     RAPTOR Tree Structure                         |
+------------------------------------------------------------------+
|                                                                   |
|  Level 3 (Root):      [    Document Summary    ]                  |
|                              ^                                    |
|  Level 2 (Summaries):  [S1]    [S2]    [S3]                       |
|                         ^       ^       ^                         |
|  Level 1 (Clusters):  +-+-+   +-+-+   +-+-+                       |
|                       |   |   |   |   |   |                       |
|  Level 0 (Leaves):   [C1][C2][C3][C4][C5][C6][C7]...              |
|                       ^   ^   ^   ^   ^   ^   ^                   |
|                    Original Document Chunks                       |
+------------------------------------------------------------------+
```

### 2.2 Step-by-Step Tree Building

**Step 1: Initial Chunking**
- Split documents into 100-token chunks (paper default)
- Preserve sentence boundaries
- *RAGLab adaptation:* Use existing 800-token section chunks as leaves (already optimized for our corpus)

**Step 2: Embed All Chunks**
- Embed using SBERT (paper uses `multi-qa-mpnet-base-cos-v1`)
- *RAGLab adaptation:* Use `text-embedding-3-large` (our standard model)

**Step 3: Dimensionality Reduction with UMAP**
- High-dimensional embeddings (1536 dims) are difficult for GMM
- UMAP reduces to ~10 dimensions while preserving structure
- Key parameter: `n_neighbors`
  - **High n_neighbors (15-30):** Global structure, larger clusters
  - **Low n_neighbors (5-10):** Local structure, smaller clusters

```python
# UMAP configuration from paper
from umap import UMAP
reducer = UMAP(
    n_neighbors=10,      # Balance local/global
    n_components=10,     # Target dimensions
    min_dist=0.0,        # Tight clusters
    metric='cosine'      # Match embedding distance
)
reduced_embeddings = reducer.fit_transform(embeddings)
```

**Step 4: Soft Clustering with GMM**
- Gaussian Mixture Models allow chunks to belong to multiple clusters
- This is critical: one chunk about "stress and cortisol" can belong to both "neuroscience" AND "health effects" clusters
- Optimal cluster count determined by Bayesian Information Criterion (BIC)

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def optimal_cluster_count(embeddings, max_clusters=50):
    """Find optimal K using BIC."""
    bics = []
    for k in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))

    # Lower BIC is better; find elbow point
    return np.argmin(bics) + 2

# Soft clustering
gmm = GaussianMixture(n_components=optimal_k)
gmm.fit(reduced_embeddings)
probs = gmm.predict_proba(reduced_embeddings)  # Shape: (n_chunks, n_clusters)
```

**Step 5: Summarization**
- For each cluster, concatenate member chunks
- Generate summary using LLM (paper uses gpt-3.5-turbo)
- Average summary length: 131 tokens (~72% compression)
- Average children per parent: 6.7 chunks

```
System: "You are a Summarizing Text Portal"
User: "Write a summary of the following, including as many
key details as possible: {context}:"
```

**Step 6: Recursive Application**
- Treat summaries as new "chunks"
- Embed summaries using same model
- Repeat clustering -> summarization until tree can't grow
- Stopping criteria: cluster count = 1 OR further clustering infeasible
- Typical depth: 2-4 levels depending on corpus size

### 2.3 Two Retrieval Methods

**Method A: Tree Traversal**
```
1. Embed query
2. At root level: select top-k most similar nodes
3. Descend to children of selected nodes
4. At each level: select top-k from pool
5. Continue for d layers
6. Return all selected nodes (summaries + leaves)
```

- **Pros:** Guaranteed multi-level representation
- **Cons:** Fixed ratio of summary vs detail regardless of query type
- **Paper finding:** Less effective than collapsed tree

**Method B: Collapsed Tree (RECOMMENDED)**
```
1. Flatten all nodes (leaves + all summary levels) into single pool
2. Embed query
3. Rank ALL nodes by cosine similarity
4. Retrieve top nodes until token budget exhausted
```

- **Pros:** Query-adaptive - factual queries get more leaves, thematic queries get more summaries
- **Cons:** May miss relevant leaves if summaries dominate
- **Paper finding:** Consistently outperforms tree traversal

**Configuration used in paper:**
- Token budget: 2000 tokens (~20 nodes)
- UnifiedQA experiments: 400 tokens

### 2.4 Key Parameters Summary

| Parameter | Paper Default | RAGLab Recommendation | Rationale |
|-----------|---------------|--------------------------|-----------|
| Leaf chunk size | 100 tokens | 800 tokens | Our existing chunks are optimized |
| Embedding model | SBERT | text-embedding-3-large | Consistency with pipeline |
| Summarization model | gpt-3.5-turbo | claude-3-haiku | Fast, cheap, sufficient |
| UMAP n_neighbors | 10 | 10-15 | Start with paper default |
| UMAP n_components | 10 | 10 | Standard for GMM input |
| Max clusters | Dynamic (BIC) | Dynamic (BIC) | Corpus-adaptive |
| Summary max tokens | ~150 | 150 | Match paper compression ratio |
| Retrieval method | Collapsed | Collapsed | Paper shows superiority |
| Token budget | 2000 | 2000 | Match existing retrieval limits |

---

## 3. Project-Specific Analysis: RAGLab Integration

### 3.1 Current Architecture Mapping

| RAPTOR Component | RAGLab Equivalent | Integration Strategy |
|------------------|---------------------|---------------------|
| Leaf chunks | `section_chunker.py` output | Use existing 800-token chunks |
| Embedding | `embedder.py` | Reuse `embed_texts()` |
| Vector storage | `weaviate_client.py` | Extend schema for tree metadata |
| Retrieval | `weaviate_query.py` | Modify to query collapsed tree |
| Summarization | N/A | New: use `openrouter_client.py` |
| Clustering | N/A | New: add UMAP + GMM modules |

### 3.2 Intermediate Embedding Storage (Stage 5 Pattern)

RAPTOR must save embeddings to disk following the existing pattern before uploading to Weaviate. This enables:
- Inspecting intermediate results
- Re-uploading to Weaviate without re-embedding
- Debugging tree structure issues

**Storage Location:**
```
data/processed/06_embeddings/raptor/{book}.json
```

**File Format (matches existing pattern):**
```json
{
  "book_id": "Behave, The_Biology of Humans at Our Best Worst (Robert M. Sapolsky)",
  "embedding_model": "text-embedding-3-large",
  "tree_metadata": {
    "total_nodes": 185,
    "leaf_count": 150,
    "summary_count": 35,
    "max_level": 3,
    "build_time_seconds": 120.5
  },
  "chunks": [
    {
      "chunk_id": "Behave::chunk_0",
      "book_id": "Behave...",
      "context": "Behave > Chapter 1 > Introduction",
      "section": "Introduction",
      "text": "...",
      "token_count": 750,
      "chunking_strategy": "raptor",
      "tree_level": 0,
      "is_summary": false,
      "parent_ids": ["Behave::L1_cluster_0"],
      "child_ids": [],
      "cluster_id": "cluster_0",
      "embedding": [0.023, -0.017, ...],
      "embedding_model": "text-embedding-3-large",
      "embedding_dim": 3072
    },
    {
      "chunk_id": "Behave::L1_cluster_0",
      "book_id": "Behave...",
      "context": "Behave > Chapter 1 (Summary)",
      "section": "Chapter 1 Summary",
      "text": "[SUMMARY] This cluster covers the introduction...",
      "token_count": 145,
      "chunking_strategy": "raptor",
      "tree_level": 1,
      "is_summary": true,
      "parent_ids": ["Behave::L2_cluster_0"],
      "child_ids": ["Behave::chunk_0", "Behave::chunk_1", "..."],
      "cluster_id": "L1_cluster_0",
      "source_chunk_ids": ["chunk_0", "chunk_1", "..."],
      "embedding": [...],
      "embedding_model": "text-embedding-3-large",
      "embedding_dim": 3072
    }
  ]
}
```

**Tree-Specific Fields (additions to base chunk schema):**
| Field | Type | Description |
|-------|------|-------------|
| `tree_level` | int | 0=leaf, 1+=summary level |
| `is_summary` | bool | Quick filter for summaries |
| `parent_ids` | List[str] | Parent chunk IDs (for tree traversal) |
| `child_ids` | List[str] | Child chunk IDs (for tree traversal) |
| `cluster_id` | str | Which cluster this node belongs to |
| `source_chunk_ids` | List[str] | (Summaries only) Original chunks summarized |

### 3.3 Weaviate Schema Extension

Current schema properties:
```python
["chunk_id", "book_id", "section", "context", "text",
 "token_count", "chunking_strategy", "embedding_model"]
```

RAPTOR additions needed:
```python
[
    Property(name="tree_level", data_type=DataType.INT),
    Property(name="parent_ids", data_type=DataType.TEXT_ARRAY),
    Property(name="child_ids", data_type=DataType.TEXT_ARRAY),
    Property(name="cluster_id", data_type=DataType.TEXT),
    Property(name="is_summary", data_type=DataType.BOOL),
]
```

### 3.4 Chunk ID Convention for Tree Structure

```
# Leaf chunks (existing format)
{book_id}::chunk_0
{book_id}::chunk_1
...

# Level 1 summaries
{book_id}::L1_cluster_0
{book_id}::L1_cluster_1
...

# Level 2 summaries
{book_id}::L2_cluster_0
...

# Root summary (if single)
{book_id}::root
```

### 3.5 Data Flow Integration

```
RAPTOR Pipeline Flow:
=====================

Stage 4b/4c: RAPTOR Tree Building
  Input:  data/processed/05_final_chunks/section/{book}.json
  Process:
    1. Load section chunks as leaves
    2. Embed leaves using embedder.py
    3. UMAP dimensionality reduction
    4. GMM clustering with BIC
    5. Summarize clusters via LLM
    6. Embed summaries
    7. Repeat steps 3-6 until tree complete
  Output: data/processed/05_final_chunks/raptor/{book}.json (tree structure, no embeddings)

Stage 5: Embedding (--strategy raptor)
  Input:  data/processed/05_final_chunks/raptor/{book}.json
  Process:
    1. Load RAPTOR tree (all nodes: leaves + summaries)
    2. Embed all nodes using embed_texts()
    3. Save with embeddings
  Output: data/processed/06_embeddings/raptor/{book}.json

Stage 6: Weaviate Upload (--strategy raptor)
  Input:  data/processed/06_embeddings/raptor/{book}.json
  Process:
    1. Create collection with extended schema
    2. Upload all nodes (leaves + summaries)
  Output: Weaviate collection: RAG_raptor_embed3large_v1

Stage 7: Retrieval (collapsed tree query)
  Query -> embed -> search all nodes -> return mixed levels
```

### 3.6 File Structure Plan

```
src/rag_pipeline/chunking/
+-- section_chunker.py        # Existing (leaves)
+-- contextual_chunker.py     # Existing
+-- semantic_chunker.py       # Existing
+-- raptor/                   # NEW: RAPTOR module
|   +-- __init__.py
|   +-- tree_builder.py       # Main orchestration
|   +-- clustering.py         # UMAP + GMM logic
|   +-- summarizer.py         # LLM summarization
|   +-- schemas.py            # RaptorNode dataclass
|   +-- raptor_chunker.py     # Strategy interface
+-- strategies.py             # Add 'raptor' strategy
```

---

## 4. Implementation Plan

### Phase 7A: Core RAPTOR Infrastructure (Tree Building)

**Task 7A.1: Dependencies & Schemas**
- [ ] Add to requirements: `umap-learn>=0.5.0` (scikit-learn already present)
- [ ] Create `schemas.py` with `RaptorNode` dataclass:
  ```python
  @dataclass
  class RaptorNode:
      chunk_id: str
      text: str
      tree_level: int  # 0=leaf, 1+=summary
      is_summary: bool
      parent_ids: List[str]
      child_ids: List[str]
      cluster_id: str
      # Inherited from chunk:
      book_id: str
      context: str
      section: str
      token_count: int
      # Summary-specific:
      source_chunk_ids: Optional[List[str]] = None
  ```

**Task 7A.2: Clustering Module** (`clustering.py`)
- [ ] Implement `reduce_dimensions(embeddings, n_neighbors=10, n_components=10)`:
  - Apply UMAP transformation
  - Return reduced embeddings
- [ ] Implement `optimal_cluster_count(embeddings, min_k=2, max_k=50)`:
  - Iterate GMM fits with different K
  - Return K with lowest BIC
- [ ] Implement `soft_cluster(embeddings, n_clusters)`:
  - Fit GMM
  - Return cluster assignments and probabilities
- [ ] Add unit tests with synthetic embeddings

**Task 7A.3: Summarizer Module** (`summarizer.py`)
- [ ] Add `RAPTOR_SUMMARY_PROMPT` to config.py:
  ```python
  RAPTOR_SUMMARY_PROMPT = """Write a comprehensive summary of the following text passages.
  Include as many key details, names, and specific concepts as possible.
  The summary should capture the main ideas while preserving important specifics.

  Passages:
  {context}

  Summary:"""
  ```
- [ ] Add `RAPTOR_SUMMARY_MODEL = "anthropic/claude-3-haiku"` to config.py
- [ ] Implement `summarize_cluster(chunks: List[Dict], model: str) -> str`:
  - Concatenate chunk texts
  - Truncate if exceeds model context
  - Call `call_chat_completion()` with summary prompt
  - Return summary text

**Task 7A.4: Tree Builder** (`tree_builder.py`)
- [ ] Implement `build_raptor_tree(chunks: List[Dict], book_id: str) -> List[RaptorNode]`:
  1. Convert input chunks to RaptorNodes (level 0, is_summary=False)
  2. Embed all nodes using `embed_texts()`
  3. While can_cluster(nodes):
     a. Reduce dimensions with UMAP
     b. Find optimal K with BIC
     c. Cluster with GMM
     d. For each cluster:
        - Collect member nodes
        - Generate summary
        - Create new RaptorNode (level+1, is_summary=True)
        - Update parent_ids on children
        - Embed new summary node
     e. Add summary nodes to tree
  4. Return all nodes (leaves + summaries)

- [ ] Implement `can_cluster(nodes)`:
  - Return False if len(nodes) < 3
  - Return False if all nodes are at max_level
  - Return True otherwise

**Task 7A.5: Strategy Integration** (`raptor_chunker.py`)
- [ ] Implement `run_raptor_chunking(overwrite_context=None) -> Dict[str, int]`:
  1. Load section chunks from `DIR_FINAL_CHUNKS/section/`
  2. For each book:
     a. Build RAPTOR tree
     b. Save to `DIR_FINAL_CHUNKS/raptor/{book}.json`
  3. Return stats: {book: node_count}
- [ ] Add to `strategies.py` registry

### Phase 7B: Embedding Storage (Intermediate Results)

**Task 7B.1: Modify Stage 5 for RAPTOR**
- [ ] Update `run_stage_5_embedding.py` to handle raptor strategy:
  - Load tree structure from `05_final_chunks/raptor/`
  - Embed all nodes (leaves + summaries)
  - Save to `06_embeddings/raptor/{book}.json`
  - Include tree_metadata in output file
- [ ] Ensure tree fields (tree_level, parent_ids, etc.) are preserved

**Task 7B.2: Create run_stage_4b_raptor.py**
- [ ] New stage runner for RAPTOR tree building (between Stage 4 and 5)
- [ ] CLI: `python -m src.stages.run_stage_4b_raptor`
- [ ] Options: `--overwrite`, `--max-levels`, `--min-cluster-size`

### Phase 7C: Weaviate Schema & Upload

**Task 7C.1: Extend Weaviate Schema**
- [ ] Create `create_raptor_collection()` in weaviate_client.py:
  - Base properties + tree properties
  - Ensure backward compatibility
- [ ] Update `upload_embeddings()` to handle tree fields

**Task 7C.2: Config Updates**
- [ ] Add RAPTOR to `AVAILABLE_CHUNKING_STRATEGIES`:
  ```python
  ("raptor", "RAPTOR", "Hierarchical summarization tree (+20% comprehension)"),
  ```
- [ ] Add RAPTOR config parameters:
  ```python
  RAPTOR_SUMMARY_MODEL = "anthropic/claude-3-haiku"
  RAPTOR_MAX_LEVELS = 4
  RAPTOR_MIN_CLUSTER_SIZE = 3
  RAPTOR_UMAP_N_NEIGHBORS = 10
  RAPTOR_UMAP_N_COMPONENTS = 10
  ```
- [ ] Add `StrategyMetadata` for "raptor"

### Phase 7D: Retrieval Integration

**Task 7D.1: Collapsed Tree Query**
- [ ] Modify `query_hybrid()` in weaviate_query.py:
  - Add `include_summaries: bool = True` parameter
  - When True, query all nodes (default for RAPTOR collections)
  - When False, filter `is_summary == false` (leaf-only mode)
- [ ] Scoring works naturally across levels (same embedding space)

**Task 7D.2: Search Result Display**
- [ ] Add tree_level to SearchResult dataclass
- [ ] Update UI to display "[L1 Summary]" or "[Leaf]" badges
- [ ] Show source_chunk_ids for summaries on hover

### Phase 7E: Evaluation & Tuning

**Task 7E.1: Comprehensive Evaluation**
- [ ] Add RAPTOR collection to evaluation grid
- [ ] Run: `python -m src.stages.run_stage_7_evaluation --comprehensive`
- [ ] Compare metrics across: section, contextual, RAPTOR

**Task 7E.2: Parameter Experiments**
- [ ] Test UMAP n_neighbors: [5, 10, 15, 20]
- [ ] Test summary models: haiku vs gemini-flash
- [ ] Test max levels: [2, 3, 4]
- [ ] Document best configuration in evaluation-history.md

---

## 5. Dependencies & Requirements

### New Python Packages
```
umap-learn>=0.5.0     # Dimensionality reduction
# scikit-learn already in project (for GMM)
```

### LLM Cost Estimate (per book)
- Average book: ~150 chunks
- Summaries needed: ~30 (level 1) + ~5 (level 2) + 1 (root) = ~36 summaries
- Tokens per summary call: ~2000 input + 150 output = 2150
- Total tokens: ~77,400 per book
- Cost with claude-3-haiku ($0.25/$1.25 per 1M tokens): ~$0.02 per book
- Full corpus (19 books): ~$0.40 total

### Embedding Cost (per book)
- ~185 nodes (150 leaves + 35 summaries)
- ~150,000 tokens (800 avg * 150 leaves + 130 avg * 35 summaries)
- Cost with text-embedding-3-large ($0.13 per 1M tokens): ~$0.02 per book

### Computational Requirements
- UMAP: CPU-intensive but fast for <1000 chunks (~5 seconds)
- GMM fitting: O(n * k * d) per iteration (~2 seconds)
- LLM summarization: ~36 calls * 2s = ~72 seconds (rate-limited)
- Expected tree build time: **2-3 minutes per book** (dominated by LLM calls)

---

## 6. Open Questions & Design Decisions

### Q1: Should summaries inherit section context?
**Options:**
- A) Summaries get combined context from children (e.g., "Ch1 > Sec1 + Ch1 > Sec2")
- B) Summaries get new context based on content (e.g., "Chapter 1 Summary")
- C) Summaries have no context (rely on text)

**Recommendation:** Option B - generate descriptive context like "Chapter 1 Summary" or "Neuroscience Cluster"

### Q2: How to handle soft clustering (multi-cluster membership)?
**Options:**
- A) Hard assignment to highest probability cluster only
- B) Duplicate chunks in multiple clusters (increases summary redundancy)
- C) Threshold-based: include in cluster if P > 0.3

**Recommendation:** Option A for initial implementation; Option C for future improvement

### Q3: Should we build per-book trees or cross-book trees?
**Options:**
- A) Per-book trees (simpler, matches current structure)
- B) Cross-book thematic trees (complex, may improve cross-reference queries)

**Recommendation:** Option A - per-book is sufficient for our corpus size

### Q4: Integration with existing strategies?
**Options:**
- A) RAPTOR as standalone strategy (uses section chunks as input)
- B) RAPTOR on top of contextual chunks (contextual leaves + RAPTOR hierarchy)

**Recommendation:** Option A initially; Option B as future enhancement (RAPTOR + Contextual = maximum context)

### Q5: When to embed during tree building?
**Options:**
- A) Embed all at once at end (faster, single batch)
- B) Embed incrementally per level (needed for clustering)

**Decision:** Option B is required - we need embeddings for clustering at each level. Final save includes all embeddings.

---

## 7. References

### Primary Sources
- [RAPTOR Paper (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059) - Original research
- [Official GitHub](https://github.com/parthsarthi03/raptor) - Reference implementation
- [OpenReview](https://openreview.net/forum?id=GN921JHCRw) - ICLR 2024 reviews

### Implementation Tutorials
- [VectorHub: Improving RAG with RAPTOR](https://superlinked.com/vectorhub/articles/improve-rag-with-raptor) - Detailed walkthrough
- [VelociRAPTOR](https://github.com/satvshr/VelociRAPTOR) - From-scratch NumPy implementation
- [LangChain RAPTOR Discussion](https://github.com/langchain-ai/langchain/discussions/18621) - Integration patterns

### Related Techniques
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) - Complementary approach
- [HyDE (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496) - Query-side enhancement
- [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130) - Alternative hierarchy via knowledge graphs

---

## 8. Success Criteria

RAPTOR implementation is successful when:

1. **Tree Construction Works:**
   - Builds 2-4 level trees from section chunks
   - Summaries are coherent and capture cluster themes
   - Tree structure is saved to `05_final_chunks/raptor/`

2. **Embedding Storage Works:**
   - All nodes (leaves + summaries) saved to `06_embeddings/raptor/`
   - Tree metadata preserved (levels, parent/child relationships)
   - File format matches existing embedding pattern

3. **Weaviate Integration Works:**
   - Extended schema accepts tree fields
   - Collection named: `RAG_raptor_embed3large_v1`
   - All nodes searchable via hybrid query

4. **Retrieval Improves:**
   - Mixed-level retrieval returns both summaries and leaves
   - Thematic questions get summary nodes
   - Factual questions still get relevant leaves

5. **Metrics Improve:**
   - Answer relevancy >= contextual strategy
   - Context precision improves for multi-hop questions
   - Faithfulness maintained (no hallucination from summaries)

6. **Integration Complete:**
   - `python -m src.stages.run_stage_4b_raptor` works
   - `--strategy raptor` works in Stage 5 and Stage 6
   - RAPTOR appears in evaluation grid

---

*Last Updated: 2025-12-25*
