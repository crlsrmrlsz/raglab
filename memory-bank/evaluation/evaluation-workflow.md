# Evaluation Workflow Analysis

This document provides a comprehensive analysis of the RAGLab evaluation system, including strategy taxonomy, retrieval diagrams, and the black-box evaluation philosophy.

## Strategy Taxonomy

The RAG pipeline uses two independent axes of strategies:

### 1. Chunking Strategies (Stage 4 - Index Time)

Chunking strategies determine how documents are split and stored in Weaviate. Each creates a separate collection.

| Strategy | Collection Pattern | Description | Research |
|----------|-------------------|-------------|----------|
| **section** | `RAG_section_*` | Sequential chunking with 2-sentence overlap | Baseline |
| **contextual** | `RAG_contextual_*` | LLM-generated context prepended to chunks | [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) |
| **raptor** | `RAG_raptor_*` | Hierarchical tree with GMM clustering + summaries | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) |

### 2. Search Types (Weaviate Query Method)

Search types determine HOW chunks are retrieved from Weaviate. Orthogonal to preprocessing.

| Search Type | Method | Alpha | Description |
|-------------|--------|-------|-------------|
| **keyword** | BM25 only | N/A | Pure keyword matching, no embeddings |
| **hybrid** | Vector + BM25 | 0.5, 1.0 | Combines semantic similarity with keyword matching |

### 3. Preprocessing Strategies (Query Transformation)

Preprocessing strategies transform queries before retrieval. They work with any search type.

| Strategy | Transform | Retrieval | Research |
|----------|-----------|-----------|----------|
| **none** | Query unchanged | Single search | Baseline |
| **hyde** | Hypothetical answer | Search with HyDE passage | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| **decomposition** | 3-4 sub-questions | Multi-query + RRF merge | [arXiv:2507.00355](https://arxiv.org/abs/2507.00355) |
| **graphrag** | Entity extraction | Vector + Neo4j graph hybrid | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |

## Comprehensive Evaluation Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│            COMPREHENSIVE EVALUATION (5D Grid Search)                  │
│            run_stage_7_evaluation.py --comprehensive                  │
└───────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FOR EACH COMBINATION:                                                │
│                                                                       │
│  Collections × Search Types × Alphas × Strategies × Top-K             │
│       │             │           │          │           │              │
│       │             │           │          │           └── [10, 20]   │
│       │             │           │          └── [none, hyde, decomp,   │
│       │             │           │               graphrag]             │
│       │             │           └── [0.5, 1.0] (hybrid only)          │
│       │             └── [keyword, hybrid]                             │
│       └── [section, contextual, semantic, raptor]                     │
│                                                                       │
│  Total: ~102 valid combinations (after compatibility filtering)       │
│  Questions: 15 curated (5 single-concept + 10 cross-domain)           │
│                                                                       │
│  Note: For keyword search, alpha is N/A (pure BM25)                   │
│        For hybrid search, alpha values [0.5, 1.0] are tested          │
└───────────────────────────────────────────────────────────────────────┘
```

## Strategy-Specific Retrieval Paths

### None Strategy (Baseline)

```
Question ──► No preprocessing ──► Hybrid Search ──► Contexts
                                       │
                                       ▼
                             Weaviate (BM25 + Vector)
                                       │
                                       ▼
                                Optional Reranking
                                       │
                                       ▼
                                  Top-K Results
```

### HyDE Strategy (Hypothetical Document Embeddings)

```
Question ──► LLM generates hypothetical answer ──► Hybrid Search ──► Contexts
                        │                               │
                        ▼                               ▼
           "Procrastination stems from..."    Weaviate (BM25 + Vector)
                                                       │
                                                       ▼
                                              Optional Reranking
                                                       │
                                                       ▼
                                                  Top-K Results
```

**Theory**: HyDE bridges the semantic gap between questions and documents by searching for passages similar to a plausible answer rather than the question itself.

### Decomposition Strategy (Multi-Query + RRF)

```
Question ──► decompose_query() ──► [sub_q1, sub_q2, sub_q3, sub_q4]
                                            │
                                            ▼
                                   ┌───────────────────┐
                                   │ Search sub_q1 ────┼──► Results1
                                   │ Search sub_q2 ────┼──► Results2
                                   │ Search sub_q3 ────┼──► Results3
                                   │ Search sub_q4 ────┼──► Results4
                                   └───────────────────┘
                                            │
                                            ▼
                                      RRF Merge
                                   (k=60 formula)
                                            │
                                            ▼
                                    Optional Reranking
                                            │
                                            ▼
                                      Top-K Contexts
```

**RRF Formula**:
```
RRF_score(d) = sum(1 / (k + rank(d, q))) for each query q
```
where k=60 (standard from literature).

**Theory**: Complex questions are decomposed into simpler sub-questions. Results from each sub-query are merged, with documents appearing in multiple result lists receiving boosted scores.

### GraphRAG Strategy (Hybrid Graph + Vector)

```
Question ──► extract_query_entities() ──► [entity1, entity2, ...]
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
       Vector Search      Neo4j Graph
       (Weaviate)         Traversal
            │                   │
            │                   ▼
            │          Entity → Neighbors → chunk_ids
            │                   │
            └─────────┬─────────┘
                      ▼
               Graph Boost Merge
          (graph-found chunks ranked higher)
                      │
                      ▼
              Optional Reranking
                      │
                      ▼
                Top-K Contexts
```

**Theory**: GraphRAG combines dense vector retrieval with knowledge graph traversal. Chunks that are both semantically similar AND connected via entity relationships receive boosted ranking.

## Black-Box Evaluation Philosophy

Each strategy is evaluated as a **black box**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      BLACK BOX STRATEGY                         │
│                                                                 │
│   Questions ───────────►  [Strategy]  ───────────► Contexts    │
│      (input)                                        (output)    │
│                                                                 │
│   The evaluation ONLY measures:                                 │
│   - Quality of retrieved contexts (context_precision)          │
│   - Faithfulness of generated answers                          │
│   - Relevancy of answers to questions                          │
│                                                                 │
│   The evaluation does NOT care about:                          │
│   - How the strategy works internally                          │
│   - Number of LLM calls                                        │
│   - Latency or cost                                            │
└─────────────────────────────────────────────────────────────────┘
```

This allows fair comparison between fundamentally different strategies (e.g., single-query hyde vs multi-query decomposition vs graph-based graphrag).

## Strategy Comparison Table

| Aspect | none | hyde | decomposition | graphrag |
|--------|------|------|---------------|----------|
| **LLM Calls** | 0 | 1 | 1 | 1 |
| **Search Queries** | 1 | 1 | 3-4 | 1 + graph |
| **Merge Strategy** | N/A | N/A | RRF | Graph boost |
| **External DB** | Weaviate | Weaviate | Weaviate | Weaviate + Neo4j |
| **Best For** | Simple queries | Semantic matching | Multi-aspect questions | Entity-centric queries |

## RAPTOR as Chunking Strategy

RAPTOR is a **chunking strategy**, not a preprocessing strategy. It creates hierarchical summaries at index time:

```
Original Chunks (Level 0)
        │
        ▼
   GMM Clustering
        │
        ▼
   LLM Summarization ──► Level 1 Summaries
        │
        ▼
   GMM Clustering
        │
        ▼
   LLM Summarization ──► Level 2 Summaries
        │
        ... (up to 4 levels)
```

All nodes (leaves + summaries) are stored flat in Weaviate, enabling "collapsed tree" retrieval where both detailed and thematic content can be retrieved.

RAPTOR collections are tested via the collection axis (e.g., `RAG_raptor_embed3large_v1`), not the preprocessing strategy axis.

## Implementation Details

### File: `src/evaluation/ragas_evaluator.py`

The `retrieve_contexts()` function implements strategy-aware retrieval with search type:

```python
def retrieve_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessed: Optional[PreprocessedQuery] = None,  # Strategy-aware
    search_type: str = "hybrid",  # "keyword" or "hybrid"
) -> List[str]:
```

Routing logic:
1. **search_type** determines Weaviate query method (BM25 vs hybrid)
2. If `preprocessed.strategy_used == "decomposition"`: Execute multi-query RRF
3. If `preprocessed.strategy_used == "graphrag"`: Execute Neo4j hybrid retrieval
4. Otherwise: Standard search with configured search_type

### File: `src/stages/run_stage_7_evaluation.py`

Comprehensive mode iterates through all combinations (5D grid):

```python
for collection in collections:                    # Chunking strategies
    for search_type in ["keyword", "hybrid"]:     # Search method
        if search_type == "keyword":
            alphas = [0.0]                        # Placeholder (ignored for BM25)
        else:
            alphas = [0.5, 1.0]                   # Hybrid balance values
        for alpha in alphas:
            for strategy in strategies:           # Preprocessing strategies
                for top_k in [10, 20]:            # Retrieval depth (innermost for caching)
                    run_evaluation(..., search_type=search_type)
```

Note: `top_k` is innermost loop to maximize retrieval cache hits (see Design Decisions).

## Output

See [docs/evaluation/README.md](../../docs/evaluation/README.md) for output file locations and metrics reference.

## Data Flow Diagram

```
Test Questions (JSON)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   run_stage_7_evaluation.py                 │
│                                                             │
│  STANDARD MODE              COMPREHENSIVE MODE (4D Grid)    │
│  ─────────────              ─────────────────────────────   │
│  Single config              For each combination:           │
│  → eval_*.json              collections × alphas ×          │
│  → trace_*.json             top_k × strategies              │
│                             → comprehensive_*.json          │
│                             → checkpoint (crash recovery)   │
│                             → failed_combinations.json      │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 5D Evaluation Grid with search_type (Jan 2025)
Added `search_type` as a separate dimension from preprocessing strategies:
- **search_type**: How Weaviate queries ("keyword" BM25 or "hybrid" vector+BM25)
- **preprocessing**: Query transformation (none, hyde, decomposition, graphrag)

This provides clearer architecture: preprocessing transforms queries, search_type determines retrieval method. Any preprocessing strategy can work with any search type.

### 4D Evaluation Grid (Dec 2024)
Added `top_k [10, 20]` as 4th dimension. Retrieval depth significantly affects precision/recall tradeoff - more chunks increase recall but may dilute precision.

### Trace Persistence (Dec 2024)
Save `QuestionTrace` to JSON for each question. Enables:
- Metric recalculation without re-running expensive retrieval/generation
- Debugging specific question failures
- Historical comparison across runs

### Retrieval Caching (Dec 2024)
Cache key: `(question, collection, search_type, alpha, strategy)` - excludes `top_k`.
Retrieve once at `max(top_k)`, slice for smaller values. Halves API calls during grid search.

### Retry with Exponential Backoff (Dec 2024)
Max 3 retries, base delay 2.0s. RAGAS metrics use LLM calls that hit rate limits during 100+ combination grid search.

### Metrics Selection (Dec 2024)
Use 5 native RAGAS metrics only:
- Removed `composite_score` (not a RAGAS metric, hid individual weaknesses)
- Removed `squad_f1` (token-based, no semantic understanding)
- Added `context_recall` (measures if retrieval missed relevant info)

## References

- [RAPTOR Paper](https://arxiv.org/abs/2401.18059) - Hierarchical summarization
- [HyDE Paper](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [Query Decomposition](https://arxiv.org/abs/2507.00355) - Multi-hop retrieval
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130) - Knowledge graph + vector hybrid
- [RAGAS Framework](https://docs.ragas.io/) - LLM-as-judge evaluation
