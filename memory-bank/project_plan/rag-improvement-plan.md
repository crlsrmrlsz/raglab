# RAG Improvement Plan: Comprehensive Quality Enhancement

## Overview

This plan implements three major RAG improvements (Contextual Embeddings, RAPTOR, GraphRAG) plus supporting infrastructure for UI-based testing and automatic RAGAS evaluation logging.

**Goal**: Improve answer quality through isolated, measurable experiments with easy A/B testing from the UI.

**Current Best Metrics** (Run 3):
- Relevancy: 0.786
- Faithfulness: 0.885
- Failures: 0/23 (0%)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
├─────────────────────────────────────────────────────────────────┤
│  Sidebar                    │  Main Area                        │
│  ┌───────────────────────┐  │  ┌─────────────────────────────┐ │
│  │ Collection Selector   │  │  │ Tabs: Answer | Pipeline |   │ │
│  │ ├─ RAG_section800_v1  │  │  │       Chunks               │ │
│  │ ├─ RAG_contextual_v1  │  │  └─────────────────────────────┘ │
│  │ ├─ RAG_raptor_v1      │  │                                   │
│  │ └─ RAG_graphrag_v1    │  │  (Evaluation runs via CLI:        │
│  │                       │  │   python -m src.stages.run_stage_7_...)  │
│  │ Stage 1-4 Controls    │  │                                   │
│  │ (existing)            │  │                                   │
│  └───────────────────────┘  │                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE STAGES                            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Chunking          │  Stage 5: Embedding               │
│  ├─ naive_chunker.py        │  ├─ embedder.py                │
│  ├─ contextual_chunker.py   │  └─ (same for all strategies)     │
│  ├─ raptor_chunker.py       │                                   │
│  └─ (run outside UI)        │  Stage 6: Weaviate                │
│                             │  ├─ weaviate_client.py            │
│                             │  └─ Multiple collections          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GRAPHRAG (Neo4j)                           │
├─────────────────────────────────────────────────────────────────┤
│  Entity Extraction → Graph Construction → Graph + Vector Search │
│  ├─ src/graph/extractor.py                                      │
│  ├─ src/graph/neo4j_client.py                                   │
│  └─ src/graph/query.py                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: UI Foundation

**No UI changes needed!**

### Collection Selector

**Already implemented** at `src/ui/app.py:220-235`:
- Lists all `RAG_*` collections from Weaviate via `list_collections()`
- Dropdown at top of sidebar
- Selected collection passed to `search_chunks()` via `collection_name` parameter

New collections (contextual, raptor, graphrag) will automatically appear in the dropdown after running Stage 6.

### Evaluation (CLI Only)

Evaluation stays outside the UI for simplicity. Improvements needed for `src/run_stage_7_evaluation.py`:

**Current CLI arguments (already implemented):**
```bash
python -m src.stages.run_stage_7_evaluation [OPTIONS]

Options:
  -n, --questions N         Limit to first N questions
  -m, --metrics METRICS     Metrics to compute (default: faithfulness relevancy context_precision)
  -k, --top-k K             Chunks to retrieve (default: 10)
  --generation-model MODEL  Answer generation model (default: openai/gpt-5-mini)
  --evaluation-model MODEL  RAGAS judge model (default: anthropic/claude-haiku-4.5)
  -a, --alpha ALPHA         Hybrid search: 0.0=keyword, 0.5=balanced, 1.0=vector
  --reranking/--no-reranking  Enable/disable cross-encoder
  -o, --output PATH         Output file path
```

**To add:**
1. `--collection` argument to select which Weaviate collection (e.g., `RAG_contextual_v1`)
2. Auto-append results to `memory-bank/evaluation-history.md`
3. Update `data/evaluation/tracking.json` with run config

**Implementation:**

```python
# Add to argparse:
parser.add_argument(
    "--collection",
    type=str,
    default=None,
    help="Weaviate collection to evaluate (default: auto from config)",
)

# Add auto-logging function:
def append_to_evaluation_history(results, config, output_path):
    """Append run summary to memory-bank/evaluation-history.md"""
    history_path = Path("memory-bank/evaluation-history.md")

    # Get next run number
    run_number = get_next_run_number(history_path)

    # Format markdown
    entry = f"""
---

## Run {run_number}: {config['collection']}

**Date:** {datetime.now().strftime('%B %d, %Y')}
**File:** `{output_path.relative_to(Path.cwd())}`

### Configuration
- **Collection:** {config['collection']}
- **Search Type:** Hybrid
- **Alpha:** {config['alpha']}
- **Top-K:** {config['top_k']}
- **Reranking:** {'Yes' if config['reranking'] else 'No'}
- **Generation Model:** {config['generation_model']}
- **Evaluation Model:** {config['evaluation_model']}

### Results
| Metric | Score |
|--------|-------|
| Faithfulness | {results['scores'].get('faithfulness', 'N/A'):.3f} |
| Relevancy | {results['scores'].get('relevancy', 'N/A'):.3f} |
| Context Precision | {results['scores'].get('context_precision', 'N/A'):.3f} |

### Key Learning
[Add notes about this run manually]
"""

    with open(history_path, "a") as f:
        f.write(entry)

    logger.info(f"Appended to evaluation-history.md as Run {run_number}")
```

**Example usage after improvements:**
```bash
# Test contextual embeddings collection
python -m src.stages.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1

# Test RAPTOR with different alpha
python -m src.stages.run_stage_7_evaluation --collection RAG_raptor_embed3large_v1 --alpha 0.7

# Compare collections
python -m src.stages.run_stage_7_evaluation --collection RAG_section800_embed3large_v1
python -m src.stages.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1
```

---

## Phase 1: Contextual Retrieval (Anthropic-style)

**Expected Impact**: +35% reduction in retrieval failures (per Anthropic research)

**Concept**: Before embedding each chunk, prepend an LLM-generated context summary that describes where the chunk fits in the document.

### 1.1 Create Contextual Chunker

**File**: `src/rag_pipeline/chunking/contextual_chunker.py` (IMPLEMENTED)

```python
def create_contextual_chunks(chunks: List[Dict], book_text: str) -> List[Dict]:
    """Add contextual prefix to each chunk before embedding."""
    for chunk in chunks:
        context_prompt = f"""
Document: {chunk['book_id']}
Section: {chunk['context']}

Here is the chunk:
{chunk['text']}

Provide a brief (2-3 sentence) context that situates this chunk
within the document. Start with "This passage..."
"""
        context = call_openrouter_chat(context_prompt, model=PREPROCESSING_MODEL)
        chunk['contextual_text'] = f"{context}\n\n{chunk['text']}"
        chunk['context_summary'] = context
    return chunks
```

### 1.2 New Stage 4 Variant

**Stage runner**: `python -m src.stages.run_stage_4_chunking --strategy contextual` (IMPLEMENTED)

Run via: `python -m src.stages.run_stage_4_chunking --strategy contextual`

### 1.3 Embedding & Upload

- Run `python -m src.stages.run_stage_5_embedding` (reads from contextual/)
- Run `python -m src.stages.run_stage_6_weaviate` with `CHUNKING_STRATEGY_NAME=contextual`
- Creates collection: `RAG_contextual_embed3large_v1`

### 1.4 Test from UI

1. Select `RAG_contextual_embed3large_v1` in collection dropdown
2. Run same queries as baseline
3. Trigger RAGAS evaluation
4. Compare metrics to Run 3 baseline

---

## Phase 2: RAPTOR (Hierarchical Summarization)

**Expected Impact**: +20% improvement on long-document comprehension

**Concept**: Build a tree where leaves are chunks, nodes are LLM summaries of clustered chunks. Retrieval can match at any level.

### 2.1 RAPTOR Architecture

```
Level 3:  [Book Summary]
              │
Level 2:  [Theme A] ─── [Theme B] ─── [Theme C]
              │              │             │
Level 1:  [Sec1][Sec2]   [Sec3][Sec4]  [Sec5][Sec6]
              │              │             │
Level 0:  [c1][c2][c3]   [c4][c5]      [c6][c7][c8]  ← Original chunks
```

### 2.2 Create RAPTOR Chunker

**File**: `src/rag_pipeline/chunking/raptor/raptor_chunker.py` (IMPLEMENTED)

```python
def build_raptor_tree(chunks: List[Dict], max_levels: int = 3) -> List[Dict]:
    """Build hierarchical summary tree from chunks."""
    all_nodes = []
    current_level = chunks.copy()

    for level in range(max_levels):
        # 1. Embed current level
        embeddings = embed_texts([c['text'] for c in current_level])

        # 2. Cluster by semantic similarity (k-means or HDBSCAN)
        clusters = cluster_embeddings(embeddings, target_clusters=len(current_level)//3)

        # 3. Generate summary for each cluster
        summaries = []
        for cluster_chunks in clusters:
            combined_text = "\n\n".join([c['text'] for c in cluster_chunks])
            summary = generate_cluster_summary(combined_text)
            summaries.append({
                'text': summary,
                'level': level + 1,
                'children': [c['chunk_id'] for c in cluster_chunks],
                'chunk_id': f"raptor_L{level+1}_{uuid4().hex[:8]}"
            })

        all_nodes.extend(current_level)
        current_level = summaries

        if len(current_level) <= 1:
            break

    all_nodes.extend(current_level)  # Add top-level summaries
    return all_nodes
```

### 2.3 RAPTOR Collection Schema

Weaviate schema additions:
- `raptor_level`: int (0=leaf, 1+=summary)
- `children`: text[] (child chunk_ids)
- `parent`: text (parent chunk_id)

### 2.4 RAPTOR Query Strategy

```python
def query_raptor(query: str, collection: str, top_k: int = 10):
    # 1. Search across all levels
    results = query_hybrid(query, top_k=top_k*2, collection=collection)

    # 2. For each summary node, optionally fetch children
    expanded = []
    for r in results:
        if r.raptor_level > 0 and r.score > 0.7:
            # High-scoring summary: fetch children for detail
            children = fetch_children(r.children)
            expanded.extend(children)
        else:
            expanded.append(r)

    # 3. Deduplicate and re-rank
    return deduplicate_and_rank(expanded)[:top_k]
```

### 2.5 New Stage 4 Variant

**Stage runner**: `python -m src.stages.run_stage_4_5_raptor` (IMPLEMENTED)

Run via: `python -m src.stages.run_stage_4_5_raptor`

### 2.6 Test from UI

1. Select `RAG_raptor_embed3large_v1`
2. Test queries requiring synthesis (cross-domain questions)
3. Run RAGAS evaluation
4. Compare to baseline and contextual

---

## Phase 3: GraphRAG (Neo4j Integration)

**Expected Impact**: +70-80% win rate on comprehensiveness vs baseline (Microsoft research)

**Concept**: Extract entities and relationships, build knowledge graph in Neo4j, combine graph traversal with vector search.

### 3.1 Neo4j Setup

```bash
# Docker Compose addition
services:
  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
```

### 3.2 Entity Extraction

**New file**: `src/graph/extractor.py`

```python
def extract_entities_and_relations(chunk: Dict) -> Dict:
    """Extract entities and relationships using LLM."""
    prompt = f"""
Extract entities and relationships from this text.

Text: {chunk['text']}

Return JSON:
{{
  "entities": [
    {{"name": "...", "type": "CONCEPT|PERSON|THEORY|BRAIN_REGION|..."}}
  ],
  "relationships": [
    {{"source": "...", "relation": "DEFINES|ARGUES|RELATES_TO|...", "target": "..."}}
  ]
}}
"""
    return call_openrouter_json(prompt, model=PREPROCESSING_MODEL)
```

### 3.3 Neo4j Client

**New file**: `src/graph/neo4j_client.py`

```python
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_entity(self, name: str, entity_type: str, chunk_ids: List[str]):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type, e.chunk_ids = $chunk_ids
        RETURN e
        """
        # Execute query

    def create_relationship(self, source: str, relation: str, target: str):
        query = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATES {type: $relation}]->(b)
        RETURN r
        """
        # Execute query

    def query_neighborhood(self, entity: str, hops: int = 2) -> List[str]:
        """Get chunk_ids from entity neighborhood."""
        query = f"""
        MATCH (e:Entity {{name: $entity}})-[*1..{hops}]-(related)
        RETURN DISTINCT related.chunk_ids as chunks
        """
        # Return merged chunk_ids
```

### 3.4 Graph-Augmented Query

**New file**: `src/graph/query.py`

```python
def query_graphrag(query: str, top_k: int = 10) -> List[SearchResult]:
    """Combine graph traversal with vector search."""
    # 1. Extract entities from query
    query_entities = extract_entities_from_query(query)

    # 2. Get chunk_ids from graph neighborhood
    graph_chunks = set()
    for entity in query_entities:
        neighborhood = neo4j_client.query_neighborhood(entity, hops=2)
        graph_chunks.update(neighborhood)

    # 3. Vector search for semantic matches
    vector_results = query_hybrid(query, top_k=top_k*2)

    # 4. Boost chunks found in graph
    for result in vector_results:
        if result.chunk_id in graph_chunks:
            result.score *= 1.3  # Graph boost

    # 5. Re-rank and return
    return sorted(vector_results, key=lambda x: x.score, reverse=True)[:top_k]
```

### 3.5 GraphRAG Pipeline Stages

**New files**:
- `src/run_stage_4b_graph_extract.py` - Extract entities/relations from chunks
- `src/run_stage_6b_neo4j.py` - Upload graph to Neo4j

### 3.6 UI Integration

Add search mode toggle:
- Vector Only (existing)
- Hybrid (existing)
- GraphRAG (new) - Uses `query_graphrag()`

---

## Phase 4: Query Decomposition

**Expected Impact**: +36.7% MRR@10 improvement

**Concept**: Break complex queries into sub-questions, retrieve for each, merge results.

### 5.1 Implement Decomposition

**File**: `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` (IMPLEMENTED)

```python
def decompose_query(query: str, model: Optional[str] = None) -> List[str]:
    """Decompose multi-hop query into sub-questions."""
    prompt = f"""
Break this complex question into 2-4 simpler sub-questions that can be answered independently:

Question: {query}

Return JSON array of sub-questions:
["sub-question 1", "sub-question 2", ...]
"""
    result = call_openrouter_json(prompt, model=model or PREPROCESSING_MODEL)
    return result  # List[str]
```

### 5.2 Multi-Query Retrieval (ARCHIVED - Dec 23)

> **Note:** This section is archived. Multi-query was subsumed by the Query Decomposition strategy, which uses RRF merging for sub-queries. See Phase 4 for the implemented version.

```python
# ARCHIVED CODE - for historical reference only
def retrieve_multi_hop(query: str, sub_queries: List[str], top_k: int) -> List[SearchResult]:
    """Retrieve for each sub-query, merge with RRF."""
    all_results = {}

    for sub_q in sub_queries:
        results = query_hybrid(sub_q, top_k=top_k)
        for rank, r in enumerate(results):
            if r.chunk_id not in all_results:
                all_results[r.chunk_id] = {'result': r, 'rrf_score': 0}
            all_results[r.chunk_id]['rrf_score'] += 1 / (60 + rank)  # RRF formula

    # Sort by RRF score
    merged = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)
    return [m['result'] for m in merged][:top_k]
```

### 5.3 Update Preprocessing Flow

```python
elif query_type == QueryType.MULTI_HOP:
    sub_queries = decompose_query(query, model=model)
    search_query = query  # Keep original for display
    # Store sub_queries in PreprocessedQuery for retrieval stage
```

---

## Phase 5: Alpha Tuning Experiments

Run via CLI to find optimal hybrid search balance:

```bash
python -m src.stages.run_stage_7_evaluation --alpha 0.3  # Keyword-heavy (philosophy)
python -m src.stages.run_stage_7_evaluation --alpha 0.5  # Balanced (default)
python -m src.stages.run_stage_7_evaluation --alpha 0.7  # Vector-heavy (conceptual)
```

Update `evaluation-history.md` with results after each run.

---

## Implementation Order

**Strategy**: Test all query preprocessing techniques BEFORE chunking/embedding changes. Preprocessing changes are lower risk (prompt-only in many cases) and faster to test than chunking changes (which require re-embedding entire collections).

| Order | Phase | Status | Effort | Impact | Files |
|-------|-------|--------|--------|--------|-------|
| 0 | Evaluation CLI improvements | DONE | Low | Enables A/B testing | `run_stage_7_evaluation.py` |
| 1 | Preprocessing Strategy Infrastructure | DONE | Medium | Enables A/B testing | `strategies.py`, `config.py`, UI, CLI |
| 2 | HyDE Strategy | DONE (Dec 23) | Low | Better retrieval | `strategies.py` |
| 3 | Multi-Query Strategy | REMOVED (Dec 23) | - | Subsumed by decomposition | - |
| 4 | Implement Query Decomposition | DONE | Medium | +36.7% MRR | `strategies.py`, `rrf.py` |
| 5 | Alpha tuning (Comprehensive eval) | DONE (Dec 24) | Low | Grid search | CLI only |
| 6 | Contextual Chunking | DONE (Dec 22) | Medium | +35% failures | `contextual_chunker.py` |
| 7 | RAPTOR | DONE (Dec 25) | High | +20% comprehension | `raptor/` module |
| 8 | GraphRAG | DONE (Dec 25) | High | +70% coverage | `graph/`, Neo4j |

**Preprocessing Strategies Testing Workflow**:
```bash
# Test each strategy with default hybrid search
python -m src.stages.run_stage_7_evaluation --preprocessing none          # No transformation
python -m src.stages.run_stage_7_evaluation --preprocessing hyde          # HyDE (hypothetical answer)
python -m src.stages.run_stage_7_evaluation --preprocessing decomposition # Sub-questions + RRF merge

# Test with keyword (BM25) search
python -m src.stages.run_stage_7_evaluation --search-type keyword --preprocessing hyde

# Test different alpha values with hybrid search
python -m src.stages.run_stage_7_evaluation --search-type hybrid --alpha 0.5 --preprocessing hyde
python -m src.stages.run_stage_7_evaluation --search-type hybrid --alpha 1.0 --preprocessing hyde

# Comprehensive grid search (all combinations)
python -m src.stages.run_stage_7_evaluation --comprehensive

# Compare results in memory-bank/evaluation-history.md
```

**Note**: HyDE (arXiv:2212.10496) replaced step-back prompting (Dec 23, 2024). See `memory-bank/step-back-prompting-research.md` for historical analysis.

### GraphRAG Execution (Updated Jan 2026)

**Simplified pipeline:** Entity extraction now uses curated types from `src/graph/graphrag_types.yaml` (33 types for dual-domain corpus).

**Execution order:**
```bash
# Extract entities + relationships
python -m src.stages.run_stage_4_5_graph_extract

# Upload to Neo4j + run Leiden
docker compose up -d neo4j
python -m src.stages.run_stage_6b_neo4j
```

**Note:** Stage 4.5 RAPTOR (`run_stage_4_5_raptor.py`) is completely separate from GraphRAG - it creates hierarchical document summaries, not entity extraction.

### GraphRAG Chunk ID Compatibility (Dec 28, 2024)

**Critical discovery:** GraphRAG only works with certain collections due to chunk ID coupling:

| Collection | GraphRAG Compatible? | Reason |
|------------|---------------------|--------|
| section | ✅ Yes | Chunk IDs match extraction source |
| contextual | ✅ Yes | Preserves section chunk IDs |
| semantic | ❌ No | Different chunk boundaries = different IDs |
| raptor | ⚠️ Partial | Only leaf chunks match |

**Implementation:**
- `src/config.py` has `PREPROCESSING_COMPATIBILITY` dict mapping valid strategies per collection
- UI filters preprocessing dropdown based on selected collection (wizard-style)
- Comprehensive evaluation (`--comprehensive`) skips invalid combinations

**Valid Combinations for Evaluation:**

The evaluation grid now has 5 dimensions: Collections × Search Types × Alphas × Strategies × Top-K

| Collection | Search Types | Alphas (hybrid only) | Strategies | GraphRAG? |
|------------|--------------|---------------------|------------|-----------|
| section | keyword, hybrid | 0.5, 1.0 | none, hyde, decomposition, graphrag | ✅ Yes |
| contextual | keyword, hybrid | 0.5, 1.0 | none, hyde, decomposition, graphrag | ✅ Yes |
| semantic | keyword, hybrid | 0.5, 1.0 | none, hyde, decomposition | ❌ No |
| raptor | keyword, hybrid | 0.5, 1.0 | none, hyde, decomposition | ❌ No |

**Total combinations:** ~102 valid (51 base × 2 top_k values)
- Keyword search: 17 base combinations (1 alpha placeholder per collection)
- Hybrid search: 34 base combinations (2 alpha values per collection)

---

## Testing Protocol

For each improvement:

1. **Create new collection** (run Stage 4-6 via CLI)
2. **Select collection** in UI dropdown
3. **Run manual queries** in UI to verify basic functionality
4. **Run RAGAS evaluation** via CLI: `python -m src.stages.run_stage_7_evaluation`
5. **Update evaluation-history.md** with results
6. **Compare to baseline** (Run 3: 0.786 relevancy, 0.885 faithfulness)

---

## Implemented Files (All Complete)

### Chunking Strategies
- `src/rag_pipeline/chunking/strategies.py` - Strategy registry
- `src/rag_pipeline/chunking/section_chunker.py` - Section strategy (baseline)
- `src/rag_pipeline/chunking/semantic_chunker.py` - Semantic strategy
- `src/rag_pipeline/chunking/contextual_chunker.py` - Contextual strategy (Anthropic-style)
- `src/rag_pipeline/chunking/raptor/` - RAPTOR module (tree_builder.py, clustering.py, summarizer.py)

### Preprocessing Strategies
- `src/rag_pipeline/retrieval/preprocessing/strategies.py` - Strategy registry
- `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` - Dispatcher
- `src/rag_pipeline/retrieval/rrf.py` - RRF merging for decomposition/graphrag

### GraphRAG Module
- `src/graph/extractor.py` - Entity extraction
- `src/graph/neo4j_client.py` - Neo4j operations
- `src/graph/community.py` - Leiden communities
- `src/graph/query.py` - Hybrid graph+vector retrieval
- `src/graph/schemas.py` - GraphEntity, Community models

### Stage Runners
- `src/stages/run_stage_4_chunking.py` - All chunking strategies via `--strategy` arg
- `src/stages/run_stage_4_5_raptor.py` - RAPTOR tree building
- `src/stages/run_stage_4_6_graph_extract.py` - GraphRAG entity extraction
- `src/stages/run_stage_6b_neo4j.py` - Neo4j upload + Leiden communities
- `src/stages/run_stage_7_evaluation.py` - Evaluation with `--preprocessing` arg

---

## Success Criteria

| Metric | Baseline (Run 3) | Target |
|--------|------------------|--------|
| Relevancy | 0.786 | > 0.85 |
| Faithfulness | 0.885 | > 0.92 |
| Failures | 0/23 | 0/23 |
| MULTI_HOP queries | Not handled | Decomposed & merged |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM cost for contextual enrichment | Use gpt-5-nano, batch processing |
| RAPTOR clustering quality | Test with different k values, use HDBSCAN |
| Neo4j complexity | Start with simple entity types, expand gradually |
| Over-engineering | Implement in order, stop when targets met |

---

## Appendix A: Strategy Pattern Template

This section documents the exact pattern used for preprocessing strategies. **Copy this pattern** when implementing chunking, embedding, or retrieval strategies.

### A.1 Pattern Overview

The Strategy Pattern with Registry provides:
1. **Modularity** - Each strategy is an isolated function
2. **Testability** - Switch strategies via UI/CLI for A/B testing
3. **Extensibility** - Add new strategies without modifying existing code
4. **Traceability** - All results track which strategy was used

### A.2 File Structure

```
src/{domain}/
├── __init__.py          # Export strategy list and main function
├── {main_module}.py     # Contains dispatcher and result dataclass
└── strategies.py        # Strategy registry and implementations
```

### A.3 Code Templates

#### A.3.1 Config (src/config.py)

```python
# =============================================================================
# {DOMAIN} STRATEGY CONFIGURATION
# =============================================================================

# List of (id, display_name, description) tuples
# UI dropdowns and CLI choices are generated from this list
AVAILABLE_{DOMAIN}_STRATEGIES = [
    ("strategy_a", "Strategy A", "Description of strategy A"),
    ("strategy_b", "Strategy B", "Description of strategy B"),
]

# Default strategy when none specified
DEFAULT_{DOMAIN}_STRATEGY = "strategy_a"
```

#### A.3.2 Strategies Module (src/{domain}/strategies.py)

```python
"""Strategy implementations for {domain}.

Each strategy has the same signature:
    def strategy_name(input: InputType, **kwargs) -> ResultType

Add new strategies by:
1. Implement the function
2. Add to STRATEGIES dict
3. Add to AVAILABLE_{DOMAIN}_STRATEGIES in config.py
"""

from typing import Dict, Callable, Optional
from src.{domain}.{main_module} import ResultType

# Type alias for strategy functions
StrategyFunction = Callable[..., ResultType]


def strategy_a(input: InputType, **kwargs) -> ResultType:
    """Strategy A implementation.

    Args:
        input: The input to process.
        **kwargs: Additional options (model, etc.)

    Returns:
        ResultType with strategy_used="strategy_a"
    """
    # Implementation
    return ResultType(
        # ... fields ...
        strategy_used="strategy_a",
    )


def strategy_b(input: InputType, **kwargs) -> ResultType:
    """Strategy B implementation."""
    # Implementation
    return ResultType(
        # ... fields ...
        strategy_used="strategy_b",
    )


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

STRATEGIES: Dict[str, StrategyFunction] = {
    "strategy_a": strategy_a,
    "strategy_b": strategy_b,
}


def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of the keys in STRATEGIES dict.

    Returns:
        The strategy function.

    Raises:
        ValueError: If strategy_id not found in registry.
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGIES[strategy_id]
```

#### A.3.3 Result Dataclass (src/{domain}/{main_module}.py)

```python
from dataclasses import dataclass

@dataclass
class ResultType:
    """Result of {domain} processing.

    Attributes:
        # ... domain-specific fields ...
        strategy_used: ID of the strategy that produced this result.
        processing_time_ms: Time taken for processing.
    """
    # Domain-specific fields
    output: Any

    # Strategy tracking (REQUIRED)
    strategy_used: str = ""
    processing_time_ms: float = 0.0
```

#### A.3.4 Dispatcher Function (src/{domain}/{main_module}.py)

```python
def process_{domain}(
    input: InputType,
    strategy: Optional[str] = None,
    **kwargs,
) -> ResultType:
    """Main entry point for {domain} processing.

    Routes to appropriate strategy based on strategy parameter.
    If strategy is None, uses DEFAULT_{DOMAIN}_STRATEGY from config.

    Args:
        input: The input to process.
        strategy: Strategy ID (e.g., "strategy_a", "strategy_b").
        **kwargs: Passed to strategy function.

    Returns:
        ResultType with strategy_used field set.
    """
    from src.config import DEFAULT_{DOMAIN}_STRATEGY
    from src.{domain}.strategies import get_strategy

    if strategy is None:
        strategy = DEFAULT_{DOMAIN}_STRATEGY

    strategy_fn = get_strategy(strategy)
    return strategy_fn(input, **kwargs)
```

#### A.3.5 Module __init__.py

```python
"""Exports for {domain} module."""

from src.{domain}.{main_module} import (
    process_{domain},
    ResultType,
)

from src.config import AVAILABLE_{DOMAIN}_STRATEGIES

__all__ = [
    "process_{domain}",
    "ResultType",
    "AVAILABLE_{DOMAIN}_STRATEGIES",
]
```

### A.4 UI Integration (src/ui/app.py)

```python
from src.config import AVAILABLE_{DOMAIN}_STRATEGIES, DEFAULT_{DOMAIN}_STRATEGY

# Build dropdown options
strategy_options = {s[0]: (s[1], s[2]) for s in AVAILABLE_{DOMAIN}_STRATEGIES}
strategy_ids = list(strategy_options.keys())
default_idx = strategy_ids.index(DEFAULT_{DOMAIN}_STRATEGY)

# Streamlit selectbox
selected_strategy = st.sidebar.selectbox(
    "Strategy",
    options=strategy_ids,
    index=default_idx,
    format_func=lambda x: strategy_options[x][0],  # Display name
    help=strategy_options[strategy_ids[0]][1],  # Show description
)

# Pass to processing function
result = process_{domain}(input, strategy=selected_strategy)
```

### A.5 CLI Integration (src/run_stage_*.py)

```python
import argparse
from src.config import AVAILABLE_{DOMAIN}_STRATEGIES

# Get strategy IDs for CLI choices
strategy_choices = [s[0] for s in AVAILABLE_{DOMAIN}_STRATEGIES]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--{domain}-strategy",
    type=str,
    choices=strategy_choices,
    default="none",  # or DEFAULT_{DOMAIN}_STRATEGY
    help="{Domain} strategy to use",
)

args = parser.parse_args()

# Use in processing
result = process_{domain}(input, strategy=args.{domain}_strategy)
```

### A.6 Logging Integration (src/utils/*_logger.py)

```python
def _build_{domain}_section(result) -> Dict:
    """Build {domain} section from ResultType."""
    if not result:
        return {"enabled": False}
    return {
        "enabled": True,
        "strategy": result.strategy_used,
        # ... other fields ...
        "time_ms": round(result.processing_time_ms, 1),
    }
```

### A.7 Checklist for New Strategy Domain

When adding a new strategy domain (e.g., chunking), follow these steps:

- [ ] Add `AVAILABLE_{DOMAIN}_STRATEGIES` to `src/config.py`
- [ ] Add `DEFAULT_{DOMAIN}_STRATEGY` to `src/config.py`
- [ ] Create `src/{domain}/strategies.py` with registry
- [ ] Add `strategy_used` field to result dataclass
- [ ] Add `strategy` parameter to main dispatcher function
- [ ] Update `src/{domain}/__init__.py` exports
- [ ] Add dropdown to UI (if applicable)
- [ ] Add CLI argument (if applicable)
- [ ] Add to logging/tracking (if applicable)
- [ ] Test each strategy independently
- [ ] Run RAGAS evaluation to compare

---

## Appendix B: Preprocessing Strategy Reference

### B.1 Current Implementation

**Files:**
- `src/config.py` - `AVAILABLE_PREPROCESSING_STRATEGIES`
- `src/rag_pipeline/retrieval/preprocessing/strategies.py` - Strategy registry
- `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` - Dispatcher
- `src/ui/app.py` - UI strategy dropdown
- `src/stages/run_stage_7_evaluation.py` - CLI `--preprocessing` argument

### B.2 Available Strategies

| ID | Display | Description | When Used |
|----|---------|-------------|-----------|
| `none` | None | Return original query unchanged | Baseline testing |
| `hyde` | HyDE | Generate hypothetical answer for semantic matching | 1 LLM call, 1 search |
| `decomposition` | Decomposition | Break into sub-questions + RRF merge | 1 LLM call, 3-4 searches |
| `graphrag` | GraphRAG | Hybrid graph + vector retrieval via RRF | Neo4j traversal + search |

### B.3 Usage Examples

```python
# In code
from src.rag_pipeline.retrieval.preprocessing import preprocess_query

# Use default strategy (hyde)
result = preprocess_query("Why do humans procrastinate?")
print(result.strategy_used)  # "hyde"

# Explicit strategy
result = preprocess_query("Why do humans procrastinate?", strategy="decomposition")
print(result.strategy_used)  # "decomposition"
```

```bash
# From CLI
python -m src.stages.run_stage_7_evaluation --preprocessing none
python -m src.stages.run_stage_7_evaluation --preprocessing hyde
python -m src.stages.run_stage_7_evaluation --preprocessing decomposition
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

### B.4 Adding New Preprocessing Strategy

To add a new strategy:

1. Add to `AVAILABLE_PREPROCESSING_STRATEGIES` in `src/config.py`
2. Implement function in `src/rag_pipeline/retrieval/preprocessing/strategies.py`
3. Add to `STRATEGIES` registry dict
4. Test via CLI: `python -m src.stages.run_stage_7_evaluation --preprocessing new_strategy`
