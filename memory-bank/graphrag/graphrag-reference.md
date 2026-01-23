# GraphRAG: Technical Reference

**Date:** 2026-01-23
**Status:** All Phases Complete
**Paper:** [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (Microsoft Research, April 2024)

> **Public Docs:** [docs/preprocessing/graphrag.md](../../docs/preprocessing/graphrag.md) - Educational overview (theory + RAGLab basics)
> **Related:** [graphrag-sota-report.md](graphrag-sota-report.md) - 2025 implementation landscape and benchmarks

This file is an internal technical reference with troubleshooting, module details, and architecture diagrams. For the educational overview, see the public docs.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Reference](#quick-reference)
3. [Paper Summary](#paper-summary)
4. [Implementation Architecture](#implementation-architecture)
5. [Module Reference](#module-reference)
6. [Configuration](#configuration)
7. [Implementation Details](#implementation-details)
8. [Troubleshooting](#troubleshooting)

---

## Overview

GraphRAG augments RAG with knowledge graph structures for:
- Cross-document relationship discovery (entity linking)
- Hierarchical community detection (Leiden algorithm)
- Global query answering via community summaries

**Key Results (Paper):**
- 72-83% win rate on comprehensiveness vs baseline RAG
- 62-82% win rate on diversity of answers
- 97% fewer tokens at query time using community summaries

---

## Quick Reference

### Prerequisites

```bash
conda activate raglab
docker compose up -d neo4j weaviate
```

**Verify services:**
- Neo4j: http://localhost:7474 (neo4j / raglab_graphrag)
- Weaviate: http://localhost:8080

### Execution Steps

```bash
# Step 1: Entity Extraction (uses curated types from graphrag_types.yaml)
python -m src.stages.run_stage_4_5_graph_extract --strategy section

# Step 2: Upload + Leiden + Summarization + Entity Embeddings
python -m src.stages.run_stage_6b_neo4j

# Step 3: Query
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

### Crash Recovery

Stage 6b is crash-proof. Resume from any phase:

```bash
# Resume from Leiden (graph already uploaded)
python -m src.stages.run_stage_6b_neo4j --from leiden

# Resume from summaries (Leiden done, regenerate summaries using checkpoint)
python -m src.stages.run_stage_6b_neo4j --from summaries

# Full re-run (deterministic Leiden guarantees same IDs)
python -m src.stages.run_stage_6b_neo4j
```

### Data Flow

```
Stage 4 (chunks) -> Stage 4.5 extraction -> Stage 6b (Neo4j + Leiden) -> Query
                          |                            |
                          v                            v
               extraction_results.json         communities.json
               (uses graphrag_types.yaml)      leiden_checkpoint.json
                                               Entity embeddings (Weaviate)
                                               Community embeddings (Weaviate)
```

---

## Paper Summary

### The Problem

Vector search fails on "global" questions:
```
Query: "What are the main themes across all 19 books?"
```
No single chunk contains this answer. GraphRAG solves this with knowledge graphs + community summaries.

### GraphRAG Methodology (From Paper)

#### 1. Entity and Relationship Extraction

LLM extracts three element types:
- **Entities**: Named items with type classification and descriptions
- **Relationships**: Connections between entities with strength scoring
- **Claims**: Factual statements about entities (optional)

#### 2. Knowledge Graph Construction

- Entity instances -> nodes with aggregated descriptions
- Relationship instances -> weighted edges
- Claims -> aggregated as node/edge attributes

#### 3. Leiden Community Detection

Hierarchical community detection:
- C0: Finest granularity (specific topics)
- C1: Medium granularity (domain-level themes)
- C2: Coarsest granularity (corpus-wide themes)

#### 4. Community Summarization

For each community:
1. Collect member entities (sorted by PageRank)
2. Collect internal relationships
3. Generate LLM summary with title, executive summary, key insights

#### 5. Query Processing

**Local queries** (entity-specific):
- Entity extraction from query
- Graph traversal from matched entities
- Vector search for similar chunks
- RRF merge of results

**Global queries** (thematic):
- Map: Generate partial answer from each relevant community
- Reduce: Synthesize partial answers into final response

---

## Implementation Architecture

### Two-Phase Design

```
PHASE 1: INDEXING (Offline)
================================================================================

   Stage 4                Stage 4.5                   Stage 6b
   +----------+          +-----------------+         +---------------------------+
   | Chunks   |--------->| Entity          |-------->| Neo4j Upload + Leiden     |
   | (JSON)   |          | Extraction      |         | + Community Summaries     |
   +----------+          | (LLM per chunk) |         | + Entity Embeddings       |
                         +-----------------+         +---------------------------+

   Output Files:
   - data/processed/07_graph/extraction_results.json  (entities + relationships)
   - data/processed/07_graph/discovered_types.json    (consolidated taxonomy)
   - data/processed/07_graph/communities.json         (summaries backup)
   - data/processed/07_graph/leiden_checkpoint.json   (crash recovery)
   - Weaviate: Community_section800_v1 collection     (community embeddings)
   - Weaviate: Entity_section800_v1 collection        (entity embeddings)
   - Neo4j: Entity nodes + RELATED_TO edges           (graph)

PHASE 2: QUERY (Online)
================================================================================

   User Query
       |
       +------------------+------------------+--------------------+
       v                  v                  v                    v
   +----------+      +--------------+   +----------+       +------------+
   | Entity   |      | Neo4j        |   | Weaviate |       | Community  |
   | Extract  |----->| Graph        |   | Vector   |       | Retrieval  |
   | (embed)  |      | Traversal    |   | Search   |       |            |
   +----------+      +--------------+   +----------+       +------------+
                            |                  |                   |
                            +--------+---------+                   |
                                     v                             |
                             +---------------+                     |
                             | RRF Merge     |                     |
                             | (boost overlaps)                    |
                             +---------------+                     |
                                     |                             |
                                     +--------------+--------------+
                                                    v
                                           +---------------+
                                           | Answer Gen    |
                                           | (chunks +     |
                                           |  communities) |
                                           +---------------+
```

### Query Flow Detail

```
Query: "How does dopamine affect motivation?"
        |
        v
1. ENTITY EXTRACTION (query_entities.py)
   - Primary: Embedding similarity against entity descriptions (~50ms)
   - Fallback: LLM extraction if embedding returns empty (~1-2s)
   -> Extracted: ["dopamine", "motivation"]
        |
        v
2. NEO4J LOOKUP (neo4j_client.py)
   - Pre-normalize entity names (Python normalization)
   - Match against graph
   -> Matched: ["dopamine"]
        |
        v
3. GRAPH TRAVERSAL (neo4j_client.py)
   - Traverse 2 hops from matched entities
   - Collect source_chunk_ids from related entities
   -> graph_chunk_ids = ["behave::chunk_42", ...]
        |
        v
4. PARALLEL RETRIEVAL
   - Vector search (Weaviate) -> top-k chunks
   - Fetch ALL graph chunks (ContainsAny filter)
   - Community retrieval (embedding similarity)
        |
        v
5. RRF MERGE (rrf.py)
   - Graph chunks ranked by path_length (shorter = higher)
   - RRF score = 1/(k+rank_vector) + 1/(k+rank_graph)
   - Chunks in BOTH lists get boosted
        |
        v
6. ANSWER GENERATION
   - Prompt includes:
     - Background: community summaries + entity relationships
     - Retrieved Passages: RRF-merged chunks
     - Question: original query
```

---

## Module Reference

### Core Modules

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/graph/schemas.py` | Pydantic models | `GraphEntity`, `Community`, `CommunityRelationship` |
| `src/graph/extractor.py` | LLM entity extraction | `load_chunks_for_extraction()` |
| `src/graph/extraction.py` | Entity extraction | `run_extraction()` |
| `src/graph/neo4j_client.py` | Graph DB operations | `upload_extraction_results()`, `find_entity_neighbors()` |
| `src/graph/community.py` | Leiden + summaries | `detect_and_summarize_communities()` |
| `src/graph/hierarchy.py` | Multi-level parsing | `parse_leiden_hierarchy()`, `CommunityLevel` |
| `src/graph/centrality.py` | PageRank computation | `compute_pagerank()`, `write_pagerank_to_neo4j()` |
| `src/graph/map_reduce.py` | Global queries | `map_reduce_global_query()`, `classify_query()` |
| `src/graph/query_entities.py` | Query entity extraction | `extract_query_entities()`, `extract_query_entities_embedding()` |
| `src/graph/query.py` | Hybrid retrieval | `hybrid_graph_retrieval()`, `hybrid_graph_retrieval_with_map_reduce()` |

### Key Algorithms

#### Entity Normalization (schemas.py)

```python
def normalized_name(self) -> str:
    name = unicodedata.normalize('NFKC', self.name.strip())  # cafe -> cafe
    name = name.lower()
    # Remove leading/trailing stopwords
    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:  # {'the', 'a', 'an', ...}
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()
    name = re.sub(r'[^\w\s]', '', ' '.join(words))  # Remove punctuation
    return ' '.join(name.split())
```

#### RRF Merge (rrf.py)

```python
def reciprocal_rank_fusion(result_lists, query_types, k=60, top_k=10):
    scores = defaultdict(float)
    for query_idx, results in enumerate(result_lists):
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.chunk_id] += rrf_score
    # Chunks in BOTH lists get summed scores -> BOOSTED
```

#### Leiden (community.py)

```python
def run_leiden(gds, graph, resolution=1.0, seed=42, concurrency=1):
    """DETERMINISTIC settings for reproducibility."""
    result = gds.leiden.stream(
        graph,
        gamma=resolution,
        randomSeed=seed,         # FIXED SEED
        concurrency=concurrency, # SINGLE THREAD
        includeIntermediateCommunities=True,  # For hierarchy
    )
    # Same graph + same seed = IDENTICAL community assignments
```

---

## Configuration

### Key Parameters (src/config.py)

```python
# Entity Extraction
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-3-haiku"
GRAPHRAG_MAX_ENTITIES = 10
GRAPHRAG_MAX_RELATIONSHIPS = 7

# Auto-Tuning
GRAPHRAG_TYPES_PER_CORPUS = 12
GRAPHRAG_MIN_CORPUS_PERCENTAGE = 1.0

# Leiden Algorithm
GRAPHRAG_LEIDEN_RESOLUTION = 1.0
GRAPHRAG_LEIDEN_SEED = 42
GRAPHRAG_LEIDEN_CONCURRENCY = 1
GRAPHRAG_MIN_COMMUNITY_SIZE = 3
GRAPHRAG_LEIDEN_MAX_LEVELS = 10

# Hierarchy
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3  # C0, C1, C2
GRAPHRAG_LEVEL_MIN_SIZES = {0: 3, 1: 5, 2: 10}

# PageRank
GRAPHRAG_PAGERANK_DAMPING = 0.85
GRAPHRAG_PAGERANK_ITERATIONS = 20

# Map-Reduce
GRAPHRAG_MAP_REDUCE_TOP_K = 5  # Communities for map phase
GRAPHRAG_MAP_MAX_TOKENS = 300
GRAPHRAG_REDUCE_MAX_TOKENS = 500

# Query Time
GRAPHRAG_TRAVERSE_DEPTH = 2
GRAPHRAG_TOP_COMMUNITIES = 3
GRAPHRAG_RRF_K = 60

# Entity Extraction (Query Time)
GRAPHRAG_ENTITY_EXTRACTION_TOP_K = 10
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3
GRAPHRAG_USE_EMBEDDING_EXTRACTION = True  # Use embedding-based (fallback to LLM)
```

### Consolidation Strategies (Stage 4.5)

| Strategy | Algorithm | Use Case |
|----------|-----------|----------|
| `global` | Rank by total count | Single-domain corpora |
| `stratified` | Top-K from each corpus | Mixed corpora (recommended) |

---

## Implementation Details

### Paper Alignment Features

| Feature | Paper | Implementation | Status |
|---------|-------|----------------|--------|
| Entity Extraction | LLM per chunk | LLM with structured output | Done |
| Entity Types | Predefined in settings.yaml | Curated in graphrag_types.yaml | Done |
| Entity Resolution | String matching | Normalized name matching | Done |
| Leiden Communities | Multi-level | C0, C1, C2 with summaries | Done |
| PageRank Centrality | Hub entity ranking | Neo4j GDS PageRank | Done |
| Community Summaries | LLM with importance ratings | LLM + embeddings | Done |
| Query Entity Extract | Embedding similarity | Embedding primary + LLM fallback | Done |
| Map-Reduce Global | Parallel map + reduce | Async map-reduce | Done |
| Self-reflection loop | 3 iterations | Single pass | Skipped (3x cost) |
| Claims extraction | Verifiable facts | Not implemented | Skipped (scope) |

### Crash-Proof Design

1. **Deterministic Leiden**: `randomSeed=42` + `concurrency=1`
   - Same graph + same seed = same community assignments (guaranteed)
   - Enables resume after Neo4j reset without ID mismatches

2. **Weaviate Storage**: Community embeddings stored in Weaviate
   - Efficient HNSW vector search (O(log n) vs O(n) for JSON file)
   - ~12MB total vs 383MB JSON with inline embeddings

3. **Atomic Uploads**: Each community uploaded to Weaviate immediately
   - Resume skips existing communities (checks Weaviate)
   - No data loss on crash

### Neo4j Schema

```cypher
(:Entity {
  name, normalized_name, entity_type, description,
  chunk_ids, mention_count, community_id,
  pagerank  -- PageRank centrality score
})

(:Entity)-[:RELATED_TO {type, description, strength, chunk_ids}]->(:Entity)
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Neo4j connection refused | `docker compose up -d neo4j`, wait 30s |
| "No extraction results" | Run Stage 4.5 or 4.6 first |
| Leiden "GDS not found" | Check `NEO4J_PLUGINS` in docker-compose.yml |
| Empty communities | Increase `GRAPHRAG_MIN_COMMUNITY_SIZE` |
| Entity not found | Check normalization (case, stopwords, punctuation) |
| Map-reduce timeout | Reduce `GRAPHRAG_MAP_REDUCE_TOP_K` |

### Useful Neo4j Queries

```cypher
-- Count entities
MATCH (e:Entity) RETURN count(e)

-- Entity types distribution
MATCH (e:Entity) RETURN e.entity_type, count(*) as count ORDER BY count DESC

-- Find entity relationships
MATCH (e:Entity {normalized_name: 'dopamine'})-[r]-(n) RETURN e, r, n LIMIT 20

-- Community sizes
MATCH (e:Entity) WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(*) as size ORDER BY size DESC

-- PageRank top entities
MATCH (e:Entity) WHERE e.pagerank IS NOT NULL
RETURN e.name, e.pagerank ORDER BY e.pagerank DESC LIMIT 10
```

---

*Last Updated: 2026-01-09*
