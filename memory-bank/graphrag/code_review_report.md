# GraphRAG Code Review Report

**Date**: 2026-01-25
**Scope**: Complete comparison of Microsoft Reference Implementation vs RAGLab code vs RAGLab documentation

## Executive Summary

This audit traces every function in both indexing and query workflows, comparing RAGLab's implementation against:
1. Microsoft GraphRAG reference implementation (arXiv:2404.16130)
2. RAGLab's own documentation (`graphrag_raglab_implementation.md`)

**Critical Finding**: A hierarchy level bug causes community context retrieval to fail in full hierarchy mode.

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [Indexing Pipeline Comparison](#2-indexing-pipeline-comparison)
3. [Query Pipeline Comparison](#3-query-pipeline-comparison)
4. [Token Budget Analysis](#4-token-budget-analysis)
5. [Community Hierarchy Analysis](#5-community-hierarchy-analysis)
6. [Combined Degree Implementation](#6-combined-degree-implementation)
7. [Documentation Accuracy](#7-documentation-accuracy)
8. [Recommendations](#8-recommendations)

---

## 1. Critical Issues

### 1.1 ~~CRITICAL BUG~~ FIXED: Hierarchy Level Mismatch

**Location**: `src/graph/query.py:288-301`

**Original Bug**:
```python
# Only include L0 (finest level where entities have community_id)
if level == 0:
    community_lookup[cid] = c
```

**Problem**:
- Entity `community_id` in Neo4j is from Leiden's `communityId` field = **FINEST level**
- In full hierarchy mode, finest level is L2 (max_level), not L0
- The filter `level == 0` looks for L0 (coarsest) communities
- **Result**: Entity community lookup will FAIL for full hierarchy indexing

**Fix Applied** (2026-01-25):
```python
# Finest level = GRAPHRAG_MAX_HIERARCHY_LEVELS - 1 (e.g., L2 for 3 levels)
# Entities in Neo4j store community_id from Leiden's finest level
finest_level = GRAPHRAG_MAX_HIERARCHY_LEVELS - 1

# Only include finest level (where entities have community_id)
if level == finest_level:
    community_lookup[cid] = c
```

**Status**: **FIXED**

---

### 1.2 ~~Inconsistent Hierarchy Convention in Comments~~ FIXED

**Original Conflicting Comments**:

| File | Line | Said | Fixed |
|------|------|------|-------|
| `hierarchy.py` | 6-8 | "Level 0 (C0): Coarsest granularity" | Correct |
| `hierarchy.py` | 73-76 | "Index 0: Level 0 community (finest)" | **FIXED** → "coarsest" |
| `hierarchy.py` | 118-121 | "intermediate_ids[0] = coarsest (L0)" | Correct |
| `community.py` | 856 | "L0=finest, L1=medium, L2=coarsest" | **FIXED** → "L0=coarsest, L2=finest" |
| `query.py` | 292 | "L0 (finest level)" | **FIXED** → dynamic finest_level |
| `query.py` | 443 | "0=coarsest for global queries" | Correct |

**Current Convention** (consistent after fix):
- L0 = coarsest (corpus-wide themes) - used for global queries
- L2 = finest (specific topics) - used for entity membership lookup
- This matches Microsoft's convention: C0 = Root/Coarsest

**Status**: **FIXED**

---

## 2. Indexing Pipeline Comparison

### 2.1 Entity Extraction

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Output format | Delimited tuples (`<\|>##<\|COMPLETE\|>`) | Pydantic JSON Schema | **DIFFERENT** (acceptable) |
| Entity types | 4 generic (PERSON, ORG, LOC, EVENT) | 33 curated (domain-specific) | **DIFFERENT** (improved) |
| Gleaning passes | 1 round (paper recommendation) | `GRAPHRAG_MAX_GLEANINGS=1` | **MATCHES** |
| Chunk size | 600 tokens (paper) | 800 tokens | **DIFFERENT** (configurable) |
| Max entities/chunk | Not specified | 10 | **ACCEPTABLE** |
| Max relationships/chunk | Not specified | 7 | **ACCEPTABLE** |

**Code Location**: `src/graph/extraction.py`

### 2.2 Relationship Weight

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Scale | 1-10 (strength score) | 0.0-1.0 (confidence) | **DIFFERENT** |
| Accumulation | `sum(all_extracted_weights)` | Not implemented (single extraction per pair) | **DIFFERENT** |
| Default fallback | 1.0 if unparseable | 1.0 | **MATCHES** |

**Impact**: Relationship weights have different semantics but same ranking behavior.

### 2.3 Deduplication

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Entity matching | Exact string match | Normalized name (NFKC + lowercase + stopword removal) | **DIFFERENT** (improved) |
| Description handling | Concatenate or LLM summarize | LLM summarize with GPT-4o-mini | **MATCHES** |
| source_chunk_ids | Accumulated as list | Accumulated as list | **MATCHES** |

### 2.4 Leiden Community Detection

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Algorithm | Leiden (hierarchical) | GDS Leiden with `includeIntermediateCommunities=True` | **MATCHES** |
| Resolution (gamma) | Configurable | 1.0 (`GRAPHRAG_LEIDEN_RESOLUTION`) | **MATCHES** |
| Seed | Configurable | 42 (fixed, deterministic) | **MATCHES** |
| Concurrency | Not specified | 1 (single-threaded for reproducibility) | **IMPROVED** |
| Graph projection | Undirected weighted | Undirected (weight not used) | **DEVIATION** |

**Missing**: Edge weight from relationship.weight not used in graph projection.

### 2.5 Community Summarization

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Processing order | Bottom-up (Leaf → Root) | Bottom-up (L2 → L1 → L0) | **MATCHES** |
| Token limit | 8,000 tokens | 8,000 characters | **DIFFERENT** (chars vs tokens) |
| Member ranking | combined_degree | PageRank | **DIFFERENT** |
| Output format | JSON with title, rating, findings[] | Plain text (2-3 paragraphs) | **DIFFERENT** |
| Child substitution | Yes, for large contexts | Yes, implemented | **MATCHES** |

**Code Location**: `src/graph/community.py:920-1070`

---

## 3. Query Pipeline Comparison

### 3.1 Entity Extraction at Query Time

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Primary method | Embedding similarity on entity descriptions | Embedding similarity (Weaviate HNSW) | **MATCHES** |
| top_k entities | 10 | 10 (`GRAPHRAG_ENTITY_EXTRACTION_TOP_K`) | **MATCHES** |
| oversample_scaler | 2 | Not implemented | **MISSING** |
| Fallback | Not specified | LLM extraction + regex | **IMPROVED** |
| Min similarity | Not specified | 0.3 (`GRAPHRAG_ENTITY_MIN_SIMILARITY`) | **ACCEPTABLE** |

**Code Location**: `src/graph/query_entities.py:61-124`

### 3.2 Graph Traversal

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Traversal depth | Not specified | 2 hops (`GRAPHRAG_TRAVERSE_DEPTH`) | **ACCEPTABLE** |
| Neighbor limit | Not specified | 50 per entity | **ACCEPTABLE** |
| Relationship prioritization | In-network first, then out-of-network | All neighbors equally | **DIFFERENT** |
| combined_degree calculation | `degree(source) + degree(target)` | `start_degree + neighbor_degree` | **MATCHES** |

**Cypher Query** (`src/graph/neo4j_client.py:456-468`):
```cypher
MATCH (start:Entity {normalized_name: $normalized_name})
MATCH path = (start)-[*1..{max_hops}]-(neighbor:Entity)
WHERE start <> neighbor
RETURN DISTINCT
    neighbor.name, neighbor.entity_type, neighbor.description,
    neighbor.source_chunk_ids, length(path) as path_length,
    coalesce(neighbor.degree, 0) as degree,
    coalesce(start.degree, 0) as start_degree
ORDER BY path_length, name
LIMIT $limit
```

### 3.3 Chunk Retrieval

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Source | Entity's `source_chunk_ids` | Entity's `source_chunk_ids` | **MATCHES** |
| Method | Direct retrieval by ID | Weaviate `ContainsAny` filter | **MATCHES** |
| Vector search | No (pure graph) | No (batch filter, not vector) | **MATCHES** |

**Code Location**: `src/graph/query.py:151-217` (`fetch_chunks_by_ids()`)

### 3.4 Chunk Ranking

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Ranking metric | combined_degree | combined_degree | **MATCHES** |
| Formula | `degree(source) + degree(target)` | `start_degree + neighbor_degree` | **MATCHES** |
| Multiple paths | Not specified | Use MAX combined_degree | **ACCEPTABLE** |
| Sort order | Descending | Descending | **MATCHES** |

**Code Location**: `src/graph/query.py:560-619` (`_build_graph_ranked_list()`)

### 3.5 Community Context Retrieval

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Method | By entity membership | By entity membership | **MATCHES** |
| Source | Entity's community_id | Neo4j `e.community_id` | **MATCHES** |
| Level | Finest (for entity membership) | **BUG: filters for L0 instead of finest** | **BROKEN** |

**Bug Location**: `src/graph/query.py:292-293`

### 3.6 Global Query (Map-Reduce)

| Aspect | Microsoft Reference | RAGLab Implementation | Status |
|--------|--------------------|-----------------------|--------|
| Community level | C0 (coarsest) | L0 (coarsest) | **MATCHES** |
| Map phase | Parallel LLM calls | `asyncio.gather` | **MATCHES** |
| Map max tokens | Not specified | 300 (`GRAPHRAG_MAP_MAX_TOKENS`) | **ACCEPTABLE** |
| Reduce max tokens | Not specified | 500 (`GRAPHRAG_REDUCE_MAX_TOKENS`) | **ACCEPTABLE** |
| Score filtering | Remove score=0 | Filter "Not relevant" responses | **DIFFERENT** (simpler) |
| Importance scores | 0-100 integer | Not implemented (text-based filtering) | **DIFFERENT** |

**Code Location**: `src/graph/map_reduce.py`

---

## 4. Token Budget Analysis

### 4.1 Microsoft Token Budget Allocation (Local Search)

| Component | Proportion | With 12K max |
|-----------|------------|--------------|
| Text Units (chunks) | 50% | 6,000 tokens |
| Community Reports | 15% | 1,800 tokens |
| Entities + Relationships | 35% | 4,200 tokens |

### 4.2 RAGLab Implementation

**Status**: **NOT IMPLEMENTED**

RAGLab does NOT implement token budget allocation. Instead:
- Returns top_k chunks (default 10) ranked by combined_degree
- Community summaries retrieved separately (not part of token budget)
- No explicit token counting or allocation

**Code Evidence**: No `text_unit_prop`, `community_prop`, or token budget variables found in codebase.

**Impact**: May include more/fewer tokens than optimal; no cross-component optimization.

---

## 5. Community Hierarchy Analysis

### 5.1 Level Convention

**Microsoft GraphRAG**:
- C0 = Root (Coarsest) - corpus-wide themes
- C3 = Leaf (Finest) - specific topics

**RAGLab Implementation** (per code):
- L0 = Coarsest (index 0 of `intermediateCommunityIds`)
- L2 = Finest (last index, equals Leiden's `communityId`)

**Status**: Conventions MATCH conceptually, but comments are inconsistent.

### 5.2 Entity Community Assignment

**What's Stored**: `e.community_id` = Leiden's `communityId` = FINEST level (L2)

**What's Queried**: `level == 0` filter in `retrieve_community_context_by_membership()`

**Result**: **MISMATCH** - query will not find entity's community in full hierarchy mode.

### 5.3 Skip-Leiden Fallback Mode

When `--from summaries` is used (skip Leiden):
- Only L0 communities created
- Entity community_ids used directly as L0
- Query would work (L0 filter matches L0 communities)

**Inconsistency**: Full hierarchy mode and skip-leiden mode have different behaviors.

---

## 6. Combined Degree Implementation

### 6.1 Formula Verification

**Microsoft Definition**:
```
combined_degree(relationship) = degree(source_entity) + degree(target_entity)
```

**RAGLab Implementation** (`src/graph/query.py:581-584`):
```python
start_degree = entity.get("start_degree", 0)
neighbor_degree = entity.get("degree", 0)
combined_degree = start_degree + neighbor_degree
```

**Status**: **MATCHES** - `start_degree` = degree of matched entity, `neighbor_degree` = degree of discovered entity.

### 6.2 Degree Source

**Neo4j Query** (`src/graph/neo4j_client.py:465-466`):
```cypher
coalesce(neighbor.degree, 0) as degree,
coalesce(start.degree, 0) as start_degree
```

**Degree Computation** (`src/graph/centrality.py`):
- Computed via `count(RELATED_TO)` relationships
- Stored on Entity nodes as `.degree` property

**Status**: **CORRECT** implementation of Microsoft's combined_degree.

---

## 7. Documentation Accuracy

### 7.1 `graphrag_raglab_implementation.md` Accuracy

| Documented | Code Reality | Status |
|------------|--------------|--------|
| "Entity matching via embedding similarity" | Correct | **ACCURATE** |
| "combined_degree = start_degree + neighbor_degree" | Correct | **ACCURATE** |
| "L0 = coarsest for global queries" | Correct | **ACCURATE** |
| "Chunks fetched by batch filter, not vector search" | Correct | **ACCURATE** |
| ~~"RRF merge for hybrid retrieval"~~ | Updated to "Pure graph retrieval" | **FIXED** |
| "Map-reduce for global queries" | Correct | **ACCURATE** |
| "PageRank for member sorting" | Correct (not combined_degree) | **ACCURATE** |

### 7.2 ~~Stale Documentation~~ UPDATED

The documentation previously mentioned "Hybrid graph + vector retrieval with RRF merge" which was **removed** in the recent refactor. Documentation has been updated to reflect pure graph retrieval.

**Status**: **FIXED** (2026-01-25)

---

## 8. Recommendations

### 8.1 Critical Fixes - ✅ COMPLETED

1. ~~**Fix Community Level Bug**~~ ✅ DONE (`src/graph/query.py:288-301`)
2. ~~**Standardize Hierarchy Comments**~~ ✅ DONE (`hierarchy.py`, `community.py`)

### 8.2 Consider Implementing (Future Work)

1. **Token Budget Allocation**: Add Microsoft's 50% text / 15% community / 35% entities allocation
2. **Oversample Scaler**: Add 2x oversampling for entity retrieval
3. **Relationship Weight in Graph Projection**: Use edge weights in Leiden detection
4. **Importance Scores for Map Phase**: Add 0-100 scoring for better reduce phase filtering

### 8.3 Documentation Updates - ✅ COMPLETED

1. ~~Remove references to "hybrid search" and "RRF merge" for GraphRAG~~ ✅ DONE
2. Add clear hierarchy convention diagram (optional, low priority)
3. ~~Document token budget differences from Microsoft~~ ✅ DONE

---

## Appendix A: File-by-Function Trace

### Indexing Pipeline Files

| File | Key Functions | Purpose |
|------|---------------|---------|
| `src/stages/run_stage_4_5_autotune.py` | `main()` | Entry point for entity extraction |
| `src/graph/extraction.py` | `run_extraction()`, `extract_chunk()`, `consolidate_entities()` | Entity/relationship extraction |
| `src/stages/run_stage_6b_neo4j.py` | `main()` | Entry point for Neo4j upload + Leiden |
| `src/graph/neo4j_client.py` | `upload_entities()`, `upload_relationships()` | Neo4j storage |
| `src/graph/community.py` | `detect_and_summarize_communities()`, `run_leiden()` | Leiden + summarization |
| `src/graph/hierarchy.py` | `parse_leiden_hierarchy()` | Hierarchy extraction |
| `src/graph/centrality.py` | `compute_pagerank()`, `compute_degree()` | Centrality metrics |

### Query Pipeline Files

| File | Key Functions | Purpose |
|------|---------------|---------|
| `src/rag_pipeline/retrieval/strategies/graphrag.py` | `GraphRAGRetrieval.execute()` | Strategy entry point |
| `src/graph/query.py` | `graph_retrieval()`, `graph_retrieval_with_map_reduce()` | Main retrieval logic |
| `src/graph/query_entities.py` | `extract_query_entities_embedding()`, `extract_query_entities_llm()` | Entity extraction |
| `src/graph/neo4j_client.py` | `find_entity_neighbors()`, `find_entities_by_names()` | Graph traversal |
| `src/graph/map_reduce.py` | `map_reduce_global_query()`, `should_use_map_reduce()` | Global query handling |

---

## Appendix B: Configuration Parameters

```python
# Entity Extraction (Indexing)
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-3-haiku"
GRAPHRAG_MAX_EXTRACTION_TOKENS = 4000
GRAPHRAG_MAX_ENTITIES = 10
GRAPHRAG_MAX_RELATIONSHIPS = 7
GRAPHRAG_MAX_GLEANINGS = 1
GRAPHRAG_STRICT_MODE = True

# Leiden
GRAPHRAG_LEIDEN_RESOLUTION = 1.0
GRAPHRAG_LEIDEN_MAX_LEVELS = 10
GRAPHRAG_LEIDEN_SEED = 42
GRAPHRAG_LEIDEN_CONCURRENCY = 1
GRAPHRAG_MIN_COMMUNITY_SIZE = 3

# Community Summarization
GRAPHRAG_SUMMARY_MODEL = "openai/gpt-4o-mini"
GRAPHRAG_MAX_CONTEXT_TOKENS = 8000
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3

# Query-Time Entity Extraction
GRAPHRAG_ENTITY_EXTRACTION_TOP_K = 10
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3
GRAPHRAG_USE_EMBEDDING_EXTRACTION = True
GRAPHRAG_TRAVERSE_DEPTH = 2

# Map-Reduce
GRAPHRAG_MAP_MAX_TOKENS = 300
GRAPHRAG_REDUCE_MAX_TOKENS = 500
```

---

*Report generated from comprehensive code analysis on 2026-01-25*
