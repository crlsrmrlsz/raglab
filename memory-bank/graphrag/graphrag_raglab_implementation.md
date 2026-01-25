# GraphRAG RAGLab Implementation

> **Purpose**: This document provides a detailed analysis of the GraphRAG implementation in RAGLab, comparing it one-to-one against the Microsoft reference implementation documented in `graphrag_reference_implementation.md`.

## Document Structure

1. [Overview](#1-overview)
2. [Implementation Comparison Matrix](#2-implementation-comparison-matrix)
3. [Indexing Pipeline](#3-indexing-pipeline)
4. [Entity & Relationship Extraction](#4-entity--relationship-extraction)
5. [Community Detection (Leiden)](#5-community-detection-leiden)
6. [Community Summarization](#6-community-summarization)
7. [Embeddings & Storage](#7-embeddings--storage)
8. [Local Search Query Pipeline](#8-local-search-query-pipeline)
9. [Global Search (Map-Reduce)](#9-global-search-map-reduce)
10. [Data Models](#10-data-models)
11. [Configuration Reference](#11-configuration-reference)
12. [Prompts Comparison](#12-prompts-comparison)
13. [Code Snippets](#13-code-snippets)
14. [What's Different](#14-whats-different)
15. [What's Not Implemented](#15-whats-not-implemented)

---

## 1. Overview

RAGLab implements GraphRAG based on Microsoft's paper (arXiv:2404.16130) and reference implementation. The implementation is split across two stages:

- **Stage 4.5**: Entity/relationship extraction from chunks
- **Stage 6b**: Neo4j upload, Leiden community detection, summarization

### Architecture Overview

```mermaid
flowchart TB
    subgraph Phase1["Stage 4.5: Extraction"]
        CHUNKS[Section Chunks<br/>data/chunks/section/*.json] --> EXTRACT[extraction.py]
        EXTRACT --> |LLM calls| ENTITIES[Entities + Relationships]
        ENTITIES --> CONSOLIDATE[Consolidation<br/>by normalized_name + type]
        CONSOLIDATE --> RESULTS[extraction_results.json<br/>data/graph/extraction_results.json]
    end

    subgraph Phase2["Stage 6b: Graph Construction"]
        RESULTS --> UPLOAD[neo4j_client.py]
        UPLOAD --> NEO4J[(Neo4j<br/>localhost:7687)]

        NEO4J --> GDS[GDS Projection<br/>graphrag]
        GDS --> LEIDEN[community.py<br/>Leiden Algorithm]
        LEIDEN --> PAGERANK[centrality.py<br/>PageRank]
        PAGERANK --> NEO4J

        LEIDEN --> SUMMARY[Community Summaries<br/>LLM + Embeddings]
        SUMMARY --> WEAVIATE_COMM[(Weaviate<br/>Community_section800_v1)]
        SUMMARY --> JSON[communities.json]

        NEO4J --> ENTITY_EMB[Entity Embeddings]
        ENTITY_EMB --> WEAVIATE_ENT[(Weaviate<br/>Entity_section800_v1)]
    end

    subgraph Phase3["Query Time (Graph-Only)"]
        QUERY[User Query] --> ENTITY_EXT[query_entities.py<br/>Embedding + LLM fallback]
        ENTITY_EXT --> |matched entities| NEO4J
        NEO4J --> |graph traversal| NEIGHBORS[Related Entities]
        NEIGHBORS --> CHUNK_IDS[Source Chunk IDs]

        CHUNK_IDS --> FETCH[Weaviate Batch Fetch<br/>by chunk_id]
        FETCH --> RANK[Rank by combined_degree<br/>start_degree + neighbor_degree]

        ENTITY_EXT --> |community lookup| COMM_CTX[Community Context<br/>by entity membership]

        RANK --> GENERATION[Answer Generation]
        COMM_CTX --> GENERATION
    end

    style NEO4J fill:#4db33d,color:white
    style WEAVIATE_COMM fill:#fa8072
    style WEAVIATE_ENT fill:#fa8072
```

---

## 2. Implementation Comparison Matrix

### Indexing Features

| Feature | Microsoft Reference | RAGLab Implementation | Status |
|---------|--------------------|-----------------------|--------|
| Chunking | 600 tokens (paper) | 800 tokens (configurable) | Different |
| Entity Extraction | LLM with delimited output | LLM with **Pydantic JSON Schema** | Different format |
| Gleaning | 1+ rounds | 1 round (configurable) | Matches |
| Entity Types | Configurable in settings.yaml | **graphrag_types.yaml** (8 types) | Matches |
| Entity Matching | Exact string match | Normalized name (NFKC + lowercase + stopwords) | Enhanced |
| Relationship Types | Open-ended | Open-ended | Matches |
| Description Summarization | LLM consolidation | LLM consolidation | Matches |
| Community Detection | Leiden (graspologic) | Leiden (Neo4j GDS) | Different library |
| Community Hierarchy | C0 (coarsest) to C3 | L0 (coarsest) to L2 (3 levels) | Matches convention |
| Community Summaries | Bottom-up LLM | **Bottom-up with child substitution** | Matches |
| Entity Embeddings | Description embeddings | Description embeddings | Matches |

### Query Features

| Feature | Microsoft Reference | RAGLab Implementation | Status |
|---------|--------------------|-----------------------|--------|
| Local Search | Entity matching → traversal | Entity matching → traversal | Matches |
| Entity Matching | Embedding similarity | Embedding similarity + LLM fallback | Enhanced |
| Relationship Prioritization | combined_degree | **combined_degree** (start + neighbor degree) | Matches |
| Token Budget | 50% text, 10% community | combined_degree ranking (no explicit budget) | Different |
| Community Context | In token budget | By entity membership (separate) | Different |
| Global Search | Map-reduce over communities | Map-reduce over L0 communities | Matches |
| Query Classification | LLM-based | LLM-based | Matches |
| DRIFT Search | Dynamic iterative search | **Not implemented** | Missing |

### Storage

| Feature | Microsoft Reference | RAGLab Implementation | Status |
|---------|--------------------|-----------------------|--------|
| Chunks | Parquet files | JSON files | Different |
| Entities | Parquet | Neo4j + Weaviate | Different |
| Relationships | Parquet | Neo4j | Matches (different store) |
| Communities | Parquet | JSON + Weaviate | Different |
| Vector Store | LanceDB (default) | Weaviate | Different |
| Graph Store | None (Parquet) | **Neo4j** (native graph DB) | Enhanced |

---

## 3. Indexing Pipeline

### RAGLab Pipeline

```mermaid
flowchart TB
    subgraph Stage1["Stage 1-4: Content Preparation"]
        PDF[PDF Documents] --> EXTRACT_DOC[Stage 1: Docling]
        EXTRACT_DOC --> MARKDOWN[Markdown Files]
        MARKDOWN --> CLEAN[Stage 2: Cleaning]
        CLEAN --> SEGMENT[Stage 3: Segmentation]
        SEGMENT --> CHUNK[Stage 4: Section Chunking<br/>800 tokens, 2 sentences overlap]
        CHUNK --> CHUNKS_JSON[data/chunks/section/*.json]
    end

    subgraph Stage45["Stage 4.5: Entity Extraction"]
        CHUNKS_JSON --> EXTRACT[extract_chunk()<br/>Per chunk with gleaning]
        EXTRACT --> BOOK_JSON[data/graph/extractions/*.json<br/>Per book extraction results]
        BOOK_JSON --> MERGE[merge_extractions()]
        MERGE --> CONSOLIDATE[consolidate_entities()<br/>consolidate_relationships()]
        CONSOLIDATE --> FINAL[data/graph/extraction_results.json]
    end

    subgraph Stage6b["Stage 6b: Neo4j + Communities"]
        FINAL --> UPLOAD[upload_extraction_results()]
        UPLOAD --> NEO4J[(Neo4j<br/>Entity nodes + RELATED_TO edges)]
        NEO4J --> PROJECT[project_graph()<br/>GDS undirected projection]
        PROJECT --> LEIDEN[run_leiden()<br/>seed=42, concurrency=1]
        LEIDEN --> CHECKPOINT[leiden_checkpoint.json]
        LEIDEN --> WRITE_COMM[write_communities_to_neo4j()]
        LEIDEN --> PAGERANK[compute_pagerank()<br/>damping=0.85]
        PAGERANK --> WRITE_PR[write_pagerank_to_neo4j()]
        LEIDEN --> SUMMARIZE[summarize_community()<br/>Per community LLM + embedding]
        SUMMARIZE --> WEAVIATE[(Weaviate Community Collection)]
        SUMMARIZE --> COMM_JSON[communities.json]
    end

    style NEO4J fill:#4db33d,color:white
    style WEAVIATE fill:#fa8072
```

### Key Differences from Reference

1. **Chunking Strategy**: RAGLab uses section-aware chunking (800 tokens) vs reference's 600 tokens
2. **Output Format**: JSON files vs Parquet columnar storage
3. **Graph Storage**: Native Neo4j graph database vs Parquet entity/relationship tables
4. **Deterministic Leiden**: Uses `seed=42` and `concurrency=1` for reproducibility

---

## 4. Entity & Relationship Extraction

### Extraction Process

```mermaid
flowchart LR
    subgraph Chunk["Per Chunk"]
        TEXT[Chunk Text] --> PROMPT[GRAPHRAG_CHUNK_EXTRACTION_PROMPT<br/>+ entity_types from YAML]
        PROMPT --> LLM[Claude 3 Haiku<br/>temperature=0.0]
        LLM --> |Pydantic JSON| RESULT[ExtractionResult<br/>entities + relationships]
    end

    subgraph Gleaning["Gleaning Loop"]
        RESULT --> CHECK{_should_continue_gleaning?}
        CHECK --> |Y| CONTINUE[GRAPHRAG_CONTINUE_PROMPT]
        CONTINUE --> LLM
        CHECK --> |N| DEDUP[_deduplicate_entities]
    end

    subgraph Filter["Strict Mode"]
        DEDUP --> STRICT{GRAPHRAG_STRICT_MODE?}
        STRICT --> |Yes| FILTER[filter_entities_strict<br/>Remove non-matching types]
        STRICT --> |No| PASS[Pass through]
    end
```

### RAGLab vs Reference: Extraction Format

**Microsoft Reference** (delimited tuples):
```
(\"entity\"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>Description)
##
(\"relationship\"<|>SOURCE<|>TARGET<|>Description<|>9)
<|COMPLETE|>
```

**RAGLab** (Pydantic JSON Schema):
```python
class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str

class ExtractedRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    weight: float  # 0.0-1.0
```

### Entity Normalization

RAGLab uses a more sophisticated normalization than exact string matching:

```python
# src/graph/schemas.py:72-101
def normalized_name(self) -> str:
    name = self.name.strip()
    # Unicode normalization (café → cafe)
    name = unicodedata.normalize('NFKC', name)
    name = name.lower()

    # Remove leading/trailing stopwords
    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()

    name = ' '.join(words)
    # Strip punctuation
    name = re.sub(r'[^\w\s]', '', name)
    return ' '.join(name.split())
```

**Stopwords removed**: `{'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and'}`

### Entity Types Configuration

**File**: `src/graph/graphrag_types.yaml`

```yaml
entity_types:
  # Generic (2)
  - PERSON           # researchers, philosophers, historical figures
  - WORK             # books, papers, texts

  # Neuroscience (3)
  - BRAIN_STRUCTURE  # regions, systems
  - CHEMICAL         # neurotransmitters, hormones
  - DISORDER         # clinical conditions

  # Psychology bridge (2)
  - MENTAL_STATE     # internal experiences
  - BEHAVIOR         # observable patterns

  # Philosophy (1)
  - PRINCIPLE        # ideas, theories, virtues
```

**Comparison**: Microsoft suggests 4 types (PERSON, ORGANIZATION, LOCATION, EVENT). RAGLab uses **8 domain-specific types** tailored to the dual-domain corpus (neuroscience + philosophy).

---

## 5. Community Detection (Leiden)

### Implementation

**File**: `src/graph/community.py`

```mermaid
flowchart TB
    subgraph Projection["Graph Projection"]
        NEO4J[(Neo4j Graph)] --> PROJECT[gds.graph.project]
        PROJECT --> GDS_GRAPH[In-Memory GDS Graph<br/>name='graphrag']
    end

    subgraph Algorithm["Leiden Algorithm"]
        GDS_GRAPH --> LEIDEN[gds.leiden.stream]
        LEIDEN --> |Deterministic| CONFIG["gamma=1.0<br/>maxLevels=10<br/>randomSeed=42<br/>concurrency=1"]
        CONFIG --> RESULT[node_communities<br/>with intermediateCommunityIds]
    end

    subgraph Hierarchy["Hierarchy Parsing"]
        RESULT --> PARSE[parse_leiden_hierarchy]
        PARSE --> L0["Level 0 (L0)<br/>Coarsest - corpus themes"]
        PARSE --> L1["Level 1 (L1)<br/>Domain themes"]
        PARSE --> L2["Level 2 (L2)<br/>Finest - specific topics"]
    end

    subgraph Persist["Persistence"]
        RESULT --> CHECKPOINT[leiden_checkpoint.json<br/>For crash recovery]
        RESULT --> WRITE[write_communities_to_neo4j<br/>SET e.community_id]
    end

    style GDS_GRAPH fill:#4db33d,color:white
```

### Key Configuration

| Parameter | Microsoft Default | RAGLab Value | Notes |
|-----------|------------------|--------------|-------|
| `resolution` (gamma) | Configurable | **1.0** | Higher = more, smaller communities |
| `max_levels` | Configurable | **10** | Maximum hierarchy depth |
| `seed` | Optional | **42** | Fixed for determinism |
| `concurrency` | Multi-threaded | **1** | Single-threaded for reproducibility |

### Hierarchy Levels

RAGLab follows Microsoft's convention:
- **L0**: Coarsest (corpus-wide themes) - used for global queries
- **L1**: Medium (domain-level themes)
- **L2**: Finest (specific topics) - used for local queries

**Note**: Neo4j GDS returns `intermediateCommunityIds` where index 0 = coarsest, which matches Microsoft's L0 = coarsest convention.

### Crash Recovery

RAGLab implements deterministic Leiden for crash-proof design:

```python
# src/graph/community.py:117-119
leiden_result = run_leiden(
    gds, graph,
    seed=GRAPHRAG_LEIDEN_SEED,       # 42
    concurrency=GRAPHRAG_LEIDEN_CONCURRENCY,  # 1
)
save_leiden_checkpoint(leiden_result)  # Immediate checkpoint
```

Resume from checkpoint:
```bash
python -m src.stages.run_stage_6b_neo4j --from summaries
```

---

## 6. Community Summarization

### Process (Microsoft Bottom-Up Approach)

RAGLab implements Microsoft's bottom-up community summarization algorithm (arXiv:2404.16130):

1. **Processing Order**: Finest level (L2) processed first, then L1, then L0 (coarsest)
2. **Child Summary Substitution**: When raw entity/relationship data exceeds token limit, child community summaries are substituted
3. **No Output Token Limit**: Summaries are allowed to be complete (no truncation)

```mermaid
flowchart TB
    subgraph Order["Bottom-Up Processing Order"]
        L2[Level 2<br/>Finest - specific topics] --> L1[Level 1<br/>Domain themes]
        L1 --> L0[Level 0<br/>Coarsest - corpus themes]
    end

    subgraph Input["Community Data"]
        MEMBERS[get_community_members<br/>Sorted by PageRank] --> CHECK{Raw tokens<br/>> 8000?}
        RELS[get_community_relationships] --> CHECK
    end

    subgraph Context["Context Building"]
        CHECK --> |No| RAW[build_community_context<br/>Raw entity/relationship data]
        CHECK --> |Yes| HIER[build_hierarchical_context<br/>Substitute child summaries]
        HIER --> |"[Sub-Community]"| CHILD[Child summaries from<br/>finer level]
    end

    subgraph Summarize["LLM Summarization"]
        RAW --> PROMPT1[GRAPHRAG_COMMUNITY_PROMPT]
        CHILD --> PROMPT2[GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT]
        PROMPT1 --> LLM[GPT-4o-mini<br/>temperature=0.3<br/>no max_tokens]
        PROMPT2 --> LLM
        LLM --> SUMMARY[Community Summary<br/>Complete, no truncation]
    end

    subgraph Store["Storage + Tracking"]
        SUMMARY --> TRACK[child_summaries dict<br/>for parent processing]
        SUMMARY --> WEAVIATE[(Weaviate)]
        SUMMARY --> JSON[communities.json]
    end

    style WEAVIATE fill:#fa8072
    style L2 fill:#90EE90
    style L1 fill:#FFD700
    style L0 fill:#87CEEB
```

### Context Building

Members are sorted by **PageRank** (not degree) for context building:

```python
# src/graph/community.py:508-551
def build_community_context(members, relationships, max_tokens=8000):
    lines = ["## Entities"]
    for member in members:  # Already sorted by PageRank
        desc = f" - {member.description}" if member.description else ""
        pr_note = f" [PR:{member.pagerank:.3f}]" if member.pagerank > 0 else ""
        lines.append(f"- {member.entity_name} ({member.entity_type}){pr_note}{desc}")

    if relationships:
        lines.append("\n## Relationships")
        for rel in relationships:
            lines.append(f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}")

    context = "\n".join(lines)
    # Truncate to max_tokens * 4 characters
```

### Comparison to Reference

| Aspect | Microsoft Reference | RAGLab |
|--------|--------------------|----|
| Context building | Bottom-up (leaf summaries → parents) | **Bottom-up with child substitution** |
| Member ordering | By `combined_degree` | By **PageRank** |
| Token limit | 8,000 | **8,000** (matches Microsoft) |
| Output token limit | None (complete summaries) | **None** (matches Microsoft) |
| Output format | JSON with findings array | Plain text summary |

---

## 7. Embeddings & Storage

### What Gets Embedded

| Artifact | Microsoft | RAGLab | Collection Name |
|----------|-----------|--------|-----------------|
| Entity descriptions | Yes | Yes | `Entity_section800_v1` |
| Text units (chunks) | Yes | Yes | `RAG_section800_embed3large_v1` |
| Community summaries | Yes | Yes | `Community_section800_v1` |

### Weaviate Collections

```mermaid
flowchart LR
    subgraph Collections["Weaviate Collections"]
        CHUNKS[(RAG_section800_embed3large_v1<br/>Chunk vectors)]
        COMMS[(Community_section800_v1<br/>Community summary vectors)]
        ENTS[(Entity_section800_v1<br/>Entity description vectors)]
    end

    subgraph Usage["Query Time Usage"]
        QUERY[User Query] --> ENTS
        QUERY --> COMMS

        ENTS --> |Entity extraction| NEO4J[(Neo4j)]
        NEO4J --> |Graph traversal| CHUNKS
        CHUNKS --> |Batch fetch| RESULTS[Final Results]
        COMMS --> |Global queries| MAPRED[Map-Reduce]
    end

    style CHUNKS fill:#fa8072
    style COMMS fill:#fa8072
    style ENTS fill:#fa8072
    style NEO4J fill:#4db33d,color:white
```

### Neo4j Graph Model

```
(:Entity {
    name: "dopamine",
    normalized_name: "dopamine",
    entity_type: "CHEMICAL",
    description: "Neurotransmitter involved in reward and motivation",
    source_chunk_ids: ["behave::chunk_1", "behave::chunk_5"],
    community_id: 42,
    pagerank: 0.0042,
    created_at: datetime()
})

-[:RELATED_TO {
    type: "MODULATES",
    description: "Dopamine modulates reward processing",
    weight: 0.95,
    source_chunk_ids: ["behave::chunk_2"],
    created_at: datetime()
}]->

(:Entity {...})
```

---

## 8. Local Search Query Pipeline

### Query Flow

```mermaid
flowchart TB
    subgraph EntityExtraction["Entity Extraction"]
        QUERY[User Query] --> EMBED[embed_texts]
        EMBED --> SEARCH[Weaviate Vector Search<br/>Entities collection<br/>top_k=10, min_similarity=0.3]
        SEARCH --> ENTITIES[Matched Entities]

        ENTITIES --> |empty?| LLM_FALLBACK[LLM Extraction<br/>extract_query_entities_llm]
        LLM_FALLBACK --> ENTITIES

        ENTITIES --> |empty?| REGEX[Regex Fallback<br/>Capitalized words]
    end

    subgraph GraphTraversal["Neo4j Traversal"]
        ENTITIES --> VALIDATE[find_entities_by_names<br/>Validate in Neo4j]
        VALIDATE --> TRAVERSE[find_entity_neighbors<br/>max_hops=2, limit=50]
        TRAVERSE --> GRAPH_CHUNKS[Graph Chunk IDs<br/>from source_chunk_ids]
    end

    subgraph ChunkRetrieval["Chunk Retrieval (Graph-Only)"]
        GRAPH_CHUNKS --> FETCH[fetch_chunks_by_ids<br/>Batch retrieval from Weaviate]
        FETCH --> RANK_GRAPH[_build_graph_ranked_list<br/>Sort by combined_degree]
        RANK_GRAPH --> RESULTS[Ranked Results<br/>Hub entities = more informative]
    end

    subgraph CommunityContext["Community Context"]
        VALIDATE --> COMM_IDS[get_entity_community_ids<br/>From matched entities]
        COMM_IDS --> LOAD_COMM[Load communities<br/>by membership]
        LOAD_COMM --> CONTEXT[Community Summaries]
    end

    RESULTS --> ANSWER[Answer Generation]
    CONTEXT --> ANSWER

    style NEO4J fill:#4db33d,color:white
```

### Key Differences from Reference

| Aspect | Microsoft Reference | RAGLab |
|--------|--------------------|----|
| Entity matching | Embedding similarity | Embedding + LLM fallback + Regex |
| Relationship ranking | `combined_degree` | `combined_degree` (start + neighbor degree) |
| Context composition | Token budget allocation | combined_degree ranking (no explicit budget) |
| Community retrieval | In token budget | By entity membership (separate context) |

### Combined Degree Ranking

RAGLab ranks chunks by `combined_degree` (Microsoft's approach):

```python
# combined_degree = start_degree + neighbor_degree
# Higher combined_degree = hub entities = more informative chunks
# Chunks reached via multiple entities use MAX combined_degree
```

This aligns with Microsoft's relationship prioritization strategy.

---

## 9. Global Search (Map-Reduce)

### Pipeline

```mermaid
flowchart TB
    subgraph Classification["Query Classification"]
        QUERY[User Query] --> CLASSIFY[classify_query<br/>LLM: "local" or "global"]
        CLASSIFY --> |global| RETRIEVE[retrieve_communities_for_map_reduce<br/>All L0 communities]
        CLASSIFY --> |local| LOCAL[Standard hybrid retrieval]
    end

    subgraph MapPhase["Map Phase (Parallel)"]
        RETRIEVE --> C1[Community 1]
        RETRIEVE --> C2[Community 2]
        RETRIEVE --> CN[Community N]

        C1 --> |async| MAP1[LLM: GRAPHRAG_MAP_PROMPT<br/>max_tokens=300]
        C2 --> |async| MAP2[LLM]
        CN --> |async| MAPN[LLM]

        MAP1 --> P1[Partial Answer 1]
        MAP2 --> P2[Partial Answer 2]
        MAPN --> PN[Partial Answer N]
    end

    subgraph Filter["Filter"]
        P1 --> FILTER{Not relevant?}
        P2 --> FILTER
        PN --> FILTER
        FILTER --> |Yes| DISCARD[Discard]
        FILTER --> |No| KEEP[Keep for reduce]
    end

    subgraph ReducePhase["Reduce Phase"]
        KEEP --> REDUCE[LLM: GRAPHRAG_REDUCE_PROMPT<br/>max_tokens=500]
        REDUCE --> FINAL[Final Synthesized Answer]
    end

    style LOCAL fill:#90EE90
```

### Map-Reduce Configuration

| Parameter | Microsoft Reference | RAGLab |
|-----------|--------------------|----|
| Community level | Configurable (C0-C3) | **L0 only** (coarsest) |
| Map concurrency | asyncio.Semaphore(32) | `asyncio.gather` (no limit) |
| Map max tokens | Configurable | 300 |
| Reduce max tokens | Configurable | 500 |
| Score filtering | Remove score=0 | Remove "Not relevant" responses |

### Query Classification Prompt

```python
# src/prompts.py:242-256
GRAPHRAG_CLASSIFICATION_PROMPT = """Classify this query as 'local' or 'global'.

LOCAL queries ask about specific entities, concepts, or facts:
- "What is dopamine?"
- "How does the prefrontal cortex affect decision-making?"

GLOBAL queries ask about themes, patterns, or overviews:
- "What are the main themes in this corpus?"
- "How do neuroscience and philosophy approaches differ?"

Query: {query}

Respond with ONLY the word 'local' or 'global'."""
```

---

## 10. Data Models

### Pydantic Models

```python
# src/graph/schemas.py

class GraphEntity(BaseModel):
    name: str
    entity_type: str
    description: str = ""
    source_chunk_id: str = ""

    def normalized_name(self) -> str: ...
    def to_neo4j_properties(self) -> dict: ...

class GraphRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str = ""
    weight: float = 1.0  # 0.0-1.0 (not 1-10 like Microsoft)
    source_chunk_id: str = ""

class Community(BaseModel):
    community_id: str       # "community_L0_42"
    level: int              # 0 = coarsest
    parent_id: Optional[str]
    members: list[CommunityMember]
    member_count: int
    relationships: list[CommunityRelationship]
    relationship_count: int
    summary: str
    embedding: Optional[list[float]]

class CommunityMember(BaseModel):
    entity_name: str
    entity_type: str
    description: str = ""
    degree: int = 0
    pagerank: float = 0.0
```

### Comparison to Microsoft

| Field | Microsoft | RAGLab | Notes |
|-------|-----------|--------|-------|
| Entity `rank` | Degree-based | **PageRank** | Different metric |
| Relationship `weight` | 1-10 (LLM assigned) | **0.0-1.0** (confidence) | Different scale |
| Community `rating` | 0-10 impact score | **Not used** | Omitted |
| Community `findings` | Array of insights | **Plain summary** | Simpler format |

---

## 11. Configuration Reference

### GraphRAG Settings (`src/config.py`)

```python
# Extraction
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-3-haiku"
GRAPHRAG_SUMMARY_MODEL = "openai/gpt-4o-mini"
GRAPHRAG_MAX_EXTRACTION_TOKENS = 4000
GRAPHRAG_MAX_ENTITIES = 10        # Per chunk (paper: no limit)
GRAPHRAG_MAX_RELATIONSHIPS = 7    # Per chunk (paper: no limit)
GRAPHRAG_MAX_GLEANINGS = 1        # Additional extraction passes
GRAPHRAG_STRICT_MODE = True       # Filter non-matching entity types

# Leiden Community Detection
GRAPHRAG_LEIDEN_RESOLUTION = 1.0  # gamma parameter
GRAPHRAG_LEIDEN_MAX_LEVELS = 10   # Maximum hierarchy depth
GRAPHRAG_MIN_COMMUNITY_SIZE = 3   # Minimum members to summarize
GRAPHRAG_LEIDEN_SEED = 42         # For determinism
GRAPHRAG_LEIDEN_CONCURRENCY = 1   # Single-threaded for reproducibility

# Community Summarization (Microsoft bottom-up approach)
# GRAPHRAG_MAX_SUMMARY_TOKENS removed - allow complete summaries
GRAPHRAG_MAX_CONTEXT_TOKENS = 8000  # Matches Microsoft
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3  # L0, L1, L2

# PageRank
GRAPHRAG_PAGERANK_DAMPING = 0.85
GRAPHRAG_PAGERANK_ITERATIONS = 20

# Query-Time Entity Extraction
GRAPHRAG_ENTITY_EXTRACTION_TOP_K = 10
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3
GRAPHRAG_USE_EMBEDDING_EXTRACTION = True

# Local Search
GRAPHRAG_TOP_COMMUNITIES = 3
GRAPHRAG_TRAVERSE_DEPTH = 2       # Max hops for entity traversal
GRAPHRAG_RRF_K = 60               # RRF constant

# Map-Reduce
GRAPHRAG_MAP_MAX_TOKENS = 300
GRAPHRAG_REDUCE_MAX_TOKENS = 500
```

### Comparison Table

| Parameter | Microsoft Default | RAGLab Value |
|-----------|------------------|--------------|
| Chunk size | 300 (default), 600 (paper) | 800 |
| Gleanings | 0-1 | 1 |
| Max entities/chunk | Unlimited | 10 |
| Max relationships/chunk | Unlimited | 7 |
| Leiden resolution | Configurable | 1.0 |
| PageRank damping | Not specified | 0.85 |
| Max context tokens (summary) | 8,000 | **8,000** (matches) |
| Text unit prop | 0.5 (50%) | N/A (RRF merge) |
| Community prop | 0.15 (15%) | N/A (separate context) |
| RRF k | N/A | 60 |

---

## 12. Prompts Comparison

### Entity Extraction Prompt

**Microsoft** (delimited output):
```
Format each entity as ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
Format each relationship as ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>WEIGHT)
```

**RAGLab** (JSON output):
```python
# src/prompts.py:146-168
GRAPHRAG_CHUNK_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

ENTITY TYPES (use ONLY these): {entity_types}

For each entity:
- name: The entity name as it appears in text
- entity_type: One of the types above
- description: Brief description (under 15 words)

For each relationship:
- source_entity / target_entity: Entity names from above
- relationship_type: Free-form type (e.g., CAUSES, MODULATES)
- description: Brief description
- weight: 0.0-1.0 (strength/importance)

LIMITS: Up to {max_entities} entities and {max_relationships} relationships.

IMPORTANT: Respond ONLY with valid JSON:
{{"entities": [...], "relationships": [...]}}"""
```

### Gleaning Prompts

RAGLab implements gleaning similarly to Microsoft:

```python
# Loop check (expects Y/N)
GRAPHRAG_LOOP_PROMPT = """Based on the text and your previous extraction,
are there any important entities or relationships you may have missed?
Answer with ONLY 'Y' if there are missed entities, or 'N' if complete."""

# Continue extraction
GRAPHRAG_CONTINUE_PROMPT = """MANY entities and relationships were missed.
...
Extract ADDITIONAL entities and relationships that were missed."""
```

### Community Summary Prompt

**Microsoft** (structured JSON with findings):
```json
{
    "title": "...",
    "summary": "...",
    "rating": 5.0,
    "rating_explanation": "...",
    "findings": [{"summary": "...", "explanation": "..."}]
}
```

**RAGLab** (plain text):
```python
GRAPHRAG_COMMUNITY_PROMPT = """You are analyzing a community...

Community entities and their relationships:
{community_context}

Write a summary (2-3 short paragraphs, ~150-200 words) that:
1. Identifies the main theme
2. Explains key relationships
3. Highlights important details

Summary:"""
```

---

## 13. Code Snippets

### Entity Extraction with Gleaning

```python
# src/graph/extraction.py:120-203
def extract_chunk(chunk, model=GRAPHRAG_EXTRACTION_MODEL, max_gleanings=1):
    entity_types = get_entity_types_string()
    text = chunk["text"]

    # Initial extraction
    prompt = GRAPHRAG_CHUNK_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        text=text,
        max_entities=GRAPHRAG_MAX_ENTITIES,
        max_relationships=GRAPHRAG_MAX_RELATIONSHIPS,
    )
    result = call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )

    all_entities = list(result.entities)
    all_relationships = list(result.relationships)

    # Gleaning loop
    for i in range(max_gleanings):
        if not _should_continue_gleaning(text, all_entities, all_relationships, model):
            break
        additional = _glean_additional_entities(...)
        all_entities.extend(additional.entities)
        all_relationships.extend(additional.relationships)

    # Deduplicate within chunk
    final_entities = _deduplicate_entities(all_entities)
    final_relationships = _deduplicate_relationships(all_relationships)

    # Apply strict mode filtering
    if GRAPHRAG_STRICT_MODE:
        filtered, discarded = filter_entities_strict(result.entities, allowed_types)
```

### Hybrid Graph Retrieval with RRF

```python
# src/graph/query.py:602-752
def hybrid_graph_retrieval(query, driver, vector_results, top_k=10):
    # Get graph chunk IDs and metadata
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Community context by entity membership (Microsoft approach)
    query_entities = graph_meta.get("query_entities", [])
    community_context = retrieve_community_context_by_membership(
        entity_names=query_entities,
        driver=driver,
    )

    # Fetch ALL graph-discovered chunks from Weaviate
    all_graph_chunks = fetch_chunks_by_ids(graph_chunk_ids)

    # Build graph-ranked list ordered by combined_degree (Microsoft approach)
    graph_ranked_results = _build_graph_ranked_list(graph_context, all_graph_chunks)

    # RRF merge: chunks in BOTH lists get boosted
    rrf_result = reciprocal_rank_fusion(
        result_lists=[vector_search_results, graph_ranked_results],
        query_types=["vector", "graph"],
        k=GRAPHRAG_RRF_K,  # 60
        top_k=top_k,
    )
```

### Deterministic Leiden

```python
# src/graph/community.py:112-184
def run_leiden(gds, graph, resolution=1.0, max_levels=10, seed=42, concurrency=1):
    """Deterministic Leiden with fixed seed and single-threaded execution."""
    result = gds.leiden.stream(
        graph,
        gamma=resolution,
        maxLevels=max_levels,
        includeIntermediateCommunities=True,  # For hierarchy
        randomSeed=seed,           # Fixed seed
        concurrency=concurrency,   # Single-threaded
    )

    node_communities = []
    for record in result.itertuples():
        node_communities.append({
            "node_id": record.nodeId,
            "community_id": record.communityId,
            "intermediate_ids": list(record.intermediateCommunityIds),
        })

    return {
        "community_count": len(unique_communities),
        "node_communities": node_communities,
        "seed": seed,
    }
```

---

## 14. What's Different

### 1. Output Format

| Component | Microsoft | RAGLab | Rationale |
|-----------|-----------|--------|-----------|
| Extraction output | Delimited tuples | **Pydantic JSON** | Structured validation |
| Storage format | Parquet | **JSON + Neo4j** | Graph-native queries |
| Vector store | LanceDB | **Weaviate** | Production-ready |

### 2. Entity Matching

**Microsoft**: Exact string matching
**RAGLab**: Normalized matching (NFKC + lowercase + stopword removal)

```python
# "The Dopamine" → "dopamine"
# "café" → "cafe"
```

### 3. Relationship Weight Scale

**Microsoft**: 1-10 (LLM assigns strength)
**RAGLab**: 0.0-1.0 (confidence score)

### 4. Token Budget vs RRF

**Microsoft**: Explicit token budget allocation (50% text, 15% community, 35% entities/relationships)
**RAGLab**: RRF merge with separate community context

### 5. Community Context Retrieval

**Microsoft**: Part of token budget
**RAGLab**: By entity membership (separate from chunk context)

### 6. Relationship Prioritization

**Microsoft**: `combined_degree = degree(source) + degree(target)`
**RAGLab**: `combined_degree = start_degree + neighbor_degree` (Matches Microsoft)

Higher combined_degree indicates relationships involving "hub" entities that are
well-connected in the knowledge graph, providing more informative context.

### 7. Community Summary Format

**Microsoft**: Structured JSON with `title`, `rating`, `findings[]`
**RAGLab**: Plain text summary (150-200 words)

### 8. Graph Storage

**Microsoft**: No native graph DB (Parquet tables)
**RAGLab**: **Neo4j** with native graph queries and GDS algorithms

---

## 15. What's Not Implemented

### 1. DRIFT Search

Microsoft's Dynamic Reasoning and Inference with Flexible Traversal is **not implemented**:
- Iterative search with follow-up queries
- Confidence-based stopping
- Multi-phase exploration

### 2. Covariates/Claims

The optional claims extraction system is **not implemented**:
- Subject, object, type, status
- Claim verification

### 3. Token Budget Allocation

RAGLab uses RRF merge instead of explicit token budgets:
- No `text_unit_prop` (50%)
- No `community_prop` (15%)
- No relationship/entity budget

### 4. Community Impact Rating

The 0-10 impact severity rating is **not used**:
- No `rating` field
- No `rating_explanation`
- No `findings[]` array

### 5. Graph Embeddings (Node2Vec)

Optional graph structure embeddings:
- `embed_graph` block
- Node2Vec walks

### 6. Graph Pruning

Pre-processing to remove noise:
- `min_node_freq`
- `min_node_degree`
- `min_edge_weight_pct`

### 7. Oversample Scaler

Entity retrieval oversampling:
- `oversample_scaler` (default 2)
- Retrieve more candidates, then filter

---

## Summary

RAGLab implements the core GraphRAG pipeline with several enhancements:

**Strengths**:
- Native Neo4j graph storage (vs Parquet tables)
- Deterministic Leiden with crash recovery
- Pydantic validation for extraction
- Enhanced entity normalization
- RRF-based hybrid retrieval

**Simplifications**:
- Simpler community summary format
- No token budget allocation
- No DRIFT search
- No claims/covariates

**Design Philosophy**:
- Learning-focused with extensive documentation
- Production-ready storage (Weaviate, Neo4j)
- Modular stage-based pipeline
- Crash-proof with checkpoints
