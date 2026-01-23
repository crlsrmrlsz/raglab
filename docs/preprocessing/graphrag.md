# GraphRAG: Knowledge Graph Communities

[<- Query Decomposition](query-decomposition.md) | [Home](../../README.md)

When you ask "What are the main themes across all 19 books?", no single chunk contains this answer. Vector search finds chunks similar to your query, but similarity doesn't help when the answer requires synthesizing information scattered across thousands of pages. This is the **global query problem**: questions about patterns, themes, and relationships that span an entire corpus.

[GraphRAG](https://arxiv.org/abs/2404.16130) solves this by building a knowledge graph from your documents. Entities (people, concepts, brain regions) become nodes; relationships become edges. The Leiden algorithm then detects communities of densely connected entities. Each community gets an LLM-generated summary. For global queries, GraphRAG uses map-reduce over these summaries to synthesize corpus-wide answers.

**When GraphRAG helps:** Cross-document synthesis, thematic questions, relationship discovery, "compare authors' perspectives on X" queries.

**When it struggles:** Simple factual lookups (overkill), frequently updated corpora (requires reindexing), small datasets (setup cost exceeds benefit).



## The GraphRAG Paper and Algorithm

**Paper:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
**Authors:** Edge et al. (Microsoft Research)
**Published:** April 2024 ([arXiv:2404.16130](https://arxiv.org/abs/2404.16130))

The authors wanted RAG systems that could answer questions requiring corpus-wide understanding—what they call "query-focused summarization" at scale. Standard RAG retrieves relevant chunks, but relevance isn't synthesis. Their solution: transform documents into a hierarchical knowledge structure that can be queried at different abstraction levels.

The algorithm has two phases:

### Indexing (Offline)

1. **Entity extraction.** An LLM processes each chunk, extracting entities (named items with types like PERSON, CONCEPT, BRAIN_REGION) and relationships (connections between entities with descriptions and strength scores). The paper uses multiple extraction rounds with "gleaning"—prompting the LLM to find missed entities.

2. **Knowledge graph construction.** Entities become nodes; relationships become weighted edges. Duplicate entity names are merged via string normalization. The result is a connected graph where each node tracks which source chunks mentioned it.

3. **Community detection.** The Leiden algorithm partitions the graph into communities of densely connected entities. Unlike Louvain (its predecessor), Leiden guarantees well-connected communities through a refinement phase. The algorithm runs recursively, producing a hierarchy: C0 (finest granularity, specific topics), C1 (medium, domain themes), C2 (coarsest, corpus-wide patterns).

4. **Community summarization.** For each community, the LLM generates a summary describing its entities, relationships, and themes. Entities are sorted by PageRank (hub entities first). These summaries become the index for global queries.

### Query (Online)

**Local queries** (entity-focused): Extract entities from the query, traverse the graph to find related entities and their source chunks, merge with vector search results using RRF.

**Global queries** (thematic): Map-reduce over community summaries. The map phase generates partial answers from each relevant community in parallel. The reduce phase synthesizes these into a coherent final answer.

### Key Findings

**Community summaries work.** 72-83% win rate on comprehensiveness versus baseline RAG. The hierarchical structure captures corpus-level patterns that chunk retrieval misses entirely.

**8k context beats larger.** Surprisingly, 8k token context windows outperformed 16k, 32k, and 64k on comprehensiveness (58.1% average win rate). Larger contexts trigger "lost in the middle" effects where models fail to utilize information in long prompts.

**Leiden over Louvain.** Louvain can produce disconnected communities (pathological cases). Leiden's refinement phase guarantees connectivity, making it robust for downstream summarization.

**Token efficiency.** Root-level community summaries (C0) require 9-43x fewer tokens than processing source text directly, amortizing indexing cost across queries.



## RAGLab Implementation

RAGLab implements GraphRAG across two stages with several adaptations for the dual-domain corpus (neuroscience + philosophy):

```bash
# Stage 4.5: Entity extraction with auto-discovered types
python -m src.stages.run_stage_4_5_autotune --strategy semantic_std3

# Stage 6b: Neo4j upload + Leiden + summaries + entity embeddings
docker compose up -d neo4j
python -m src.stages.run_stage_6b_neo4j
```

**Semantic chunking for entity extraction.** The paper uses fixed 300-token chunks, but RAGLab uses [semantic chunking](../chunking/semantic-chunking.md) (std coefficient=3.0) for GraphRAG. While entity deduplication happens at merge time regardless of chunk boundaries, *relationships* are extracted per-chunk—entities must appear together to form edges. Semantic chunking keeps related concepts together, improving relationship capture. The conservative std=3.0 threshold preserves 98% of sections as single chunks (avg 665 tokens for neuroscience, 1331 for philosophy), accepting some longer chunks to avoid splitting mid-argument.

**Auto-tuning entity types.** Rather than hardcoding entity types, Stage 4.5 uses open-ended extraction and discovers types from the corpus. A consolidation step merges similar types (RESEARCHER/SCIENTIST/ACADEMIC -> RESEARCHER). The `stratified` strategy balances types across corpora—preventing neuroscience (larger) from dominating philosophy.

**Deterministic Leiden.** Stage 6b runs Leiden with `seed=42` and `concurrency=1`, guaranteeing identical community assignments on every run. This enables crash recovery: if summarization fails midway, re-running picks up where it stopped because community IDs remain stable.

**Dual extraction at query time.** Entity extraction uses embedding similarity search (~50ms) with LLM fallback (~1-2s) for complex conceptual queries. The embedding approach searches pre-indexed entity descriptions in Weaviate; the LLM approach uses the discovered entity types to guide extraction.

**RRF merge with graph boost.** Hybrid retrieval combines vector search results with graph traversal results using Reciprocal Rank Fusion (k=60). Chunks appearing in both lists get boosted scores—they're semantically similar AND structurally related through the knowledge graph.

Configuration in `src/config.py`:

```python
GRAPHRAG_TRAVERSE_DEPTH = 2        # Hops from query entities
GRAPHRAG_TOP_COMMUNITIES = 3       # Community summaries in context
GRAPHRAG_RRF_K = 60                # RRF constant
GRAPHRAG_LEIDEN_RESOLUTION = 1.0   # Higher = more communities
GRAPHRAG_LEIDEN_SEED = 42          # Deterministic (crash recovery)
```

**Output collections:**
- `Entity_section800_v1` — Entity descriptions (for query extraction)
- `Community_section800_v1` — Community summaries (for thematic context)



## Navigation

**Next:** [Reranking](reranking.md) — Cross-encoder for precision

**Related:**
- [RAPTOR](../chunking/raptor.md) — Alternative hierarchy via clustering (no Neo4j)
- [HyDE](hyde.md) — Simpler query transformation (no graph)
- [Query Decomposition](query-decomposition.md) — Sub-query strategy
- [Preprocessing Overview](README.md) — Strategy comparison
- [Paper (arXiv)](https://arxiv.org/abs/2404.16130) — Original GraphRAG research
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — Official implementation
