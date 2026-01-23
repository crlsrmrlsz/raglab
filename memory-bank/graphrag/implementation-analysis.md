# GraphRAG Implementation Analysis: RAGLab vs. Original Paper & Reference

**Date:** 2026-01-23
**Status:** Analysis Complete - Action Items Pending
**Sources:** arXiv:2404.16130, Microsoft GraphRAG GitHub, RAGLab codebase

---

## Executive Summary

After reviewing the GraphRAG theory in memory-bank, the original arXiv paper (2404.16130), Microsoft's reference implementation documentation, and the RAGLab codebase, I've identified several deviations from the original design. Some are intentional simplifications (documented as such), while others may be unintentional gaps.

---

## 1. Entity Extraction: Missing Gleaning

**Paper/Reference Implementation:**
- Uses **multiple extraction passes** called "gleaning"
- Default: single pass + 0-3 gleaning iterations
- Three prompts: `GRAPH_EXTRACTION_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT`
- Each gleaning pass finds entities missed in previous passes
- Increases entity yield per chunk significantly

**RAGLab Implementation (`extraction.py:82-98`):**
```python
def extract_chunk(chunk, model):
    prompt = GRAPHRAG_OPEN_EXTRACTION_PROMPT.format(...)
    return call_structured_completion(...)  # Single pass only
```

**Deviation:** RAGLab performs **single-pass extraction only**. No gleaning/continuation prompts exist.

**Impact:**
- Lower entity yield per chunk (paper reports ~3 entities/chunk without gleaning)
- May miss entities that only become apparent when the LLM re-examines the text
- Documentation acknowledges this: `graphrag-reference.md:381` lists "Self-reflection loop | 3 iterations | Single pass | Skipped (3x cost)"

**Recommendation:** Consider adding optional gleaning with configurable `max_gleanings` parameter for higher-quality extraction.

**Action Item:** [ ] Implement optional gleaning with `GRAPHRAG_MAX_GLEANINGS` config parameter

---

## 2. Entity Resolution: Adequate but Simplified

**Paper/Reference:**
- String normalization + description merging
- Multiple descriptions for the same entity are **summarized by LLM** into a single coherent description
- Entity deduplication happens at graph construction time

**RAGLab Implementation (`schemas.py:72-101`):**
```python
def normalized_name(self) -> str:
    name = unicodedata.normalize('NFKC', name)
    name = name.lower()
    # Remove stopwords, punctuation
    return ' '.join(name.split())
```

**RAGLab Neo4j Merge (`neo4j_client.py`):**
- Uses `MERGE` on `normalized_name` to combine duplicate entities
- Descriptions are concatenated or overwritten (not LLM-summarized)

**Deviation:** Missing LLM-based **entity summarization** that combines multiple descriptions into a coherent single description.

**Impact:** Entities mentioned across many chunks may have repetitive or incomplete descriptions rather than synthesized summaries.

**Recommendation:** Add an optional summarization step after extraction that consolidates entity descriptions.

**Action Item:** [ ] Add entity description summarization step in Stage 6b

---

## 3. Community Detection: Correctly Implemented

**Paper/Reference:**
- Leiden algorithm (improvement over Louvain)
- Hierarchical detection producing multiple levels (C0, C1, C2)
- Deterministic with fixed seed for reproducibility

**RAGLab Implementation (`community.py:112-183`):**
```python
def run_leiden(gds, graph, resolution=1.0, seed=42, concurrency=1):
    result = gds.leiden.stream(
        graph,
        gamma=resolution,
        maxLevels=max_levels,
        includeIntermediateCommunities=True,
        randomSeed=seed,
        concurrency=concurrency,  # Single-threaded
    )
```

**Assessment:** ✅ **Correctly implemented** with:
- Deterministic seed (42) + single-threaded execution
- Hierarchical levels (C0, C1, C2 up to `GRAPHRAG_MAX_HIERARCHY_LEVELS=3`)
- Crash recovery via checkpoint

**Action Item:** None - implementation is correct

---

## 4. Community Summarization: Minor Deviation

**Paper/Reference:**
- Entities sorted by **PageRank** (hub entities first)
- Summary includes: title, executive summary, key insights, importance ratings
- Two-tier: full reports + shorthand summaries

**RAGLab Implementation (`community.py:554-599`):**
```python
def summarize_community(members, relationships, model):
    # Build context (entities already sorted by PageRank)
    context = build_community_context(members, relationships)
    prompt = GRAPHRAG_COMMUNITY_PROMPT.format(community_context=context)
    summary = call_chat_completion(...)
```

**The Prompt (`prompts.py:125-138`):**
```python
GRAPHRAG_COMMUNITY_PROMPT = """...
Write a summary (2-3 short paragraphs, ~150-200 words) that:
1. Identifies the main theme or topic
2. Explains the key relationships
3. Highlights important details, names, and specific findings
"""
```

**Deviations:**
1. No **structured title** generation (paper produces titled reports)
2. No **importance ratings** (paper assigns 0-100 scores)
3. No **two-tier summarization** (only generates one summary level)

**Impact:** Summaries may be less structured, making programmatic use harder. However, the core functionality (thematic summary) is preserved.

**Recommendation:** Enhance prompt to request structured output with title + importance rating for better global query routing.

**Action Item:** [ ] Enhance GRAPHRAG_COMMUNITY_PROMPT for structured output (title, importance rating)

---

## 5. Local Search: Partially Implemented

**Paper/Reference:**
- Extract entities from query → Vector search entity descriptions
- Fan out 1-2 hops from matched entities
- Collect source text chunks from traversed entities
- Combine with traditional vector search via RRF

**RAGLab Implementation (`query.py:497-642`):**

✅ **Correct:**
- Entity extraction via embedding similarity + LLM fallback (`query_entities.py`)
- Graph traversal from matched entities (`GRAPHRAG_TRAVERSE_DEPTH=2`)
- RRF merge of vector + graph results
- Graph chunks ranked by path_length (shorter = higher)

**Minor Issue (`query.py:422-424`):**
```python
# Step 1: Extract entities from query using LLM
extracted = extract_query_entities_llm(query)  # Uses LLM, not embedding
```

In `get_graph_chunk_ids()`, the code calls `extract_query_entities_llm()` directly instead of the hybrid `extract_query_entities()` which would try embedding first. This may be intentional for consistency but differs from the pattern used elsewhere.

**Assessment:** ✅ Local search is **well-implemented** with proper RRF boosting for overlapping chunks.

**Action Item:** [ ] Review `get_graph_chunk_ids()` - consider using `extract_query_entities()` for consistency

---

## 6. Global Search (Map-Reduce): Correctly Implemented

**Paper/Reference:**
- Classify query as local vs global
- For global: Map phase generates partial answer per community
- Reduce phase synthesizes into final answer

**RAGLab Implementation (`map_reduce.py`):**

✅ **Correct:**
- `classify_query()` determines local/global
- `should_use_map_reduce()` heuristic based on entity count + classification
- Async map phase with parallel LLM calls
- Reduce phase synthesizes partial answers

**Configuration (`config.py`):**
```python
GRAPHRAG_MAP_REDUCE_TOP_K = 5  # Communities for map phase
GRAPHRAG_MAP_MAX_TOKENS = 300
GRAPHRAG_REDUCE_MAX_TOKENS = 500
```

**Assessment:** ✅ Map-reduce is **well-implemented** following the paper's approach.

**Action Item:** None - implementation is correct

---

## 7. Claims Extraction: Intentionally Skipped

**Paper/Reference:**
- Optional claims extraction: factual statements with time bounds and verification status
- Stored as "Covariates"

**RAGLab:**
- Not implemented
- Documented: `graphrag-reference.md:383` - "Claims extraction | Verifiable facts | Not implemented | Skipped (scope)"

**Assessment:** ✅ Intentional omission, well-documented.

**Action Item:** None - intentionally skipped

---

## 8. DRIFT Search: Not Implemented

**Paper/Reference (October 2024):**
- Hybrid approach combining local search with community context
- Iterative refinement with "follow-up" searches
- 78% improvement over local search alone

**RAGLab:**
- Not implemented
- Only local and global search modes available

**Impact:** DRIFT was released after the initial GraphRAG paper. This is a **feature gap**, not a deviation from the original methodology.

**Action Item:** [ ] Consider implementing DRIFT search as future enhancement

---

## 9. Chunk Size Discrepancy

**Paper/Reference:**
- Default: 300 tokens per chunk
- Positive results with 1200 tokens + single glean

**RAGLab (`config.py`):**
```python
MAX_CHUNK_TOKENS = 800  # Target chunk size
```

**Analysis:** RAGLab uses 800-token chunks, which falls between the paper's defaults. Combined with no gleaning, this may result in:
- Fewer extraction passes per document
- Potentially more entities per chunk (larger context)

**Assessment:** This is a reasonable adaptation, not necessarily a problem.

**Action Item:** None - reasonable adaptation

---

## 10. Query-Time Entity Extraction Order

**RAGLab (`query_entities.py:206-272`):**
```python
def extract_query_entities(query, driver, use_embedding=True, use_llm_fallback=True):
    if use_embedding:
        entities = extract_query_entities_embedding(query)
    if not entities and use_llm_fallback:
        entities = extract_query_entities_llm(query)
    if not entities:
        # Regex fallback for capitalized words
```

**Assessment:** ✅ This cascading approach (embedding → LLM → regex) is a **good adaptation** that balances speed (~50ms embedding) with quality (LLM fallback).

**Action Item:** None - good implementation

---

## Summary Table

| Feature | Paper/Reference | RAGLab | Status |
|---------|----------------|--------|--------|
| Entity gleaning (multi-pass) | 0-3 passes | Single pass | ⚠️ **Gap** |
| Entity summarization | LLM merges descriptions | Concatenation/overwrite | ⚠️ **Gap** |
| Leiden community detection | Hierarchical | Hierarchical (C0-C2) | ✅ Correct |
| PageRank for entity ordering | Yes | Yes | ✅ Correct |
| Community summarization | Titled reports + ratings | Plain text summary | ⚠️ Minor |
| Local search (RRF merge) | Yes | Yes | ✅ Correct |
| Global search (map-reduce) | Yes | Yes (async) | ✅ Correct |
| Claims extraction | Optional | Skipped | ✅ Documented |
| DRIFT search | Available | Not implemented | ℹ️ Feature gap |

---

## Action Items Summary

### High Priority (Core Algorithm Gaps)
1. [ ] **Gleaning:** Implement optional multi-pass extraction with `GRAPHRAG_MAX_GLEANINGS` config
2. [ ] **Entity Summarization:** Add LLM-based description consolidation in Stage 6b

### Medium Priority (Quality Improvements)
3. [ ] **Community Prompts:** Enhance for structured output (title, importance rating 0-100)
4. [ ] **Entity Extraction Consistency:** Review `get_graph_chunk_ids()` to use `extract_query_entities()`

### Low Priority (Future Enhancements)
5. [ ] **DRIFT Search:** Consider implementing for complex queries needing breadth + depth

---

## Technical Details for Implementation

### 1. Gleaning Implementation Sketch

```python
# config.py
GRAPHRAG_MAX_GLEANINGS = 1  # 0 = disabled, 1-3 for more entities

# prompts.py
GRAPHRAG_CONTINUE_PROMPT = """MANY entities and relationships were missed in the last extraction.
Add any additional entities and relationships that were not captured.

Remember: Only output NEW entities/relationships not already listed.

Previous extraction:
{previous_output}

Original text:
{text}

Additional entities and relationships (JSON):"""

GRAPHRAG_LOOP_PROMPT = """Given the extraction so far, are there still clearly missing entities?
Answer YES or NO only."""

# extraction.py
def extract_chunk_with_gleaning(chunk, model, max_gleanings=GRAPHRAG_MAX_GLEANINGS):
    # Initial extraction
    result = extract_chunk(chunk, model)
    all_entities = result.entities
    all_relationships = result.relationships

    for i in range(max_gleanings):
        # Check if more gleaning needed
        loop_response = call_chat_completion(GRAPHRAG_LOOP_PROMPT, ...)
        if "NO" in loop_response.upper():
            break

        # Continue extraction
        continue_result = call_structured_completion(
            GRAPHRAG_CONTINUE_PROMPT.format(
                previous_output=result.model_dump_json(),
                text=chunk["text"]
            ),
            response_model=OpenExtractionResult
        )
        all_entities.extend(continue_result.entities)
        all_relationships.extend(continue_result.relationships)

    return OpenExtractionResult(entities=all_entities, relationships=all_relationships)
```

### 2. Entity Summarization Sketch

```python
# After Neo4j upload, before community detection
def summarize_entity_descriptions(driver, model=GRAPHRAG_SUMMARY_MODEL):
    """Consolidate multiple descriptions per entity into coherent summaries."""

    # Find entities with multiple descriptions
    query = """
    MATCH (e:Entity)
    WHERE size(e.descriptions) > 1
    RETURN e.normalized_name as name, e.descriptions as descriptions
    """

    for record in driver.execute_query(query).records:
        combined = "\n".join(record["descriptions"])
        prompt = f"""Summarize these descriptions of "{record['name']}" into a single coherent description (1-2 sentences):

{combined}

Summary:"""

        summary = call_chat_completion(prompt, model=model, max_tokens=100)

        # Update entity
        driver.execute_query(
            "MATCH (e:Entity {normalized_name: $name}) SET e.description = $summary",
            name=record["name"], summary=summary
        )
```

### 3. Structured Community Prompt

```python
GRAPHRAG_COMMUNITY_PROMPT_STRUCTURED = """Analyze this community of related entities from a knowledge graph.

Community entities and relationships:
{community_context}

Provide a structured analysis in this exact JSON format:
{{
    "title": "Short descriptive title (5-10 words)",
    "importance": <0-100 score based on centrality and theme significance>,
    "summary": "2-3 paragraph summary covering main theme, key relationships, and important findings"
}}

Respond with valid JSON only."""
```

---

## References

- [GraphRAG Paper (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG Gleaning Issue #615](https://github.com/microsoft/graphrag/issues/615)
- [GraphRAG Dataflow Documentation](https://microsoft.github.io/graphrag/index/default_dataflow/)

---

*Last Updated: 2026-01-23*
