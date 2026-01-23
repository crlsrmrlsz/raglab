# Query Decomposition

[← HyDE](hyde.md) | [Home](../../README.md)

When you ask "How does stress affect both memory formation and decision-making?", a single search struggles because the question touches multiple concepts. Documents about stress and memory won't mention decision-making, and vice versa. A single embedding can't represent both facets well—it averages them into a middle ground that matches neither topic precisely.

Query Decomposition fixes this by breaking the question into focused sub-questions before searching. Instead of one embedding representing everything, each sub-question ("How does stress affect memory?" + "How does stress affect decision-making?") gets its own search. The results are pooled and a cross-encoder reranks them against the original question, surfacing the most relevant documents from both aspects.

The key insight: complex questions often contain multiple information needs. Decomposing them lets each need get precise retrieval, while reranking ensures the final results address the whole question.

**When decomposition helps:** Multi-hop questions spanning multiple concepts, comparison queries ("How does X differ from Y?"), questions with implicit sub-parts, and "what, how, why" chains where each aspect needs independent retrieval.

**When it struggles:** Simple factual lookups (adds unnecessary latency), cross-domain synthesis where the domains share vocabulary (HyDE may work better), and latency-critical applications (<500ms budget) since multiple searches are required.



## The Decomposition Paper and Algorithm

**Paper:** "Question Decomposition for Retrieval-Augmented Generation"
**Authors:** Ammann, Egger, Geissbühler (University of Applied Sciences, Switzerland)
**Published:** ACL SRW 2025 ([arXiv:2507.00355](https://arxiv.org/abs/2507.00355))

The authors studied how to improve retrieval for complex, multi-hop questions. Their solution: instead of forcing one search to handle multiple information needs, decompose the question and search for each need independently.

The algorithm works in five steps:

1. **Decompose the query.** An LLM breaks the original question into up to 5 sub-questions. Each sub-question should be answerable independently and together cover all aspects of the original. Temperature 0.8 and top-p 0.8 encourage diverse decompositions.

2. **Retrieve for each sub-question.** Execute separate retrieval for the original query plus each sub-question. This generates multiple result sets, each focused on one aspect.

3. **Pool all results.** Combine results using simple union—concatenate all result lists and deduplicate by document ID, keeping the first occurrence. No ranking manipulation at this stage.

4. **Rerank against original query.** A cross-encoder (paper uses `bge-reranker-large`) scores each pooled document against the *original* question. This is mandatory—the cross-encoder sees both the full question and each document together, enabling it to judge relevance to the complete multi-faceted query.

5. **Generate answer.** Use the reranked top-k documents as context for LLM generation.


```mermaid
flowchart TB
    subgraph Decompose["1. Decompose Query"]
        Q[/"How does stress affect<br/>memory and decision-making?"/]
        LLM["LLM<br/>(temp=0.8)"]
        Q --> LLM
        LLM --> SQ1["Sub-Q1: How does stress<br/>affect memory?"]
        LLM --> SQ2["Sub-Q2: How does stress<br/>affect decision-making?"]
        LLM --> SQO["Original query"]
    end

    subgraph Retrieve["2. Parallel Retrieval"]
        SQ1 --> R1["Search 1"]
        SQ2 --> R2["Search 2"]
        SQO --> R3["Search 3"]
        R1 --> D1["Docs A, B, C"]
        R2 --> D2["Docs D, E, F"]
        R3 --> D3["Docs A, G, H"]
    end

    subgraph Pool["3. Union + Deduplicate"]
        D1 --> POOL["Pool: A, B, C, D, E, F, G, H"]
        D2 --> POOL
        D3 --> POOL
    end

    subgraph Rerank["4. Cross-Encoder Rerank"]
        POOL --> CE["Cross-Encoder<br/>(vs original query)"]
        CE --> TOP["Top-k: D, A, G, B, E"]
    end

    subgraph Generate["5. Answer Generation"]
        TOP --> ANS["LLM generates answer<br/>using reranked context"]
    end
```


**Why simple union instead of rank fusion?** The paper found that complex merging strategies (like RRF) don't improve results when cross-encoder reranking follows. The reranker is powerful enough to sort through the pooled candidates—sophisticated merging is redundant.

**Why is reranking mandatory?** Without reranking, the pooled results would be ordered by their original scores from different searches. These scores aren't comparable across queries. The cross-encoder provides a unified relevance judgment against the complete original question.

**Benchmark results:** +36.7% MRR@10 on MultiHop-RAG, +11.6% F1 on HotpotQA—substantial improvements on multi-hop question answering benchmarks.


### Key Findings

**Decomposition temperature matters.** The paper uses temperature 0.8 with top-p 0.8 to generate diverse sub-questions. Lower temperatures produce repetitive decompositions that don't cover the full query.

**Simple queries don't benefit.** For straightforward factual questions, decomposition adds latency without improving results. The LLM should recognize these and keep the query as-is.

**Cross-encoder reranking is essential.** The pooled results need unified scoring. Without reranking, decomposition can hurt performance by mixing incomparable scores.

**Sub-question count is bounded.** The paper caps at 5 sub-questions. More leads to irrelevant tangents; fewer may miss aspects of complex queries.



## RAGLab Implementation

RAGLab implements the paper's algorithm faithfully, with one practical addition from community practice:

```python
# src/prompts.py
DECOMPOSITION_PROMPT = """Break down this question for a knowledge base on cognitive science and philosophy.

If the question is simple enough to answer directly, keep it as a single question.
Otherwise, create 3-5 sub-questions that can be answered independently and together cover all aspects of the original.

Question: {query}

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation"
}}"""
```

**Matches paper:**
- Simple union merge (concatenate + deduplicate by chunk_id, not RRF)
- Decomposition temperature 0.8
- Cross-encoder reranking mandatory (`requires_reranking=True` in StrategyConfig)
- Original query included in retrieval alongside sub-questions

**RAGLab additions:**
- "Keep as single question" clause for simple queries (follows [Haystack practice](https://haystack.deepset.ai/blog/query-decomposition))—avoids unnecessary decomposition overhead
- JSON response with `reasoning` field for debugging decomposition decisions
- Domain-specific phrasing ("cognitive science and philosophy") matching our corpus

### Code Locations

| Component | File |
|-----------|------|
| Prompt | `src/prompts.py:DECOMPOSITION_PROMPT` |
| Preprocessing | `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` |
| Strategy | `src/rag_pipeline/retrieval/strategies/decomposition.py` |
| Config | `src/rag_pipeline/retrieval/preprocessing/strategy_config.py` |


## Navigation

**Next:** [GraphRAG](graphrag.md) — Entity-based retrieval with knowledge graph communities

**Related:**
- [HyDE](hyde.md) — Hypothetical document embeddings for vocabulary bridging
- [Preprocessing Overview](README.md) — Strategy comparison
- [Paper (arXiv)](https://arxiv.org/abs/2507.00355) — Original decomposition research
