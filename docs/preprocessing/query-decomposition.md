# Query Decomposition

[← HyDE](hyde.md) | [Home](../../README.md)

> **Paper:** [Question Decomposition for Retrieval-Augmented Generation](https://arxiv.org/abs/2507.00355) | Ammann et al. (Humboldt-Universität) | ACL SRW 2025

Breaks complex multi-part questions into simpler sub-questions, retrieves for each independently, then merges results. Documents appearing in multiple sub-query results get boosted in the final ranking.

**Type:** Query-time preprocessing | **LLM Calls:** 1 per query | **Latency:** ~500ms

---

## Theory

### The Core Problem

Some questions hide multiple questions inside:

```
Query: "How does Stoic philosophy compare to neuroscience on emotional regulation?"

Requires:
  1. What Stoics said about emotions
  2. What neuroscience says about emotional regulation
  3. How they relate
```

A single embedding search struggles because no chunk contains all three aspects—the information is scattered across your corpus.

### Research Background

The paper (ACL SRW 2025) establishes decomposition as a retrieval enhancement for multi-hop questions:

| Benchmark | Improvement |
|-----------|-------------|
| MultiHop-RAG | **+36.7% MRR@10** |
| HotpotQA | **+11.6% F1** |

**The algorithm works in four steps:**

1. **Decompose the query.** The LLM generates up to 5 sub-questions. In experiments, it generated exactly 5 sub-questions 93-99% of the time. The original query is always retained alongside sub-queries.

2. **Retrieve for each.** Execute top-k retrieval for the original query plus each sub-question. This creates a broader candidate pool covering different aspects.

3. **Merge candidates.** All retrieved passages go into one pool. (Note: The paper uses simple union; RAGLab uses RRF—see [Implementation](#raglab-implementation) for differences.)

4. **Rerank and select.** A cross-encoder (bge-reranker-large in the paper) scores each candidate against the *original* query. Top-k passages by reranker score go to generation.

**Why rerank against the original query?** Sub-queries can drift from user intent. A chunk relevant to "What do Stoics say about anger?" might not help with "How does this compare to neuroscience?" Reranking filters for overall relevance.

### Parameters

The paper uses temperature **0.8** with nucleus sampling (**Top-p=0.8**) for diverse decomposition. Higher temperatures generate varied sub-questions that explore different framings.

### Prompts

The paper describes using "a fixed natural language prompt" but doesn't disclose the exact wording. Community implementations provide guidance:

| Source | Key Practice |
|--------|--------------|
| **[Haystack](https://haystack.deepset.ai/blog/query-decomposition)** | "If the query is simple, then keep it as it is" |
| **[LangChain](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/)** | "If there are acronyms or words you are not familiar with, do not try to rephrase them" |
| **[EfficientRAG](https://arxiv.org/abs/2408.04259)** | Sub-questions should be "independently answerable" |

The pattern: instruct the LLM to simplify only when needed, preserve domain terms, and ensure each sub-question stands alone.

---

## RAGLab Implementation

### Decomposition Prompt

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

**Design rationale:**
- "If simple, keep as is" — Haystack-inspired, avoids unnecessary decomposition
- "answered independently" — EfficientRAG-inspired, enables parallel retrieval
- "cover all aspects" — ensures comprehensive retrieval
- JSON with reasoning — aids debugging

### Differences from Paper

| Aspect | Paper | RAGLab | Rationale |
|--------|-------|--------|-----------|
| **Merging** | Union (simple pool) | RRF fusion | Boosts chunks appearing in multiple results |
| **Temperature** | 0.8 | 0.7 | Balance diversity with coherence |
| **Max sub-questions** | 5 | 3-5 | "Keep as single" clause for simple queries |
| **Reranking** | Always (bge-reranker-large) | Optional | User-configurable for latency tradeoff |

**RRF vs Union:** The paper uses simple union—all results go into one pool, then rerank filters them. RAGLab uses [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) (k=60), which scores documents by `sum(1/(k+rank))` across result lists. Documents appearing in multiple sub-query results get higher scores before reranking.

### Algorithm

```
1. LLM receives query + decomposition prompt
2. Returns JSON: {sub_questions: [...], reasoning: "..."}
3. Execute search for: original query + each sub-question
4. RRF merge: score = sum(1/(60 + rank)) per document across all result lists
5. Optional reranking against original query
6. Return top-k merged results
```

### Code Locations

| Component | Location |
|-----------|----------|
| Prompt | `src/prompts.py:DECOMPOSITION_PROMPT` |
| Decompose function | `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py:decompose_query()` |
| Strategy wrapper | `src/rag_pipeline/retrieval/preprocessing/strategies.py:decomposition_strategy()` |
| RRF merge | `src/rag_pipeline/retrieval/rrf.py:reciprocal_rank_fusion()` |
| Retrieval strategy | `src/rag_pipeline/retrieval/strategies/decomposition.py` |

---

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Multi-step procedural questions | "What is X, then how does it work, then why?" |
| Comparison within single domain | "Compare X and Y in neuroscience" |
| Multi-aspect factual queries | "List features A, B, and C of X" |
| **Avoid when** | Cross-domain synthesis needed (use HyDE or GraphRAG) |

**Cross-domain anti-pattern:** For "How do neuroscience and philosophy together explain addiction?", decomposition fragments the integration—sub-queries retrieve each domain separately, missing chunks that bridge both. HyDE pre-synthesizes the bridge; GraphRAG uses knowledge graph connections.

---

## Navigation

**Next:** [GraphRAG](graphrag.md) — Knowledge graph + communities for cross-document reasoning

**Related:**
- [HyDE](hyde.md) — Better for cross-domain synthesis (pre-generates bridging content)
- [Preprocessing Overview](README.md) — Strategy comparison
- [Paper (arXiv)](https://arxiv.org/abs/2507.00355) — Original research
- [Haystack Blog](https://haystack.deepset.ai/blog/query-decomposition) — Community implementation guide
