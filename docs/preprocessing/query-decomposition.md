# Query Decomposition

[← HyDE](hyde.md) | [Home](../../README.md)

Some questions hide multiple questions inside. "How does Stoic philosophy compare to neuroscience on emotional regulation?" requires three things: what Stoics said about emotions, what neuroscience says, and how they relate. A single embedding search struggles because no chunk contains all three aspects—the information is scattered across your corpus.

Query decomposition solves this by breaking the complex question into sub-questions, retrieving for each independently, then merging the results. Documents that appear across multiple sub-queries get boosted in the final ranking.

**When decomposition helps:** Multi-hop questions requiring multiple evidence pieces, procedural "what, then how, then why" queries, and comparison questions within a single domain.

**When it struggles:** Cross-domain synthesis where you need documents that *bridge* topics (decomposition fragments the integration), simple factual lookups (unnecessary overhead), and time-sensitive applications (~500ms latency per query).



## The Paper and Community Implementations

**Paper:** "Question Decomposition for Retrieval-Augmented Generation"
**Authors:** Ammann, Golde, Akbik (Humboldt-Universität zu Berlin)
**Published:** ACL SRW 2025 ([arXiv:2507.00355](https://arxiv.org/abs/2507.00355))

The authors wanted better retrieval for multi-hop questions without specialized indexing or training. Their solution: let an LLM break down complex queries, retrieve for each piece, then merge and rerank.

The algorithm works in four steps:

1. **Decompose the query.** The LLM generates up to 5 sub-questions from the original query. In experiments, it generated exactly 5 sub-questions 93-99% of the time. The original query is always retained alongside sub-queries.

2. **Retrieve for each.** Execute top-k retrieval for the original query plus each sub-question. This creates a broader candidate pool covering different aspects.

3. **Merge candidates.** The paper uses simple union—all retrieved passages go into one pool. Documents appearing in multiple sub-query results naturally get more chances at selection.

4. **Rerank and select.** A cross-encoder (bge-reranker-large) scores each candidate against the *original* query. Top-k passages by reranker score go to generation.

**Why does reranking against the original query matter?** Sub-queries can drift from the user's actual intent. A chunk relevant to "What do Stoics say about anger?" might not help with "How does this compare to neuroscience?" Reranking filters for overall relevance.

**Benchmark results:** +36.7% MRR@10 on MultiHop-RAG, +11.6% F1 on HotpotQA. The improvement comes from capturing evidence that no single query would find.

### Parameters

The paper used temperature 0.8 with nucleus sampling (Top-p=0.8) for diverse decomposition. Higher temperatures generate varied sub-questions that explore different framings; lower temperatures produce repetitive outputs.

### Prompts

The paper describes using "a fixed natural language prompt provided to an instruction-tuned language model" but doesn't disclose the exact wording. Community implementations provide guidance:

| Source | Key Practice |
|--------|--------------|
| **[Haystack](https://haystack.deepset.ai/blog/query-decomposition)** | "If the query is simple, keep it as it is" — avoids unnecessary decomposition |
| **[LangChain](https://python.langchain.com/docs/tutorials/query_analysis/)** | "Do not try to rephrase" acronyms or unfamiliar words — preserve terminology |
| **[EfficientRAG](https://arxiv.org/abs/2408.04259)** | Sub-questions should be "independently answerable" — enables parallel retrieval |

The pattern: instruct the LLM to simplify only when needed, keep domain terms intact, and ensure each sub-question stands alone.



## RAGLab Implementation

RAGLab adapts the paper's approach with RRF merging instead of simple union, and includes community best practices in the prompt. Configuration in `src/prompts.py`:

```python
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

**Differences from Paper:**

| Aspect | Paper | RAGLab | Rationale |
|--------|-------|--------|-----------|
| **Merging** | Union (simple pool) | RRF fusion | Boosts chunks appearing in multiple results |
| **Temperature** | 0.8 | 0.7 | Balance diversity with coherence |
| **Max sub-questions** | 5 | 3-5 | "Keep as single" clause for simple queries |
| **Reranking** | Always (bge-reranker-large) | Optional | User-configurable for latency tradeoff |

**RRF vs Union:** The paper uses simple union—all results go into one pool, then rerank filters them. RAGLab uses [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) (k=60), which scores documents by `sum(1/(k+rank))` across result lists. Documents appearing in multiple sub-query results get higher scores. This provides ranking *before* reranking, useful when reranking is disabled.

**The "keep as single" clause:** Following Haystack's practice, the prompt allows the LLM to skip decomposition for simple queries. The paper's LLM nearly always generated 5 sub-questions regardless of query complexity—RAGLab's prompt explicitly permits fewer.

**JSON with reasoning:** The response includes a "reasoning" field that explains the decomposition. This aids debugging and lets you verify the LLM understood the query correctly.

### Algorithm Flow

```
1. LLM receives query + decomposition prompt
2. Returns JSON: {sub_questions: [...], reasoning: "..."}
3. Execute search for: original query + each sub-question
4. RRF merge: score = sum(1/(60 + rank)) per document across all result lists
5. Optional reranking against original query
6. Return top-k merged results
```

### Code Locations

- **Prompt:** `src/prompts.py:DECOMPOSITION_PROMPT`
- **Decompose function:** `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py:decompose_query()`
- **Strategy wrapper:** `src/rag_pipeline/retrieval/preprocessing/strategies.py:decomposition_strategy()`
- **RRF merge:** `src/rag_pipeline/retrieval/rrf.py:reciprocal_rank_fusion()`
- **Retrieval strategy:** `src/rag_pipeline/retrieval/strategies/decomposition.py`


## Navigation

**Next:** [GraphRAG](graphrag.md) — Knowledge graph + communities for cross-document reasoning

**Related:**
- [HyDE](hyde.md) — Better for cross-domain synthesis (pre-generates bridging content)
- [Preprocessing Overview](README.md) — Strategy comparison
- [Paper (arXiv)](https://arxiv.org/abs/2507.00355) — Original research
- [Haystack Blog](https://haystack.deepset.ai/blog/query-decomposition) — Community implementation guide
