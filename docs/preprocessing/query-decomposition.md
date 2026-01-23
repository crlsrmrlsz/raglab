# Query Decomposition

[← HyDE](hyde.md) | [Home](../../README.md)

> **Paper:** [Question Decomposition for RAG](https://arxiv.org/abs/2507.00355) | Ammann et al. | ACL SRW 2025

Breaks complex questions into sub-questions, retrieves for each, pools results, reranks against original query.

**+36.7% MRR@10** on MultiHop-RAG, **+11.6% F1** on HotpotQA.

## Algorithm (per paper)

```
1. Decompose query → up to 5 sub-questions (temp=0.8, top_p=0.8)
2. Retrieve for: original + each sub-question
3. Pool all results (simple union, deduplicate by chunk_id)
4. Rerank entire pool against original query (mandatory)
5. Generate answer (temp=0.8, top_p=0.8, max_tokens=512)
```

## RAGLab Implementation

**Matches paper exactly:**
- Union merge (not RRF)
- Temperature 0.8 with top_p 0.8
- Reranking mandatory (`requires_reranking=True` in StrategyConfig)
- Generation: temp=0.8, top_p=0.8, max=512 tokens

**RAGLab additions:**
- "Keep as single" clause for simple queries (Haystack practice)
- JSON response with reasoning field (debugging)

### Code Locations

| Component | File |
|-----------|------|
| Prompt | `src/prompts.py:DECOMPOSITION_PROMPT` |
| Decomposition | `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` |
| Strategy | `src/rag_pipeline/retrieval/strategies/decomposition.py` |
| Config | `src/rag_pipeline/retrieval/preprocessing/strategy_config.py` |

## When to Use

| Use | Avoid |
|-----|-------|
| Multi-hop questions | Cross-domain synthesis (use HyDE) |
| Comparison queries | Simple factual lookups |
| "What, how, why" chains | Latency-critical (<500ms) |

## Navigation

[GraphRAG →](graphrag.md) | [HyDE](hyde.md) | [Overview](README.md)
