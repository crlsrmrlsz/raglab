# HyDE: Hypothetical Document Embeddings

[← Preprocessing Overview](README.md) | [Home](../../README.md)

**The problem:** Questions and documents live in different semantic spaces. A query like "What causes stress to affect memory?" uses interrogative, conversational language, while relevant documents use declarative, scientific prose ("Chronic cortisol elevation impairs hippocampal neurogenesis..."). These vectors end up distant in embedding space despite high topical relevance.

**The solution:** [HyDE](https://arxiv.org/abs/2212.10496) generates a *hypothetical answer* first, then searches for real documents similar to that answer. The hypothetical shares declarative structure and domain vocabulary with real documents, bridging the semantic gap.



## Paper Approach

**Paper:** "Precise Zero-Shot Dense Retrieval without Relevance Labels"
**Authors:** Gao, Ma, Lin, Callan (CMU / Waterloo)
**Published:** ACL 2023 ([arXiv:2212.10496](https://arxiv.org/abs/2212.10496))

<div align="center">

| Benchmark | Contriever (baseline) | HyDE | Improvement |
|-----------|----------------------|------|-------------|
| MS MARCO | 20.6 NDCG@10 | 26.6 | +29% |
| TREC-DL19 | 45.2 | 55.4 | +23% |
| NQ | 25.4 | 35.0 | +38% |

</div>

The paper's algorithm:
1. **Instruct** an instruction-following LLM (e.g., InstructGPT) to generate a hypothetical document
2. **Encode** the hypothetical using an unsupervised contrastive encoder (e.g., Contriever)
3. **Search** for real documents similar to the hypothetical embedding

### Original Prompts

From the official implementation ([texttron/hyde](https://github.com/texttron/hyde)):

<div align="center">

| Task | Prompt |
|------|--------|
| **Web Search** | "Please write a passage to answer the question. Question: {}" |
| **SciFact** | "Please write a scientific paper passage to support/refute the claim. Claim: {}" |
| **TREC-COVID** | "Please write a scientific paper passage to answer the question. Question: {}" |
| **FiQA** | "Please write a financial article passage to answer the question. Question: {}" |
| **Arguana** | "Please write a counter argument for the passage. Passage: {}" |

</div>

Key observations:
- All prompts are **minimal** (1-2 sentences)
- They specify **document type** (passage, scientific paper, financial article)
- **No examples** provided
- **No vocabulary lists** or specific terminology

### Key Findings

<div align="center">

| Finding | Explanation |
|---------|-------------|
| **Minimal prompts work best** | Over-specification causes template bias, limiting embedding diversity |
| **Document type matters** | Mention "passage", "paper", "article" but don't overspecify vocabulary |
| **Temperature 0.7** | Provides sufficient creativity for diverse hypotheticals |
| **Dense bottleneck filters hallucinations** | The encoder discards incorrect details, preserving semantic essence |
| **K=5 averaging improves robustness** | Multiple hypotheticals capture different phrasings and perspectives |

</div>

**The dense bottleneck principle:** The embedding model (trained on real documents) compresses text into fixed-dimension vectors. This compression acts as a filter:
- **Preserved**: Topics, concepts, semantic relationships
- **Filtered**: Specific wrong facts, hallucinated details

The hypothetical doesn't need to be *correct*—it needs to be *semantically similar* to relevant documents.

### Technical Parameters

<div align="center">

| Parameter | Paper Value | Rationale |
|-----------|-------------|-----------|
| **Temperature** | 0.7 | Higher creativity for diverse hypotheticals |
| **Number of hypotheticals (K)** | 5 | Multiple hypotheticals averaged in embedding space |
| **Embedding combination** | Element-wise mean | Creates more robust query representation |
| **Max tokens** | ~100-150 | Short passages (2-3 sentences) |

</div>

---

## Implementation in RAGLab

### Algorithm

```
1. Receive user query
2. Generate K hypothetical answer passages (K=2 in RAGLab)
   For i = 1 to K:
      a. Call LLM with HYDE_PROMPT.format(query=query)
      b. Temperature = 0.7
      c. Collect passage_i
3. Embed all K passages using text-embedding-3-large
4. Average embeddings (element-wise mean)
   avg_embedding[j] = (1/K) * Σ passage_embedding[i][j]
5. Search Weaviate:
   - Hybrid: Use averaged vector for semantic + original query for BM25
   - Keyword: Search each passage via BM25, merge with RRF
```

### Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│        User Query: "Why do we procrastinate?"               │
└────────────────────────────┬────────────────────────────────┘
                             ↓
              ┌──────────────────────────────┐
              │  hyde_prompt(query, k=2)     │
              │  Temperature: 0.7            │
              └──────────────┬───────────────┘
                             ↓
         ┌───────────────────┴───────────────────┐
         ↓                                       ↓
┌─────────────────────┐             ┌─────────────────────┐
│ Passage 1:          │             │ Passage 2:          │
│ "Procrastination    │             │ "Temporal           │
│ stems from limbic   │             │ discounting causes  │
│ override of         │             │ us to prefer        │
│ prefrontal control" │             │ immediate rewards"  │
└─────────┬───────────┘             └──────────┬──────────┘
          ↓                                    ↓
   [embed_passage_1]                    [embed_passage_2]
   [0.12, 0.45, ...]                    [0.14, 0.43, ...]
          └─────────────┬──────────────────────┘
                        ↓
              ┌─────────────────────┐
              │ Average Embeddings  │
              │ [0.13, 0.44, ...]   │
              └─────────┬───────────┘
                        ↓
              ┌─────────────────────┐
              │ query_hybrid(       │
              │   vector=avg_embed, │
              │   query=original    │
              │ )                   │
              └─────────┬───────────┘
                        ↓
              ┌─────────────────────┐
              │ Top-K Chunks        │
              └─────────────────────┘
```

### RAGLab Prompt

```python
# src/prompts.py

HYDE_PROMPT = """Please write a short passage drawing on insights from brain science and classical philosophy (Stoicism, Taoism, Confucianism, Schopenhauer, Gracian) to answer the question.

Question: {query}

Passage:"""
```

**Design rationale:**

| Decision | Reasoning |
|----------|-----------|
| **"Drawing on insights from..."** | Requests cross-domain synthesis matching our mixed corpus |
| **Parenthetical tradition hints** | Provides corpus cues without vocabulary lists |
| **Covers all 10 philosophy books** | Stoicism (4), Taoism (1), Confucianism (1), Schopenhauer (3), Gracian (1) |
| **No examples** | Follows paper's finding that over-specification causes template bias |
| **Short output expected** | "short passage" guides length without rigid constraints |

### Key Design Decisions

<div align="center">

| Decision | Paper | RAGLab | Rationale |
|----------|-------|--------|-----------|
| **Temperature** | 0.7 | 0.7 | Paper default for diverse hypotheticals |
| **K (hypotheticals)** | 5 | 2 | Balance cost vs. robustness |
| **Prompt style** | Task-specific | Corpus-specific | Domain hints for neuroscience + philosophy books |
| **Embedding model** | Contriever | text-embedding-3-large | Aligns with our existing embedding pipeline |
| **Search integration** | Dense only | Hybrid | Original query provides BM25 keyword matching |

</div>



## Differences from Paper

| Aspect | Paper | RAGLab | Why |
|--------|-------|--------|-----|
| **K value** | 5 | 2 | Cost efficiency—each hypothetical requires an LLM call |
| **Prompt domain** | Task-specific (SciFact, FiQA) | Corpus-specific (brain science + philosophy) | Matches our 19-book corpus |
| **Search type** | Dense retrieval only | Hybrid (vector + BM25) | Original query provides keyword matching as fallback |
| **Keyword fallback** | Not applicable | RRF merge of K passage searches | Principled combination when vector unavailable |

---

## When HyDE Works Best

**Strong use cases:**
- Vague or contextually ambiguous questions needing context enrichment
- Complex queries requiring semantic understanding beyond keywords
- Questions with different vocabulary than source documents
- Zero-shot retrieval scenarios (no task-specific training)

**Limitations:**
- **Knowledge bottleneck**: HyDE struggles when the LLM is unfamiliar with the topic
- **Simple keyword queries**: Traditional BM25 works just as well with less latency
- **Highly specialized domains**: Factual precision may suffer from hallucinated vocabulary
- **Latency**: Requires K LLM calls before search (~1-2s per query)

---

## Navigation

**Next:** [Query Decomposition](query-decomposition.md) — Breaking complex questions into sub-queries

**Related:**
- [GraphRAG](graphrag.md) — Entity-based retrieval with knowledge graph communities
- [Preprocessing Overview](README.md) — Strategy comparison
- [Paper (arXiv)](https://arxiv.org/abs/2212.10496) — Original HyDE research
- [Official Implementation](https://github.com/texttron/hyde) — CMU reference code
