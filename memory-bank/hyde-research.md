# HyDE (Hypothetical Document Embeddings) Research

**Date:** December 24, 2024
**Purpose:** Research findings for improving the HyDE prompt in RAGLab

---

## 1. Foundational Paper

**Title:** "Precise Zero-Shot Dense Retrieval without Relevance Labels"
**Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
**ArXiv:** [2212.10496](https://arxiv.org/abs/2212.10496)
**Published:** ACL 2023 (Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics)

### Core Concept

HyDE transforms queries into hypothetical documents that capture relevance patterns. The key insight is that the contrastive encoder's "dense bottleneck" filters out incorrect details from the generated hypothetical document, while preserving semantic relevance.

**How it works:**
1. Given a query, prompt an LLM to generate a hypothetical document that would answer it
2. Encode this hypothetical document into an embedding vector
3. Use this embedding to find similar real documents in the vector store
4. The encoder naturally filters hallucinated content through the embedding space

### Key Findings from the Paper

1. **Minimal prompts work best** - Under-specification hurts, but over-specification causes template bias
2. **Task-specific document type** - Mention document type (passage, paper, article) without overspecifying vocabulary
3. **Temperature 0.7** - Provides sufficient creativity for diverse hypotheticals
4. **Dense bottleneck filters hallucinations** - The encoder filters incorrect details through embedding space
5. **Embeddings do the heavy lifting** - Trust the contrastive encoder, not the prompt

---

## 2. Original Prompt Templates

From the official implementation ([texttron/hyde](https://github.com/texttron/hyde) - `src/hyde/promptor.py`):

| Task | Prompt Template |
|------|-----------------|
| **Web Search** | `"Please write a passage to answer the question. Question: {}"` |
| **SciFact** | `"Please write a scientific paper passage to support/refute the claim. Claim: {}"` |
| **TREC-COVID** | `"Please write a scientific paper passage to answer the question. Question: {}"` |
| **FiQA** | `"Please write a financial article passage to answer the question. Question: {}"` |
| **DBpedia Entity** | `"Please write a passage to answer the question. Question: {}"` |
| **TREC-News** | `"Please write a news passage about the topic. Topic: {}"` |
| **Arguana** | `"Please write a counter argument for the passage. Passage: {}"` |
| **Mr-TyDi** | `"Please write a passage in {} to answer the question in detail. Question: {}"` |

**Key observations:**
- All prompts are extremely minimal (1-2 sentences)
- They specify document type (passage, scientific paper, financial article, news)
- No examples provided
- No vocabulary lists or specific terminology
- Temperature 0.7 used for generation

---

## 3. Technical Parameters

From the paper and implementations:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Temperature** | 0.7 | Higher creativity for diverse hypotheticals |
| **Number of hypotheticals (K)** | 5 (default) | Multiple hypotheticals averaged in embedding space |
| **Embedding averaging** | Mean of K embeddings | Creates more robust query representation |
| **Max tokens** | 100-150 | Short passages (2-3 sentences) |

**Note:** Single hypothetical works well for most applications; K=5 averaging is optional optimization.

---

## 4. Community Best Practices

### LlamaIndex Default
```
"Please write a passage to answer the question. Try to include as many key details as possible."
```
- Source: [LlamaIndex HyDE Documentation](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/hyde/)

### Haystack Implementation
```
"Given a question, generate a paragraph of text that answers the question. Question: {{question}} Paragraph:"
```
- Source: [Haystack HyDE Documentation](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- Uses: 5 hypothetical documents per query, temperature 0.75, max 400 tokens

### Domain Adaptation Guidance
From [Zilliz HyDE Guide](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings):
- For technical domains: Specify document type ("write a maintenance manual") but avoid specific terms
- System message: "Write a document that answers the question:"
- 100-token maximum for conciseness

### Size Alignment
From [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb):
- Hypothetical document size should match chunk size used during indexing
- Creates better alignment in vector space

---

## 5. When HyDE Works Best

**Good use cases:**
- Zero-shot retrieval (no task-specific training data)
- Vague or contextually ambiguous questions needing context enrichment
- Multi-language retrieval across diverse domains
- Complex queries requiring semantic understanding beyond keywords
- Out-of-domain technical scenarios where embedding models lack training data

**Limitations:**
- **Knowledge bottleneck** - HyDE struggles when topics are unfamiliar to the LLM
- **Multilingual challenges** - Performance degrades with under-resourced languages
- **Highly specialized domains** - Factual precision requirements may suffer
- **Simple keyword queries** - Traditional retrieval works just as well with less latency

---

## 6. Overfitting Analysis: Current Prompt Problems

The original RAGLab `HYDE_PROMPT` had these issues:

| Issue | Impact |
|-------|--------|
| 6 detailed examples with specific names | Creates template bias - LLM mimics structure |
| Explicit terms: "Seneca", "Epictetus", "Schopenhauer", "Kahneman" | Limits vocabulary to these exact names |
| Rigid 50/50 neuroscience/philosophy split | Over-constrains natural answer generation |
| Specific concepts: "temporal discounting", "HPA axis", "hedonic adaptation" | Vocabulary tunnel vision |
| ~800 tokens prompt | Expensive, introduces bias from examples |

**The fundamental problem:** Over-specified prompts produce templated outputs that limit embedding diversity, reducing retrieval recall.

---

## 7. Recommended Prompt Design Principles

Based on research synthesis:

### Principle 1: Minimal Instructions
The paper's prompts are 1-2 sentences. Examples create template bias.

### Principle 2: Broad Domain Hints
Mention document type/domain without vocabulary lists:
- Good: "cognitive science and philosophical wisdom traditions"
- Bad: "System 1/System 2, Seneca, Schopenhauer, dopamine, cortisol..."

### Principle 3: Trust the Encoder
The embedding space naturally filters hallucinations. Don't over-engineer the prompt.

### Principle 4: No Explicit Balance Requirements
Let queries determine natural topic balance. Forced splits constrain generation.

### Principle 5: Size Hints Are Optional
"2-3 sentences" or chunk size hints can help but aren't critical.

---

## 8. Implementation: Paper-Aligned Prompt with K=5 Averaging (Updated Dec 2024)

### Current HyDE Prompt
```python
HYDE_PROMPT = """Please write a short passage drawing on insights from brain science and classical philosophy (Stoicism, Taoism, Confucianism, Schopenhauer, Gracian) to answer the question.

Question: {query}

Passage:"""
```

### December 2024 Update: Dual-Domain + K=5 Implementation

Following the paper's recommendation for robust retrieval, we now generate **5 hypothetical passages** and average their embeddings.

**Prompt Design Rationale:**
1. **"Drawing on insights from..."** - Explicitly requests cross-domain synthesis for our mixed corpus
2. **Parenthetical tradition hints** - "(Stoicism, Taoism, Confucianism, Schopenhauer, Gracian)" provides specific corpus cues without vocabulary lists
3. **Covers all 10 philosophy books:**
   - Stoicism: Seneca, Epictetus (2 books), Marcus Aurelius
   - Taoism: Lao Tzu (Tao te ching)
   - Confucianism: Confucius (The Analects)
   - Schopenhauer: 3 books (Essays and Aphorisms, Counsels and Maxims, Wisdom of Life)
   - Gracian: The Pocket Oracle and Art of Prudence

**K=5 Multi-Hypothetical Implementation:**
```python
def hyde_prompt(query: str, model: str, k: int = 5) -> List[str]:
    """Generate k hypothetical documents for query."""
    passages = []
    for _ in range(k):
        response = call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,  # Paper default for diversity
        )
        passages.append(response.strip())
    return passages
```

At retrieval time:
1. Embed all K passages using the same embedding model
2. Average the embedding vectors (element-wise mean)
3. Use averaged vector for hybrid search
4. Original query still used for BM25 keyword matching

**Why K=5 Averaging?**
- Paper finding: Multiple hypotheticals improve retrieval robustness
- Different passages capture different phrasings and perspectives
- Averaging creates a more centered representation in embedding space
- Reduces impact of any single hallucination

### Previous Design Rationale

This prompt follows the paper's approach for **domain-specific retrieval**:

| Paper Template | Our Adaptation |
|----------------|----------------|
| SciFact: "Please write a **scientific paper passage**..." | "Please write a short passage drawing on insights from **brain science and classical philosophy**..." |
| FiQA: "Please write a **financial article passage**..." | Same pattern: domain cue in task description |

**Key decisions aligned with paper:**

1. **Temperature 0.7**: Matches paper for diverse hypothetical generation.

2. **Minimal instructions**: Single sentence prompt to avoid template bias.

3. **Domain + tradition hints**: Specific enough to match corpus vocabulary, general enough to avoid overfitting.

4. **K=5 averaging**: Paper's recommended approach for robust retrieval.

### Dense Bottleneck Principle

The encoder (trained on real documents) compresses passages into fixed-size vectors:
- Captures semantic essence (topics, concepts, relationships)
- Discards noise (specific wrong facts, hallucinations)
- This is why length constraints are unnecessary - trust the encoder
- K=5 averaging further smooths out individual passage noise

---

## 9. Sources

### Primary Research
- [HyDE Paper (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496)
- [ACL 2023 Publication](https://aclanthology.org/2023.acl-long.99/)
- [Official Implementation (texttron/hyde)](https://github.com/texttron/hyde)
- [CMU HyDE Paper PDF](https://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf)

### Implementation Guides
- [Haystack HyDE Documentation](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [LangChain HyDE](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/hyde/)
- [Zilliz HyDE Guide](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
- [RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb)

### Additional Resources
- [Pondhouse Data HyDE Blog](https://www.pondhouse-data.com/blog/advanced-rag-hypothetical-document-embeddings)
- [AI Planet HyDE Tutorial](https://medium.aiplanet.com/advanced-rag-improving-retrieval-using-hypothetical-document-embeddings-hyde-1421a8ec075a)
