# Chunking Strategies

[← Content Preparation](../content-preparation/README.md) | [Home](../../README.md)

This project implements several chunking strategies to compare their performance and measure how each affects overall retrieval quality.

I started with basic section-based and semantic chunking, as commonly recommended, but poor results drove me to explore more advanced techniques. After researching the literature, I decided to implement RAPTOR and Contextual chunking.

The specific corpus content and the types of questions the system must answer shape key design decisions. Chunking is particularly important because it is the first stage that directly influences everything downstream.

Neuroscience and philosophy texts contain dense conceptual content and intricate knowledge relationships. When a philosopher builds an argument about consciousness over 40 pages, or a neuroscientist traces the evolutionary origins of addiction through multiple chapters, naive chunking destroys the conceptual unity that makes these texts meaningful. When relating ideas across fields, preserving complete concepts becomes even more critical.

The goal is to create chunks large enough to contain complete ideas useful for generating answers, while still allowing advanced techniques—both at chunking and query time—to connect related concepts across the corpus.


## Navigation

This document explains common and general considerations about chunking, but the details of each strategy are here:

- **[Section Chunking](section-chunking.md)** — The baseline: fixed-size with sentence overlap
- **[Semantic Chunking](semantic-chunking.md)** — Embedding-based topic boundaries
- **[Contextual Chunking](contextual-chunking.md)** — LLM-generated context prepended
- **[RAPTOR](raptor.md)** — Hierarchical summarization tree



## Why Custom Implementation

This project implements custom chunking rather than using ready-to-use frameworks like LangChain's `RecursiveCharacterTextSplitter` or LlamaIndex's `SentenceSplitter`. While these tools are convenient, building from scratch provides deeper understanding of chunking internals and enables fine-grained control—such as exact token counting, configurable sentence overlap, and tunable similarity thresholds for semantic boundaries.

## Shared Infrastructure

All chunking strategies share common components:

<div align="center">

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Token counting** | `tiktoken` with `text-embedding-3-large` | Exact token counts matching embedding model |
| **Embedding model** | `text-embedding-3-large` (3072 dims) | State-of-the-art dense retrieval |
| **Vector storage** | Weaviate HNSW index + BM25 | Hybrid search (dense + keyword) |
| **Chunk metadata** | `book_id`, `section`, `context` | Hierarchical path for filtering and display |

</div>

We count tokens rather than characters because embedding models operate on tokens, and the character-to-token ratio varies with content (technical terms, punctuation, and rare words tokenize differently than common prose). Using `tiktoken` with the same encoding as `text-embedding-3-large` ensures our max-token target accurately reflects what the model will see.


### Chunk Schema

Chunks are stored in JSON files for inspection and inter-phase isolation, and every chunk includes standardized metadata:

```json
{
  "chunk_id": "BookName::chunk_42",
  "book_id": "BookName",
  "context": "BookName > Chapter 3 > Section 2",
  "section": "Section 2",
  "text": "The actual chunk content...",
  "token_count": 750,
  "chunking_strategy": "sequential_overlap_2"
}
```

These fields serve three main purposes: **attribution** (display sources in UI and answers), **filtering** (scope searches to specific books), and **advanced chunking** (RAPTOR and Contextual use them in LLM prompts to build hierarchical context). Preprocessing strategies operate on query text only and do not use chunk metadata.





## Running Chunking

Once NLP chunks are ready, you can run: 

```bash
# Section (baseline) - No dependencies
python -m src.stages.run_stage_4_chunking --strategy section

# Semantic - No dependencies, specify std coefficient
python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0

# Contextual - Requires semantic chunks (std=2) first
python -m src.stages.run_stage_4_chunking --strategy contextual

# RAPTOR - Requires semantic_std2 chunks first (Stage 4b)
python -m src.stages.run_stage_4b_raptor
```

This stage reads JSON files (one per book) from `data/processed/04_nlp_chunks/` and stores chunks in JSON in `data/processed/05_final_chunks/`, one folder per strategy. 



<div align="center">

| Strategy | Output Directory |
|----------|------------------|
| Section | `data/processed/05_final_chunks/section/` |
| Semantic (std=2) | `data/processed/05_final_chunks/semantic_std2/` |
| Semantic (std=3) | `data/processed/05_final_chunks/semantic_std3/` |
| Contextual | `data/processed/05_final_chunks/contextual/` |
| RAPTOR | `data/processed/05_final_chunks/raptor/` |

</div>




## Evaluation Results

Two evaluation runs compared all chunking strategies across 46 pipeline configurations, using [RAGAS](https://docs.ragas.io/) metrics (answer_correctness, context_recall). Run 1 used 16 curated questions; Run 2 used 46 diverse questions including cross-domain questions that require connecting concepts across different books.

### Chunking Strategy Ranking

<div align="center">

| Chunking | Run 1 (answer_correctness) | Run 2 (answer_correctness) |
|----------|---------------------------|---------------------------|
| **Section** | 4th (0.481) | **1st (0.496)** |
| **Contextual** | 2nd (0.491) | 2nd (0.490) |
| **Semantic (std3)** | 3rd (0.484) | 3rd (0.482) |
| **RAPTOR** | 1st (0.502) | 4th (0.483) |
| **Semantic (std2)** | 5th (0.472) | 5th (0.471) |

</div>

For context_recall (retrieval quality), Section chunking ranked 1st in both runs. Contextual chunking showed the strongest cross-domain performance in both runs — the LLM-generated context descriptions help retrieve chunks for questions that span multiple topics.

The best individual configuration across both runs was `Section | alpha=0.5 | None | no-rerank`, which ranked in the top tier for both answer_correctness and context_recall without requiring any additional LLM calls at query time.

**Limitations**: These results are specific to a small neuroscience/philosophy corpus (5 books). Individual configuration rankings were unstable between runs (Spearman rho = 0.27 for answer_correctness), though factor-level trends (which chunking type is best/worst) were more consistent. See [cross-run comparison](../../data/evaluation/analysis/cross_run_comparison.md) for full details.


---

## Navigation

**Next:** [Section Chunking](section-chunking.md) — The baseline: fixed-size with sentence overlap

**Related:**
- [Preprocessing Strategies](../preprocessing/README.md) — Query-time transformations
- [Evaluation Framework](../evaluation/README.md) — How strategies are compared
