# Evaluation Results

[← Evaluation Framework](README.md) | [Home](../../README.md)

Two evaluation runs tested **46 configurations** across 4 factors (chunking, alpha, query strategy, reranking) using RAGAS metrics. Run 1 used 16 curated questions; Run 2 used all 47 questions. All numbers below come from Run 2 (the larger, more reliable run) unless noted. The two key metrics are **answer correctness** (AC — how correct the final answer is) and **context recall** (CR — how much relevant evidence was retrieved).


## Performance by Question Type

The single most important finding: results depend heavily on whether a question targets one topic or spans multiple documents. The system retrieves nearly all relevant evidence for focused questions (CR 0.93) but misses half for cross-domain ones (CR 0.48) — a **45-point CR gap**.

### Single-Concept Questions (19 in Run 2)

Questions targeting one topic from one source document.

<div align="center">

| Metric | Best | Typical | Worst (excl. GraphRAG) |
|--------|------|---------|------------------------|
| Answer Correctness | 0.64 | ~0.54 | ~0.49 |
| Context Recall | 0.93 | ~0.79 | ~0.60 |

</div>

Retrieval works well here — the moderate AC (~0.54) despite high CR means the generation step, not retrieval, is the bottleneck.

### Cross-Domain Questions (28 in Run 2)

Questions requiring synthesis across multiple topics or documents.

<div align="center">

| Metric | Best | Typical | Worst (excl. GraphRAG) |
|--------|------|---------|------------------------|
| Answer Correctness | 0.47 | ~0.45 | ~0.41 |
| Context Recall | 0.48 | ~0.42 | ~0.35 |

</div>

CR of 0.48 means the system misses half the relevant evidence. No configuration tested solves this well.

### Best Configurations by Question Type

**Single-concept** — optimizing for focused, single-topic questions:

<div align="center">

| Rank | Configuration | AC | CR |
|------|--------------|-----|-----|
| 1 | Section \| alpha=0.5 \| None \| rerank | **0.637** | **0.926** |
| 2 | Semantic(std3) \| alpha=0.5 \| None \| no-rerank | 0.636 | 0.917 |
| 3 | Contextual \| alpha=0.5 \| None \| no-rerank | 0.584 | 0.852 |

</div>

Pattern: alpha=0.5 + None strategy in every slot. Section chunking leads with near-perfect retrieval (0.93 CR).

**Cross-domain** — optimizing for multi-document synthesis questions:

<div align="center">

| Rank | Configuration | AC | CR |
|------|--------------|-----|-----|
| 1 | Contextual \| alpha=0.0 \| None \| rerank | **0.475** | 0.424 |
| 2 | Section \| alpha=1.0 \| HyDE \| rerank | 0.472 | 0.449 |
| 3 | Contextual \| alpha=0.5 \| None \| rerank | 0.470 | 0.437 |

</div>

Pattern: Contextual chunking and reranking dominate. The leaderboard looks completely different from single-concept — different chunking, different alpha, different reranking preference.

**Overall best** (balancing both question types):

<div align="center">

| Goal | Configuration | Score |
|------|--------------|-------|
| Best AC | Semantic(std3) \| alpha=0.5 \| None \| no-rerank | 0.527 |
| Best CR | Section \| alpha=0.5 \| None \| rerank | 0.643 |
| Safest all-rounder | Section \| alpha=0.5 \| None \| no-rerank | AC 0.514, CR 0.613 |

</div>

The "safest all-rounder" ranks top-tier in both runs, uses no LLM calls at query time, and needs no reranker. The best CR config holds **rank 1 for CR in both runs** — the single most stable individual result.

> **Stability caveat**: individual config rankings are unreliable (Spearman rho = 0.27 for AC). These "best" configs are within 0.02 of several neighbors. Trust the factor-level patterns (Section + hybrid + None) more than the specific winning config.


## Factor Analysis

Each factor was analyzed by splitting results across question types, revealing where each decision actually matters.

### Search Alpha

Alpha controls keyword (BM25) vs semantic (vector) search balance. This is the highest-confidence finding — hybrid search won every cell in both runs.

<div align="center">

| Alpha | Single AC | Cross AC | Single CR | Cross CR |
|-------|-----------|----------|-----------|----------|
| **0.5 (Hybrid)** | **0.57** | **0.45** | **0.85** | **0.45** |
| 0.0 (BM25) | 0.55 | 0.44 | 0.81 | 0.40 |
| 1.0 (Semantic) | 0.53 | 0.45 | 0.72 | 0.42 |

</div>

The biggest advantage is on single-concept CR: +0.13 over pure semantic. Zero cost — it's a query-time parameter.

### Chunking Method

How source documents are split into retrievable units. Chunking is the most impactful indexing decision (highest eta-squared among non-GraphRAG factors in both runs).

<div align="center">

| Chunking | Single AC | Cross AC | Single CR | Cross CR |
|----------|-----------|----------|-----------|----------|
| Section | **0.56** | 0.45 | **0.85** | 0.43 |
| Contextual | 0.54 | **0.46** | 0.76 | 0.43 |
| Semantic(std3) | 0.56 | 0.44 | 0.79 | 0.43 |
| RAPTOR | 0.53 | 0.45 | 0.77 | 0.42 |
| Semantic(std2) | 0.52 | 0.44 | 0.68 | 0.38 |

</div>

The key split:
- **Single-concept**: Section wins decisively, especially on CR (0.85 vs next-best 0.79). Heading-based boundaries preserve document structure.
- **Cross-domain**: Contextual wins AC (0.46 vs 0.45). LLM-generated context summaries help chunks be discovered when queries use different vocabulary.

Semantic(std2) is consistently worst — its tight similarity threshold creates fragments too small to provide sufficient context.

### Query Strategy

How the user's question is transformed before retrieval. High confidence that it doesn't matter.

<div align="center">

| Strategy | Single AC | Cross AC | Single CR | Cross CR |
|----------|-----------|----------|-----------|----------|
| None | **0.55** | 0.45 | **0.80** | 0.42 |
| HyDE | 0.54 | **0.45** | 0.79 | **0.44** |
| Decomposition | 0.54 | 0.45 | 0.71 | 0.43 |
| GraphRAG | 0.37 | 0.38 | 0.00 | 0.00 |

</div>

Excluding GraphRAG, AC differences are <0.01. HyDE showed a 3-point advantage in Run 1 that vanished completely in Run 2. Decomposition hurts single-concept CR badly (0.71 vs 0.80) — sub-question decomposition fragments retrieval without improving complex questions.

### Reranking

Cross-encoder model re-scores initially retrieved chunks. The effect is negligible and direction-dependent.

<div align="center">

| Reranking | Single AC | Cross AC | Single CR | Cross CR |
|-----------|-----------|----------|-----------|----------|
| On | 0.54 | **0.45** | 0.75 | **0.43** |
| Off | 0.54 | 0.44 | **0.80** | 0.40 |

</div>

The directions flip by question type:
- **Single-concept CR**: reranking **hurts** (0.75 vs 0.80). The cross-encoder demotes relevant chunks that use different vocabulary.
- **Cross-domain CR**: reranking **helps** (0.43 vs 0.40). It picks the best chunks from mixed-relevance results.


## GraphRAG

The single GraphRAG configuration (`Semantic(std2) | alpha=1.0 | GraphRAG | no-rerank`) was validated with 64,483 entities in Weaviate and valid API keys. This is not an infrastructure artifact.

<div align="center">

| Metric | Single | Cross |
|--------|--------|-------|
| Answer Correctness | 0.37 | 0.38 |
| Context Recall | 0.00 | 0.00 |

</div>

- **Ranks 46/46 (dead last) in both runs** with nearly identical scores (0.376 Run 1, 0.378 Run 2)
- CR = 0.00 always — entity summaries don't map to ground-truth passages (architectural mismatch with RAGAS measurement, not a retrieval failure per se)
- ~20% below baseline. Entity-based retrieval adds noise for factual Q&A on this corpus


## Reliability

Cross-run stability was measured using Spearman rank correlation between Run 1 (16 questions) and Run 2 (47 questions).

<div align="center">

| Metric | Spearman rho (Run 1 vs Run 2) | Interpretation |
|--------|-------------------------------|----------------|
| Answer Correctness | 0.27 | Weak — config rankings are **unstable** |
| Context Recall | 0.68 | Moderate-strong — retrieval rankings are **stable** |

</div>

Most configs are within 0.02–0.05 AC of each other. Small absolute differences produce large rank swings — the Run 1 #2 config fell to #34 in Run 2, while the Run 1 #44 config rose to #14.

**What's trustworthy**: Factor-level conclusions (hybrid wins, Section leads CR, strategies are equivalent). These hold in both runs.

**What's not trustworthy**: Individual configuration rankings. The "best config" changed entirely between runs. Do not pick a config based on its specific rank.


## Key Takeaways

**1. Hybrid search (alpha=0.5) is the single best free optimization.**
Wins every metric, every question type, both runs. Zero cost — it's a query-time parameter.

**2. Chunking choice depends on your question type.**
Section for single-concept (preserves document structure); Contextual for cross-domain (LLM context helps chunks be found across topics). This is the most impactful indexing decision.

**3. Cross-domain retrieval is the real bottleneck.**
45-point CR gap (0.93 vs 0.48). No configuration solves it. Future work should target multi-document retrieval architecture, not query tricks.

**4. "Advanced" query features added no reliable value.**
HyDE, Decomposition, and reranking all produced <0.02 AC differences that shifted between runs. Save the LLM calls and complexity.


---

## Navigation

**Related:**
- [Evaluation Framework](README.md) — RAGAS metrics and test dataset
- [Cross-Run Comparison](../../data/evaluation/analysis/cross_run_comparison.md) — Detailed stability analysis
- [Run 2 Full Analysis](../../data/evaluation/analysis/run2_full_analysis.md) — Primary data source
- [Run 1 Analysis](../../data/evaluation/analysis/run1_selected_analysis.md) — 16-question subset
- [Getting Started](../getting-started.md) — Start over
