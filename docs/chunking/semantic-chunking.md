# Semantic Chunking

[← Chunking Overview](README.md) | [Home](../../README.md)

Semantic chunking splits text at topic boundaries detected by embedding similarity, creating chunks that preserve conceptual coherence rather than splitting mid-argument. Unlike fixed-size chunking, it adapts to content structure.

## Semantic Chunking Approaches

[Qu et al. 2024](https://arxiv.org/abs/2410.13070) identifies two main approaches:

- **Breakpoint-based** — "Scans over the sequence of sentences and decides where to insert a breakpoint" when "semantic distance between two consecutive sentences exceeds a threshold, meaning a significant topic change." Preserves document order but is "locally greedy" since it examines only two adjacent sentences at each decision point.
- **Clustering-based** — "Leverages clustering algorithms to group sentences together semantically, capturing global relationships and allowing for non-sequential sentence groupings." However, "risks losing contextual information hidden in the proximity of sentences."

[LangChain](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html) and [LlamaIndex](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/) both implement **breakpoint-based** chunking as their primary semantic chunking approach.

---

## Breakpoint Detection: Why Standard Deviation?

After evaluating threshold methods, standard deviation-based detection was chosen:

| Approach | Description |
|----------|-------------|
| **Fixed threshold** | Sets a constant similarity cutoff (e.g., 0.4). Splits whenever consecutive sentence similarity falls below this value. Simple to understand but lacks adaptability—optimal thresholds vary significantly across corpora and document types, requiring manual tuning for each dataset ([Qu et al. 2024](https://arxiv.org/abs/2410.13070)). |
| **Percentile-based** | Computes all pairwise distances between consecutive sentences, then splits at distances exceeding the Xth percentile (default: 95th in LangChain/LlamaIndex). Adapts to each document's distance distribution, but percentile thresholds can behave inconsistently—documents with uniform similarity may split at arbitrary points ([LangChain SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)). |
| **Standard deviation** | Splits when similarity drops below `mean - (k × std)` (default: k=3). Self-calibrating: interprets low similarities as statistical outliers, so only significant topic shifts trigger splits. Maps directly to statistical confidence intervals (k=3 ≈ 99.7% CI). Performs well on positively-skewed similarity distributions typical of most documents ([LangChain PR #16807](https://github.com/langchain-ai/langchain/pull/16807)). |

The formula `similarity < mean - (k × std)` treats low similarities as statistical outliers, making breakpoint detection self-calibrating.

---

## Algorithm

The input is NLP-segmented paragraphs with context (book > section). The algorithm embeds sentences, detects semantic breakpoints, then groups sentences into chunks.

```
For each document:
  1. Load NLP-segmented paragraphs
  2. Embed all sentences (batch API call)
  3. Compute cosine similarity between adjacent sentences
  4. Calculate mean and std of all similarities
  5. Mark breakpoints where similarity < mean - (k × std)

  For each sentence:
    If context changed (new section):
      Save current_chunk
      Start new chunk (no overlap across sections)

    If breakpoint detected:
      Save current_chunk
      Start new chunk with last 2 sentences (overlap)

    If (current_chunk + sentence) > MAX_TOKENS:
      Save current_chunk
      Start new chunk with last 2 sentences (overlap)

    Append sentence to current_chunk
```

### Parameters

- **Chunk size:** 800 tokens (max). Enforces upper limit regardless of similarity scores
- **Overlap:** 2 sentences. Maintains context continuity between chunks
- **std_coefficient:** Controls sensitivity (see table below)

| Coefficient | Effect |
|-------------|--------|
| `k = 3.0` | Conservative: only extreme similarity drops trigger splits (99.7% CI) |
| `k = 2.0` | More sensitive: smaller chunks, more splits (95% CI) |
| `k = 4.0` | Very conservative: larger chunks, fewer splits |


## Corpus Analysis

Comparison of semantic chunking with two coefficient values:

<div align="center">

| Corpus | Coefficient | Chunks | Avg Tokens | Median | Single-Chunk Sections |
|--------|-------------|--------|------------|--------|----------------------|
| **Neuroscience** | k=3.0 | 3,419 | 652 | 475 | 3,296 / 3,321 (99%) |
| **Neuroscience** | k=2.0 | 3,854 | 584 | 428 | 2,923 / 3,321 (88%) |
| **Philosophy** | k=3.0 | 576 | 1,324 | 382 | 532 / 545 (98%) |
| **Philosophy** | k=2.0 | 741 | 1,044 | 471 | 455 / 545 (83%) |

</div>

### Comparative Analysis

**std=3.0 (conservative):** Only statistically extreme similarity drops trigger splits. Result: 99% neuroscience and 98% philosophy sections remain as single chunks. Fewer total chunks, larger average size.

**std=2.0 (sensitive):** Triggers on smaller similarity drops (95% CI vs 99.7% CI). Result: 13% more chunks for neuroscience, 29% more for philosophy. Single-chunk rate drops to 88% and 83% respectively. Average tokens decrease ~10-20%.

**vs Section Chunking:** [Section chunking](section-chunking.md) produces 71% (neuroscience) and 64% (philosophy) single-chunk sections—splitting purely on token limits. Both semantic coefficients preserve significantly more content within chunks by only splitting at topic boundaries.

### Example: Section vs Semantic Chunking

The same content handled differently by each strategy:

<details>
<summary><strong>Section Chunking: 2 chunks (782 + 91 tokens)</strong></summary>
<small>

**Chunk 1 (782 tokens)** — Split at token limit, mid-paragraph:
```json
{
  "chunk_id": "Brain and behavior...::chunk_578",
  "context": "Brain and behavior... > CHAPTER 14 Motivation and Reward > Why Motivation Matters",
  "text": "Staying alive is a balancing act. From the moment an animal opens its eyes in the morning, it is faced with a series of dilemmas... [content about survival needs, intelligence vs motivation, internal drives] ...These drives are closely associated with the circuitry of the hypothalamus, as we'll see further below. External drives arise from sources outside the body itself. These include drives in response to threats, sexual or reproductive opportunities, parental attachment drives, social dominance, and affiliation.",
  "token_count": 782
}
```

**Chunk 2 (91 tokens)** — Orphaned continuation:
```json
{
  "chunk_id": "Brain and behavior...::chunk_579",
  "context": "Brain and behavior... > CHAPTER 14 Motivation and Reward > Why Motivation Matters",
  "text": "External drives arise from sources outside the body itself. These include drives in response to threats, sexual or reproductive opportunities, parental attachment drives, social dominance, and affiliation. These types of drives are more closely associated with the circuitry of the amygdala, as we'll see in a moment. At this point in the progress of neuroscience, our list of drives is almost certainly incomplete. However, the ones listed above are the most well studied so far.",
  "token_count": 91
}
```

</small>
</details>

<details>
<summary><strong>Semantic Chunking (std=3.0): 1 chunk (840 tokens)</strong></summary>
<small>

**Single coherent chunk** — No topic boundary detected, content preserved:
```json
{
  "chunk_id": "Brain and behavior...::chunk_451",
  "book_id": "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)",
  "context": "Brain and behavior... > CHAPTER 14 Motivation and Reward > Why Motivation Matters",
  "section": "Why Motivation Matters",
  "text": "Staying alive is a balancing act. From the moment an animal opens its eyes in the morning, it is faced with a series of dilemmas... [full section: survival needs, intelligence vs motivation, internal drives (hypothalamus), external drives (amygdala), concluding note about incomplete drive lists] ...However, the ones listed above are the most well studied so far.",
  "token_count": 840,
  "chunking_strategy": "semantic_std3"
}
```

</small>
</details>

**Key difference:** Section chunking splits at 800 tokens regardless of content, creating a 91-token orphan chunk. Semantic chunking recognizes no significant topic shift occurs and keeps the conceptual unit intact (840 tokens, slightly over limit but semantically coherent).

---



## Navigation

**Next:** [Contextual Chunking](contextual-chunking.md) — LLM-generated context prepended

**Related:**
- [Section Chunking](section-chunking.md) — Token-based baseline strategy
- [RAPTOR](raptor.md) — Hierarchical summarization alternative
- [Chunking Overview](README.md) — Strategy comparison
