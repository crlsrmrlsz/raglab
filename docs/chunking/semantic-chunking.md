# Semantic Chunking

[← Section Chunking](section-chunking.md) | [Chunking Overview](README.md)

Splits text at semantic boundaries detected by embedding similarity, creating chunks that preserve topic coherence rather than splitting mid-argument.

---

## Why Standard Deviation-Based Breakpoints

After evaluating different breakpoint detection methods, I chose to test standard deviation-based detection over fixed thresholds:

| Approach | Description |
|----------|-------------|
| **Fixed threshold** | Sets a constant similarity cutoff (e.g., 0.4). Splits whenever consecutive sentence similarity falls below this value. Simple to understand but lacks adaptability—optimal thresholds vary significantly across corpora and document types, requiring manual tuning for each dataset ([Qu et al. 2024](https://arxiv.org/abs/2410.13070)). |
| **Percentile-based** | Computes all pairwise distances between consecutive sentences, then splits at distances exceeding the Xth percentile (default: 95th in LangChain/LlamaIndex). Adapts to each document's distance distribution, but percentile thresholds can behave inconsistently—documents with uniform similarity may split at arbitrary points ([LangChain SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)). |
| **Standard deviation** | Splits when similarity drops below `mean - (k × std)` (default: k=3). Self-calibrating: interprets low similarities as statistical outliers, so only significant topic shifts trigger splits. Maps directly to statistical confidence intervals (k=3 ≈ 99.7% CI). Performs well on positively-skewed similarity distributions typical of most documents ([LangChain PR #16807](https://github.com/langchain-ai/langchain/pull/16807)). |

The formula `similarity < mean - (k × std)` treats low similarities as statistical outliers, making breakpoint detection self-calibrating.

---

## Algorithm

```
For each paragraph:
  1. Embed all sentences (batch API call)
  2. Compute cosine similarity between adjacent sentences
  3. Calculate mean and std of all similarities
  4. Mark breakpoints where similarity < mean - (coefficient × std)

For each sentence:
  5. If breakpoint: save current chunk, start new with overlap
  6. If chunk exceeds token limit: save chunk, start new with overlap
  7. Add sentence to current chunk
```

### Core Implementation

```python
def compute_similarity_breakpoints(sentences, std_coefficient=3.0):
    """Find semantic breakpoints using standard deviation."""

    # Embed and normalize
    embeddings = embed_texts(sentences)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute adjacent similarities
    similarities = [np.dot(normalized[i], normalized[i + 1])
                    for i in range(len(normalized) - 1)]

    # Statistical cutoff
    cutoff = np.mean(similarities) - (std_coefficient * np.std(similarities))

    # Mark breakpoints
    breakpoints = [0]
    for i, sim in enumerate(similarities):
        if sim < cutoff:
            breakpoints.append(i + 1)

    return breakpoints
```

---

## Configuration

```python
# src/config.py
SEMANTIC_STD_COEFFICIENT = 3.0   # Standard deviations below mean for breakpoint
EMBEDDING_MAX_INPUT_TOKENS = 8191  # Safety limit (embedding model max)
OVERLAP_SENTENCES = 2              # Sentence overlap between chunks
```

| Parameter | Effect |
|-----------|--------|
| `std_coefficient = 3.0` | Conservative: only extreme similarity drops trigger splits |
| `std_coefficient = 2.0` | More sensitive: smaller chunks, more splits |
| `std_coefficient = 4.0` | Very conservative: larger chunks, fewer splits |

---

## Usage

```bash
# Default (coefficient = 3.0)
python -m src.stages.run_stage_4_chunking --strategy semantic

# Custom coefficient
python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0
```

Output: `data/processed/05_final_chunks/semantic_std{coefficient}/`

---

## Corpus Analysis

Analysis of the semantic chunking output (std=3.0) reveals how the conservative coefficient affects chunk distribution:

<div align="center">

| Corpus | Avg Chunk Tokens | Median | Single-Chunk Sections |
|--------|------------------|--------|----------------------|
| **Neuroscience** | 652 | 475 | 3,296 / 3,321 (99%) |
| **Philosophy** | 1,324 | 382 | 532 / 545 (98%) |

</div>

The high single-chunk percentage (98-99%) reflects the conservative std=3.0 coefficient—only statistically extreme similarity drops trigger splits. Philosophy's lower median (382) versus high average (1,324) indicates skewed distribution: many short aphoristic sections alongside lengthy essays that remain unsplit.

Compared to [section chunking](section-chunking.md) (71% and 64% single-chunk sections), semantic chunking preserves more content within single chunks by only splitting at major topic shifts. A lower coefficient (e.g., std=2.0) would increase sensitivity and reduce chunk sizes.

---

## Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Breakpoint method** | Standard deviation | Self-calibrating to document similarity distribution |
| **Default coefficient** | 3.0 | Statistically conservative (99.7% confidence interval) |
| **Section boundaries** | Always respected | Never merge content across markdown headers |
| **Overlap** | 2 sentences | Maintains context continuity between chunks |

---

## Empirical Analysis

Analysis of the semantic chunking output (std=3.0) reveals how the conservative coefficient affects chunk distribution:

| Corpus | Avg Chunk Tokens | Median | Single-Chunk Sections |
|--------|------------------|--------|----------------------|
| Neuroscience | 651 | 475 | 3,185 / 3,219 (98%) |
| Philosophy | 1,323 | 381 | 532 / 545 (97%) |

Analysis of the semantic chunking output (std=2.0) shows the effect of a more sensitive coefficient:

| Corpus | Avg Chunk Tokens | Median | Single-Chunk Sections |
|--------|------------------|--------|----------------------|
| Neuroscience | 584 | 428 | 2,819 / 3,219 (87%) |
| Philosophy | 1,044 | 471 | 455 / 545 (83%) |

The std=2.0 coefficient produces smaller chunks (10-20% fewer tokens on average) and more multi-chunk sections, as it triggers splits on less extreme similarity drops.

---

[← Section Chunking](section-chunking.md) | [Contextual Chunking →](contextual-chunking.md) | [Chunking Overview](README.md)
