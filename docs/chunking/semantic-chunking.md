# Semantic Chunking

[← Section Chunking](section-chunking.md) | [Chunking Overview](README.md)

Splits text at semantic boundaries detected by embedding similarity, creating chunks that preserve topic coherence rather than splitting mid-argument.

---

## Why Standard Deviation-Based Breakpoints

After evaluating different breakpoint detection methods, we chose standard deviation-based detection over fixed thresholds:

| Approach | Problem |
|----------|---------|
| **Fixed threshold** (e.g., 0.4) | No universal best value; requires per-corpus tuning ([Qu et al. 2024](https://arxiv.org/abs/2410.13070)) |
| **Percentile-based** (LangChain/LlamaIndex) | Inconsistent across document types |
| **Standard deviation** | Adapts to each document's similarity distribution; only statistically significant drops trigger splits |

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

## Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Breakpoint method** | Standard deviation | Self-calibrating to document similarity distribution |
| **Default coefficient** | 3.0 | Statistically conservative (99.7% confidence interval) |
| **Section boundaries** | Always respected | Never merge content across markdown headers |
| **Overlap** | 2 sentences | Maintains context continuity between chunks |

---

[← Section Chunking](section-chunking.md) | [Contextual Chunking →](contextual-chunking.md) | [Chunking Overview](README.md)
