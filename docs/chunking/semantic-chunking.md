# Semantic Chunking

[← Section Chunking](section-chunking.md) | [Chunking Overview](README.md)

Splits text at semantic boundaries detected by embedding similarity, creating chunks that preserve topic coherence rather than splitting mid-argument.



---

## Origin

### Not Academic Research

Semantic chunking is a **practitioner-developed technique**, not from peer-reviewed research.

| | |
|---|---|
| **Creator** | Greg Kamradt (2023) |
| **Format** | YouTube video + Jupyter notebook |
| **Title** | "5 Levels of Text Splitting" |
| **Source** | [FullStackRetrieval-com/RetrievalTutorials](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) |

**Core idea:** Embed each sentence, compute cosine similarity between adjacent sentences, split where similarity drops below a threshold.

### Adoption

- [LlamaIndex SemanticChunker](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/) - adapted from Kamradt
- LangChain SemanticChunker - similar implementation
- Qu et al. (2024) - later **evaluated** the method academically

### Academic Evaluation: Qu et al. (2024)

[Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070) evaluated semantic chunking methods and found:

1. **Absolute thresholds more consistent** than percentile-based across document types
2. **No universal best threshold** - tested 0.1-0.5, performance is dataset-dependent
3. **Fixed-size chunking may be more practical** - "remains a more efficient and reliable choice for practical RAG applications"

---

## State of the Art (2025)

Beyond the breakpoint method implemented here, several other semantic chunking approaches exist:

### 1. Breakpoint-Based (This Implementation)
Compares embeddings of consecutive sentences; splits when similarity drops below threshold.

- **Threshold types**: Absolute (fixed value), Percentile (Nth percentile of distances), Gradient (sharp drops)
- **Pros**: O(n) complexity, simple to implement, consistent with absolute thresholds
- **Cons**: Only considers adjacent pairs, may miss broader context
- **Research**: [Qu et al. 2024](https://arxiv.org/abs/2410.13070)

### 2. Max-Min Semantic Chunking
Compares max(similarity to new sentence) vs min(similarity within current chunk). Adds sentence only if it fits without hurting chunk cohesion.

- **Pros**: Considers internal chunk coherence, statistically significant improvements over breakpoint
- **Cons**: O(n²) per chunk, more complex implementation
- **Research**: [Kiss et al. 2025](https://link.springer.com/article/10.1007/s10791-025-09638-7)

### 3. Cluster-Based
Uses agglomerative/DBSCAN clustering with combined distance: `λ × positional + (1-λ) × semantic`.

- **Pros**: Can group non-consecutive related sentences, considers document-wide structure
- **Cons**: Harder to control chunk sizes, clustering overhead
- **Research**: [Qu et al. 2024](https://arxiv.org/abs/2410.13070)

### 4. LLM-Based (Propositional)
LLM extracts self-contained semantic propositions directly from text.

- **Pros**: Highest quality, captures complex relationships
- **Cons**: Highest latency and cost, requires LLM calls at index time
- **Research**: [VectorHub 2024](https://superlinked.com/vectorhub/articles/semantic-chunking)

### Quick Comparison

| Method | Complexity | Quality | Cost |
|--------|------------|---------|------|
| Breakpoint (this) | O(n) | Good | Low |
| Max-Min | O(n²) | Better | Medium |
| Cluster | O(n²) | Good | Medium |
| LLM-based | O(n) | Best | High |

---

## Algorithm

```
For each paragraph:
  1. Embed all sentences (batch API call)
  2. Compute cosine similarity between adjacent sentences
  3. Compute mean and std of similarities
  4. Mark breakpoints where similarity < mean - (coefficient * std)

For each sentence:
  5. If breakpoint: save current chunk, start new chunk with overlap
  6. If chunk exceeds safeguard limit: save chunk, start new with overlap
  7. Add sentence to current chunk
```

### Similarity Computation

```python
def compute_similarity_breakpoints(sentences, std_coefficient=3.0):
    """Find semantic breakpoints by embedding similarity using std deviation."""

    # Embed all sentences (batch API call)
    embeddings = embed_texts(sentences)
    embeddings_array = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms

    # Compute adjacent similarities
    similarities = []
    for i in range(len(normalized) - 1):
        sim = np.dot(normalized[i], normalized[i + 1])
        similarities.append(sim)

    # Compute cutoff using standard deviation
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    cutoff = mean_sim - (std_coefficient * std_sim)

    # Mark breakpoints where similarity is statistically low
    breakpoints = [0]  # Always start at 0
    for i, sim in enumerate(similarities):
        if sim < cutoff:
            breakpoints.append(i + 1)  # Start new chunk here

    return breakpoints
```

---

## Implementation Details

### Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Breakpoint detection** | Standard deviation-based | Statistically identifies outlier similarity drops |
| **Default std coefficient** | 3.0 | Conservative, only significant topic shifts trigger breakpoints |
| **Section boundaries** | Always respected | Never merge content across markdown sections |
| **Token safeguard** | 8191 (embedding model limit) | Safety ceiling, not optimization target |
| **Overlap** | 2 sentences | Continuity between chunks |

### Differences from LangChain/LlamaIndex

| Aspect | LangChain/LlamaIndex | RAGLab |
|--------|---------------------|--------|
| Breakpoint detection | Fixed threshold or percentile | Standard deviation-based (mean - k*std) |
| Threshold meaning | Absolute or relative to document | Statistical outlier detection |
| Section awareness | No | Yes (markdown headers) |
| Configurability | Limited | Std coefficient in folder name for A/B testing |

---

## Configuration

### Std Coefficient Parameter

The std coefficient controls where splits occur using statistical outlier detection:
- **Higher values (e.g., 3.0)**: Fewer splits, larger chunks (only extreme drops trigger breakpoints)
- **Lower values (e.g., 2.0)**: More splits, smaller chunks (more sensitive to similarity drops)

```bash
# Default coefficient (3.0)
python -m src.stages.run_stage_4_chunking --strategy semantic

# Lower coefficient (more splits, smaller chunks)
python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0

# Higher coefficient (fewer splits, larger chunks)
python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 4.0
```

Output folders are named by coefficient (`semantic_std3/`, `semantic_std2/`) to enable comparing configurations.

### Config Parameters

```python
# src/config.py

# Standard deviation coefficient for breakpoint detection
SEMANTIC_STD_COEFFICIENT = 3.0

# Safety limit (embedding model max input)
EMBEDDING_MAX_INPUT_TOKENS = 8191

# Sentence overlap between chunks
OVERLAP_SENTENCES = 2
```



## Running the Pipeline

```bash
# 1. Chunk with semantic strategy
python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 3.0

# 2. Generate embeddings
python -m src.stages.run_stage_5_embedding --strategy semantic --std-coefficient 3.0

# 3. Upload to Weaviate
python -m src.stages.run_stage_6_weaviate --strategy semantic --std-coefficient 3.0
```

---

## References

- Kamradt, G. (2023). [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb). FullStackRetrieval Tutorials. *(Original method)*
- Qu et al. (2024). [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070). arXiv:2410.13070. *(Evaluation)*

---

## Navigation

[← Section Chunking](section-chunking.md) | [Contextual Chunking →](contextual-chunking.md) | [Chunking Overview](README.md)
