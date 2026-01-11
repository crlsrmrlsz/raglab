# Semantic Chunking

[← Section Chunking](section-chunking.md) | [Chunking Overview](README.md)

Splits text at semantic boundaries detected by embedding similarity, creating chunks that preserve topic coherence rather than splitting mid-argument.

**Type:** Index-time chunking | **LLM Calls:** 0 | **Embedding Calls:** 1 per sentence

---

## The Problem

Fixed-size chunking splits at arbitrary token boundaries, often fragmenting semantically related content:

```
Fixed-size split (token boundary):
Chunk 1: "...dopamine regulates reward. The mesolimbic pathway extends from"
Chunk 2: "the VTA to the nucleus accumbens, where dopamine release signals..."

Problem: Splits a complete thought about the dopamine pathway.

Semantic split (topic boundary):
Chunk 1: "...dopamine regulates reward. The mesolimbic pathway extends from
          the VTA to the nucleus accumbens, where dopamine release signals..."
Chunk 2: "Serotonin, in contrast, modulates mood through different mechanisms..."

Better: Complete dopamine discussion, then serotonin topic.
```

---

## Theory

### Research: Qu et al. (2024)

The implementation is based on the **breakpoint-based semantic chunker** from [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070) (Qu et al., October 2024).

**Core idea:** Embed each sentence, compute cosine similarity between adjacent sentences, and split where similarity drops below a threshold.

### Approaches Compared in the Paper

| Method | Description | Finding |
|--------|-------------|---------|
| **Percentile-based** | Split at Nth percentile of cosine distances | Inconsistent across documents |
| **Gradient-based** | Split where similarity drops sharply | Computationally expensive |
| **Absolute threshold** | Split when similarity < fixed value | More consistent behavior |

### Key Findings from Qu et al.

1. **Absolute thresholds are more consistent** than percentile-based methods across different document types

2. **No universal best threshold** - the paper tested 0.1-0.5 and found performance is dataset-dependent

3. **Fixed-size chunking may be more practical** - the paper concludes: "fixed-size chunking remains a more efficient and reliable choice for practical RAG applications" when computational costs are considered

4. **Threshold requires tuning** - optimal value depends on your specific corpus and use case

---

## Algorithm

```
For each paragraph:
  1. Embed all sentences (batch API call)
  2. Compute cosine similarity between adjacent sentences
  3. Mark breakpoints where similarity < THRESHOLD

For each sentence:
  4. If breakpoint: save current chunk, start new chunk with overlap
  5. If chunk exceeds safeguard limit: save chunk, start new with overlap
  6. Add sentence to current chunk
```

### Similarity Computation

```python
def compute_similarity_breakpoints(sentences, threshold=0.4):
    """Find semantic breakpoints by embedding similarity."""

    # Embed all sentences (batch API call)
    embeddings = embed_texts(sentences)
    embeddings_array = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms

    # Compute adjacent similarities, mark breakpoints
    breakpoints = [0]  # Always start at 0
    for i in range(len(normalized) - 1):
        sim = np.dot(normalized[i], normalized[i + 1])
        if sim < threshold:
            breakpoints.append(i + 1)  # Start new chunk here

    return breakpoints
```

---

## Implementation Details

### Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Threshold type** | Absolute (not percentile) | Qu et al. found more consistent across documents |
| **Default threshold** | 0.4 | Starting point for tuning; adjust per corpus |
| **Section boundaries** | Always respected | Never merge content across markdown sections |
| **Token safeguard** | 8191 (embedding model limit) | Safety ceiling, not optimization target |
| **Overlap** | 2 sentences | Continuity between chunks |

### Differences from LangChain/LlamaIndex

| Aspect | LangChain/LlamaIndex | RAGLab |
|--------|---------------------|--------|
| Threshold type | 95th percentile of distances | Absolute value |
| Threshold meaning | Relative to document | Consistent across documents |
| Section awareness | No | Yes (markdown headers) |
| Configurability | Limited | Threshold in folder name for A/B testing |

---

## Configuration

### Threshold Parameter

The threshold controls where splits occur:
- **Lower values (e.g., 0.3)**: Fewer splits, larger chunks, more context per chunk
- **Higher values (e.g., 0.5)**: More splits, smaller chunks, tighter topic focus

```bash
# Default threshold (0.4)
python -m src.stages.run_stage_4_chunking --strategy semantic

# Lower threshold (fewer splits, larger chunks)
python -m src.stages.run_stage_4_chunking --strategy semantic --threshold 0.3

# Higher threshold (more splits, smaller chunks)
python -m src.stages.run_stage_4_chunking --strategy semantic --threshold 0.5
```

Output folders are named by threshold (`semantic_0.4/`, `semantic_0.3/`) to enable comparing configurations.

### Config Parameters

```python
# src/config.py

# Similarity threshold for splitting (tune per corpus)
SEMANTIC_SIMILARITY_THRESHOLD = 0.4

# Safety limit (embedding model max input)
EMBEDDING_MAX_INPUT_TOKENS = 8191

# Sentence overlap between chunks
OVERLAP_SENTENCES = 2
```

---

## Cost

Semantic chunking requires embedding API calls during indexing:

| Corpus Size | Estimated Cost | Processing Time |
|-------------|----------------|-----------------|
| ~50,000 sentences | ~$0.65 | ~30 minutes |

Cost driver: One embedding call per sentence (batched). No LLM calls.

---

## When to Use

**Consider semantic chunking when:**
- Topic coherence matters more than consistent chunk sizes
- Your corpus has clear topic transitions within sections
- You can afford embedding costs during indexing

**Consider alternatives when:**
- Computational cost is a concern (use section chunking)
- You need LLM-enhanced context (use contextual chunking)

---

## Running the Pipeline

```bash
# 1. Chunk with semantic strategy
python -m src.stages.run_stage_4_chunking --strategy semantic --threshold 0.4

# 2. Generate embeddings
python -m src.stages.run_stage_5_embedding --strategy semantic_0.4

# 3. Upload to Weaviate
python -m src.stages.run_stage_6_weaviate --strategy semantic_0.4
```

---

## References

- Qu et al. (2024). [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070). arXiv:2410.13070

---

## Navigation

[← Section Chunking](section-chunking.md) | [Contextual Chunking →](contextual-chunking.md) | [Chunking Overview](README.md)
