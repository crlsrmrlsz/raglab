# RAPTOR: Hierarchical Summarization Tree

[← Contextual Chunking](contextual-chunking.md) | [Home](../../README.md)

[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) builds a hierarchical tree of summaries from document chunks, enabling retrieval at multiple levels of abstraction. It addresses a fundamental gap in flat-chunk RAG: questions requiring synthesis across multiple sections have no single chunk that contains the answer.

Traditional chunking fails for:
- **Theme questions**: "What is the author's central argument?"
- **Multi-section synthesis**: "How do chapters 3 and 7 connect?"
- **Comparative questions**: "What's the difference between X and Y approaches?"

RAPTOR solves this by recursively clustering chunks and generating LLM summaries, creating a tree where higher levels capture broader themes.

<div align="center">
    <img src="../../assets/raptor.png" alt="RAPTOR tree">
</div>



Here RAPTOR is implemented as a **post-processing step on semantic chunks (std=2)**.



## Paper Approach

The paper demonstrated significant improvements on multi-step reasoning tasks, achieving +20% absolute on QuALITY and new state-of-the-art on QASPER. Critically, 18.5-57% of retrieved nodes came from summary layers rather than original chunks, proving that hierarchical abstraction provides information simply not available in leaves alone.

<div align="center">

| Benchmark | RAPTOR | Best Baseline | Improvement |
|-----------|--------|---------------|-------------|
| QuALITY (multi-step reasoning) | 82.6% | 62.7% | **+20% absolute** |
| QASPER (scientific QA) | 55.7% F1 | 53.0% (DPR) | **New SOTA** |

</div>

### The Algorithm

The core insight is combining dimensionality reduction with probabilistic clustering to find semantically coherent groups, then summarizing them recursively:

1. **UMAP** (Uniform Manifold Approximation and Projection) reduces 1536-dim embeddings to 10 dims, preserving local and global structure while making clustering tractable
2. **GMM** (Gaussian Mixture Model) clusters the reduced embeddings, using **BIC** (Bayesian Information Criterion) to automatically select K—balancing fit vs complexity to avoid both under- and over-clustering
3. **LLM summarization** generates a summary for each cluster, creating parent nodes
4. **Recursion** repeats on the summaries until the tree stops growing
5. **Collapsed tree retrieval** queries all nodes (leaves + summaries) together; similarity naturally selects the appropriate abstraction level

The choice of GMM over K-means is deliberate: GMM provides soft clustering where a chunk about "stress and cortisol" can belong to both "neuroscience" AND "health effects" clusters, while K-means forces hard assignment.



## Differences from Paper

The main deviation in this implementation is using larger leaf chunks. The paper used 100-token chunks, optimized for their evaluation datasets (fiction, magazine articles, short papers). This corpus contains dense academic content—neuroscience textbooks and philosophy treatises—where [research](https://arxiv.org/html/2505.21700v2) shows larger chunks dramatically improve retrieval (4.8% → 71.5% accuracy from 64 to 1024 tokens on technical content). The tree structure provides hierarchical value regardless of leaf size.

<div align="center">

| Aspect | Paper | This Implementation |
|--------|-------|---------------------|
| **Leaf chunks** | 100 tokens | Semantic std=2 (~500 tokens avg) |
| **Document scope** | Per-document | Per-book |
| **Summary model** | gpt-3.5-turbo | gpt-4o-mini |
| **Cluster assignment** | Soft (P > 0.3) | Hard (highest probability) |
| **Retrieval** | Collapsed tree | Collapsed tree (same) |

</div>




## Algorithm

The input is semantic chunks (std=2). The algorithm recursively clusters and summarizes until the tree stops growing, then returns all nodes for collapsed-tree retrieval.

```
For each book:
  1. Load semantic chunks (std=2) as level-0 nodes
  2. Embed all nodes

  While nodes.count > MIN_CLUSTER_SIZE:
    1. UMAP reduce → GMM cluster → LLM summarize
    2. Create new nodes at level+1
    3. Embed summary nodes
    4. Repeat with summaries as input

  Return all nodes (leaves + summaries)
```

**Output:** Average summary = 131 tokens (~72% compression). Average children per parent = 6.7 chunks.



## Tree Depths in This Corpus

<div align="center">

| Category | Books | Leaves | Summary Levels | Example |
|----------|-------|--------|----------------|---------|
| **Large** | 5 | 500-880 | 3 (L1→L2→L3) | Cognitive Neuroscience: 881→38→7→3 |
| **Medium** | 6 | 250-500 | 2 (L1→L2) | Letters from a Stoic: 416→21→4 |
| **Short** | 8 | 70-160 | 2 (L1→L2) | Tao Te Ching: 129→14→3 |

</div>



## Navigation

**Next:** [Query-Time Strategies](../preprocessing/README.md) — How queries are transformed

**Related:**
- [Contextual Chunking](contextual-chunking.md) — Alternative approach (can be combined)
- [Semantic Chunking](semantic-chunking.md) — Prerequisite (RAPTOR uses semantic std=2 as leaves)
- [Chunking Overview](README.md) — Strategy comparison
