# Evaluation Framework

[← GraphRAG](../preprocessing/graphrag.md) | [Home](../../README.md)


RAGLab uses [RAGAS (Retrieval-Augmented Generation Assessment)](https://docs.ragas.io/) to evaluate different RAG configurations, systematically testing which combinations of chunking, alpha (search balance), reranking, and preprocessing produce the best answers. RAGAS provides LLM-as-judge metrics that don't require human evaluation for every run, enabling automated comparison across many configurations. Unlike benchmarks that test generation given pre-retrieved documents (e.g., RAGBench), RAGLab evaluates the **full end-to-end pipeline** — from retrieval through generation — so metric differences reflect real pipeline behavior, not just LLM output quality.

The **comprehensive mode** runs a 4D grid search across all valid combinations using a curated 15-question subset. **Single mode** runs one configuration against all 45 questions. Top-k is fixed at 15 for all evaluation runs.

### Comprehensive Grid

<div align="center">

| Dimension | Values | Notes |
|-----------|--------|-------|
| **Collections** | section, semantic_std2, semantic_std3, contextual, raptor | Chunking strategy used at index time |
| **Alpha** | 0.0, 0.5, 1.0 | 0.0 = pure BM25, 0.5 = hybrid, 1.0 = pure semantic |
| **Reranking** | on, off | Cross-encoder re-scoring of retrieved chunks |
| **Strategy** | none, hyde, decomposition, graphrag | Query-time preprocessing |

</div>

Not all combinations are valid — each strategy declares constraints that reduce the grid. The table below shows what each strategy allows:

<div align="center">

| Strategy | Alpha | Reranking | Collections | Rationale |
|----------|-------|-----------|-------------|-----------|
| **none** | any (0.0, 0.5, 1.0) | optional (on/off) | all 5 | Baseline, no constraints |
| **hyde** | fixed (1.0) | optional (on/off) | all 5 | Paper requires pure semantic search |
| **decomposition** | fixed (1.0) | required (on) | all 5 | Paper requires cross-encoder reranking |
| **graphrag** | fixed (1.0) | forbidden (off) | dedicated (semantic_std2) | Own graph-based retrieval and ranking |

</div>

These constraints are enforced by `StrategyConfig` — a declarative configuration system that serves as the single source of truth. The UI, evaluation grid, and runtime retrieval all consult it to prevent invalid states, so only valid combinations are tested.



## RAGAS Metrics

Five metrics evaluate both retrieval and generation quality:

<div align="center">

| Metric | Category | What It Measures | Reference Required |
|--------|----------|------------------|--------------------|
| **Faithfulness** | Generation | Are claims grounded in retrieved context? | No |
| **Relevancy** | Generation | Does the answer address the question? | No |
| **Context Precision** | Retrieval | Are retrieved chunks actually relevant? | No |
| **Context Recall** | Retrieval | Did retrieval capture all needed information? | Yes |
| **Answer Correctness** | End-to-end | Is the answer factually correct? | Yes |

</div>

**Reference-free** metrics (faithfulness, relevancy, precision) use only the query, retrieved context, and generated answer. **Reference-required** metrics (recall, correctness) compare against human-written ground truth answers.

Answer Correctness combines two signals: **75% factual similarity** (statement overlap with reference) and **25% semantic similarity** (embedding distance). This weights factual accuracy over surface-level phrasing.

### Key finding: Recall > Precision

The generator LLM can filter irrelevant context (low precision is recoverable), but cannot invent missing information (low recall is unrecoverable). This means retrieval should prioritize casting a wide net over returning only precise results.

### GraphRAG metric differences

GraphRAG contributes fewer grid combinations because it operates under dedicated constraints: a fixed collection (entity-chunk ID matching), fixed alpha (1.0), and its own graph-based ranking. The same 5 RAGAS metrics apply, but the retrieval path differs — entity extraction, graph traversal, and `combined_degree` ranking replace standard vector search.

---

## Test Dataset

### Corpus

19 books spanning two domains:
- **Neuroscience** (9 books, ~4,800 chunks): Sapolsky, Kahneman, Eagleman, Gazzaniga
- **Philosophy** (10 books, ~1,400 chunks): Stoics, Schopenhauer, Tao Te Ching, Confucius

### Question design

45 questions total, each with a paragraph-length human-written reference answer.

<div align="center">

| Difficulty | Count | Description |
|------------|-------|-------------|
| **Single-concept** | 5 | Factual questions within one domain |
| **Cross-domain** | 10+ | Require synthesis across neuroscience and philosophy |

</div>

**Design principles:**
- **Open-ended phrasing** — Questions don't mention "neuroscience AND philosophy" explicitly; they ask broadly, testing whether RAG retrieves cross-domain content on its own
- **Multi-book retrieval** — Each question requires content from 4-6 books to answer fully
- **15 cross-cutting themes** — Free will, self-control, suffering, happiness, death, and others that bridge both domains

### Comprehensive subset

The 15-question curated subset (`comprehensive_questions.json`) was selected for grid search based on: multi-source retrieval requirement, synthesis difficulty, and coverage of both single-concept and cross-domain question types. This smaller set keeps grid search tractable while maintaining evaluation signal.

---

## Running Evaluation

```bash
# Single configuration (full 45 questions)
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --search-type hybrid \
  --preprocessing hyde \
  --alpha 0.7 \
  --top-k 15

# Comprehensive grid search (15-question curated subset)
python -m src.stages.run_stage_7_evaluation --comprehensive

# Retry failed combinations from a previous run
python -m src.stages.run_stage_7_evaluation --retry-failed comprehensive_20251231_120000
```

<div align="center">

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--alpha`, `-a` | 0.0-1.0 | 0.5 | Search balance: 0.0=BM25, 0.5=hybrid, 1.0=semantic |
| `--preprocessing`, `-p` | none, hyde, decomposition, graphrag | none | Query transformation strategy |
| `--top-k`, `-k` | int | 10 | Chunks to retrieve (fixed at 15 in comprehensive) |
| `--collection` | string | auto | Weaviate collection name |
| `--reranking` | flag | off | Enable cross-encoder reranking |
| `--comprehensive` | flag | - | Run 4D grid search over all valid combinations |

</div>

---

## Navigation

**Next:** [Evaluation Results](results.md) — Metrics and leaderboards

**Related:**
- [Preprocessing Strategies](../preprocessing/README.md) — Query-time techniques being evaluated
- [Chunking Strategies](../chunking/README.md) — Index-time strategies being evaluated
