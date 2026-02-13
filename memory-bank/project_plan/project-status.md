# RAGLab Project Status

**Last Updated:** January 1, 2026

## Overview

RAGLab is a Retrieval-Augmented Generation pipeline designed for learning and experimentation. It processes PDF documents through an 8-stage pipeline to build a searchable knowledge base with AI-powered answers.

**Core Goal:** Master RAG pipeline components while building a practical system for document-based question answering.

## Current Status: Phase 1 Infrastructure Complete

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Extraction | PDF to Markdown | Markdown files |
| 2. Cleaning | Manual review + cleaning | Cleaned MD files |
| 3. Segmentation | NLP sentence segmentation | JSON files |
| 4. Chunking | Section-aware (800 tokens, 2-sentence overlap) | Semantic chunks |
| 5. Embedding | text-embedding-3-large via OpenRouter | Embedding files |
| 6. Weaviate | Vector storage (HNSW + cosine) | Vector objects |
| 7A. Query | `query_similar()`, `query_hybrid()` | weaviate_query.py |
| 7B. UI | Streamlit interface | src/ui/app.py |
| 7C. RAGAS | Evaluation framework | src/evaluation/ |
| 8A. Preprocessing | Strategy-based query transformation | src/rag_pipeline/retrieval/ |
| 8B. Generation | LLM answer synthesis | src/rag_pipeline/generation/ |

## Data Flow

```
data/raw/ (PDF documents)
    |
    v  Stage 1: extract_pdf()
data/processed/01_raw_extraction/
    |
    v  Manual review
data/processed/02_manual_review/
    |
    v  Stage 2: run_structural_cleaning()
data/processed/03_markdown_cleaning/
    |
    v  Stage 3: segment_document()
data/processed/04_nlp_chunks/
    |
    v  Stage 4: run_section_chunking()
data/processed/05_final_chunks/section/
    |
    v  Stage 5: embed_texts()
data/processed/06_embeddings/
    |
    v  Stage 6: upload_embeddings()
Weaviate: RAG_section800_embed3large_v1
    |
    v  Stage 7: Query + UI + Evaluation
    |
    v  Stage 8: Preprocessing + Generation
User Query -> preprocess_query(strategy) -> search -> generate_answer() -> Answer
```

## RAGAS Evaluation

The pipeline includes RAGAS-based evaluation metrics:
- **Faithfulness:** Are answers grounded in the retrieved context?
- **Answer Relevancy:** Do answers address the questions?
- **Context Precision:** Are retrieved chunks relevant?

**Configuration:** Hybrid search, alpha=0.5, top-k=10, cross-encoder reranking

**Reranking Model (Jan 2025):** `mxbai-rerank-xsmall-v1` (70.8M params, BEIR NDCG 43.9)
- Chosen for cross-domain corpus (philosophy + neuroscience)
- 8x faster than large-v1 on CPU (~3s vs ~60s for 50 docs)
- MiniLM alternatives faster but trained only on web search (lower BEIR scores)

## Stage 8: Query Preprocessing + Answer Generation (Completed Dec 22)

| Component | Purpose | Status |
|-----------|---------|--------|
| HyDE | Generate hypothetical answers for semantic matching | Complete |
| Query Decomposition | Break into sub-questions + union merge | Complete |
| Answer Generator | Synthesize LLM answer from retrieved chunks | Complete |
| LLM Call Logging | Log all LLM calls with model and char counts | Complete |

**Key Modules:**
- `src/rag_pipeline/retrieval/` - Strategy-based query transformation
- `src/rag_pipeline/generation/` - LLM answer synthesis with source citations

**Design Decisions (Dec 22):**
- Removed query classification (not in original research papers)
- Each strategy applies its transformation directly to any query
- Unified answer generation prompt (works for all query types)
- LLM call logging: `[LLM] model=X chars_in=Y chars_out=Z`

## RAG Improvement Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Evaluation CLI (--collection, auto-logging) | COMPLETE |
| 1 | Preprocessing Strategy Infrastructure | COMPLETE |
| 2 | Remove Classification + Simplify (Dec 22) | COMPLETE |
| 3 | Multi-Query Strategy | REMOVED (Dec 23) - Subsumed by decomposition |
| 3b | Step-Back → HyDE (Dec 23) | COMPLETE |
| 4 | Query Decomposition (always-on) | COMPLETE |
| 2.5 | Domain-Agnostic Refactoring (Dec 22) | COMPLETE |
| 5 | Comprehensive Evaluation + Alpha Tuning (Dec 24) | COMPLETE |
| 6 | Contextual Chunking (Anthropic-style, Dec 22) | COMPLETE |
| 7 | RAPTOR (hierarchical summarization, Dec 25) | COMPLETE |
| 8 | GraphRAG (Neo4j integration, Dec 25) | COMPLETE |
| 8.1 | GraphRAG Auto-Tuning (per-book resume, Dec 26) | COMPLETE |
| 9 | Codebase Cleanup (Jan 1, 2026) | COMPLETE |

See `memory-bank/rag-improvement-plan.md` for detailed implementation plans.

## Codebase Cleanup (Phase 9, Jan 2026)

Comprehensive code revision for publishing:

| Change | Description |
|--------|-------------|
| Dead code removal | Removed unused `_group_by_pattern()`, orphaned `count_tokens` import |
| Model IDs fixed | Replaced invalid `gpt-5-nano` with `gpt-4o-mini`, later upgraded to Jan 2026 models |
| Type hints modernized | All files use Python 3.9+ style (`list[x]` instead of `List[x]`) |
| Logger naming standardized | All stages use `setup_logging(__name__)` |
| Import patterns unified | All use `from src.shared.files import ...` |
| Return types added | All `main()` functions have `-> None` |
| Prompts extracted | Created `src/prompts.py` for all LLM templates |
| Unused imports removed | Cleaned up 6+ files with orphaned imports |

## GraphRAG Entity Extraction (Phase 8.1)

Entity extraction uses curated types from `src/graph/graphrag_types.yaml` (9 types for dual-domain corpus).

**Process:**
1. LLM extracts entities constrained to predefined types from YAML
2. Relationships extracted with open-ended types (per GraphRAG paper)
3. Results saved per-book for crash recovery
4. Merged into `extraction_results.json` for Neo4j upload

**Key Features:**
- **Per-book atomic processing**: Each book saved as separate JSON file
- **Resumable**: If interrupted, use `--overwrite skip` to continue from first missing book
- **File logging**: Execution logged to `data/logs/extraction_TIMESTAMP.log`
- **OverwriteContext integration**: Same `--overwrite {prompt|skip|all}` pattern as other stages

**Output Files:**
```
data/processed/05_final_chunks/graph/
├── extractions/                  # Per-book results (atomic)
│   ├── Behave, The_Biology....json
│   ├── Biopsychology.json
│   └── ... (17 more)
└── extraction_results.json       # Merged from all books

data/logs/
└── extraction_TIMESTAMP.log      # Execution log
```

**Entity Types (9 curated in graphrag_types.yaml):**
- Generic (1): PERSON
- Neuroscience (4): BRAIN_STRUCTURE, BRAIN_FUNCTION, CHEMICAL, DISORDER
- Psychology bridge (2): MENTAL_STATE, BEHAVIOR
- Frameworks (2): THEORY, PRECEPT

## Strategy Pattern Architecture

The project uses a **Strategy Pattern with Registry** for modular, testable RAG components. This pattern is implemented for preprocessing and will be applied to chunking, embedding, and retrieval.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY PATTERN FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   config.py                    strategies.py                    │
│   ┌──────────────────┐        ┌──────────────────────────────┐ │
│   │ AVAILABLE_*      │───────▶│ STRATEGIES = {               │ │
│   │ DEFAULT_*        │        │   "none": none_strategy,     │ │
│   └──────────────────┘        │   "hyde": hyde_strategy,     │ │
│            │                  │   "decomposition": decomp_...│ │
│            ▼                  │ }                            │ │
│   ┌──────────────────┐        │ def get_strategy(id) -> fn   │ │
│   │ UI Dropdown      │        └──────────────────────────────┘ │
│   │ CLI --arg        │                       │                   │
│   └────────┬─────────┘                       │                   │
│            │                               │                   │
│            └──────────────────▶ dispatcher() ◀─────────────────┘
│                                     │                          │
│                                     ▼                          │
│                            Result dataclass                    │
│                            (with strategy_used)                │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Config** | `src/config.py` | `AVAILABLE_*_STRATEGIES` list, `DEFAULT_*_STRATEGY` |
| **Registry** | `src/*/strategies.py` | `STRATEGIES` dict mapping ID → function |
| **Dispatcher** | Main module | `process_*(strategy=...)` routes to correct function |
| **Result** | Dataclass | Contains `strategy_used` field for tracking |
| **UI** | `src/ui/app.py` | Dropdown populated from `AVAILABLE_*` |
| **CLI** | `src/run_stage_*.py` | `--strategy` argument with choices |
| **Logging** | `src/shared/query_logger.py` | Records `strategy` in JSON logs |

### Implemented: Preprocessing Strategies

**Files:**
- `src/config.py:AVAILABLE_PREPROCESSING_STRATEGIES`
- `src/rag_pipeline/retrieval/strategy_registry.py` (registry)
- `src/rag_pipeline/retrieval/query_preprocessing.py` (dispatcher)
- `src/rag_pipeline/retrieval/rrf.py` (RRF merging for graphrag)

**Available strategies (each applies directly to any query):**
- `none` - No transformation, use original query (0 LLM calls)
- `hyde` - Generate hypothetical answer for semantic matching (1 LLM call, 1 search) [arXiv:2212.10496]
- `decomposition` - Break into 2-4 sub-questions + union merge (1 LLM call, 3-4 searches) [arXiv:2507.00355]
- `graphrag` - Hybrid graph + vector retrieval via RRF (entity traversal + semantic search) [arXiv:2404.16130]

### Implemented: Chunking Strategies

**Files:**
- `src/config.py:AVAILABLE_CHUNKING_STRATEGIES`
- `src/rag_pipeline/chunking/strategies.py` (registry)
- `src/rag_pipeline/chunking/section_chunker.py` (section strategy)
- `src/rag_pipeline/chunking/contextual_chunker.py` (contextual strategy)

**Available strategies:**
- `section` - 800-token section-aware chunks with 2-sentence overlap
- `semantic` - Similarity threshold-based boundaries
- `contextual` - LLM-generated context prepended to chunks (Anthropic-style)
- `raptor` - Hierarchical summarization tree with dynamic n_neighbors (sqrt formula like original RAPTOR) [arXiv:2401.18059]

### Implemented: Retrieval Strategies

Retrieval is handled via search functions in `src/rag_pipeline/indexing/weaviate_query.py`:
- `vector` - Pure semantic search via `query_similar()`
- `hybrid` - BM25 + vector with alpha via `query_hybrid()`
- `graphrag` - Graph-augmented retrieval via `src/graph/query.py` (COMPLETE)

## LLM Response Validation

Uses Pydantic schemas for type-safe LLM outputs with JSON Schema enforcement:
- `src/shared/schemas.py` - `get_openrouter_schema()` utility
- `src/rag_pipeline/retrieval/query_schemas.py` - Response models (DecompositionResult)

Key function: `call_structured_completion(messages, model, response_model)` in `openrouter_client.py`

## Run Commands

```bash
conda activate raglab

# Pipeline stages (baseline)
python -m src.stages.run_stage_1_extraction
python -m src.stages.run_stage_2_processing
python -m src.stages.run_stage_3_segmentation
python -m src.stages.run_stage_4_chunking
python -m src.stages.run_stage_5_embedding
python -m src.stages.run_stage_6_weaviate

# RAPTOR (hierarchical summarization)
# Tree depth is dynamic based on corpus size:
#   - 100-1000 chunks: typically 2 levels
#   - 5000+ chunks: typically 3-4 levels
# Uses dynamic n_neighbors = sqrt(n-1) like original RAPTOR paper
python -m src.stages.run_stage_4b_raptor

# GraphRAG (Neo4j knowledge graph)
python -m src.stages.run_stage_4c_graph_extract  # Extract entities/relationships (resumable)
python -m src.stages.run_stage_6b_neo4j           # Upload to Neo4j + Leiden communities

# Extraction resume options:
python -m src.stages.run_stage_4c_graph_extract --overwrite skip  # Resume from failure
python -m src.stages.run_stage_4c_graph_extract --overwrite all   # Force reprocess
python -m src.stages.run_stage_4c_graph_extract --list-books      # Preview books

# UI
streamlit run src/ui/app.py

# Evaluation (see model-selection.md for model options)
python -m src.stages.run_stage_7_evaluation                  # Single config
python -m src.stages.run_stage_7_evaluation --comprehensive  # Grid search all configs
```

## Code Standards

See `CLAUDE.md` for complete standards:
- Function-based design (classes only for state)
- Absolute imports (`from src.module import ...`)
- Modern Python 3.9+ type hints (`list`, `dict`, `tuple` instead of `List`, `Dict`, `Tuple`)
- Fail-fast error handling
- Logger only (no print, no emoji)
- Google-style docstrings
- All LLM prompts in `src/prompts.py` (imported by config.py for backward compatibility)
