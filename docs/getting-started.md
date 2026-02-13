# Getting Started

[Home](../README.md)

## About This Project

RAGLab is a **learning and showcase project** created to experiment with RAG (Retrieval-Augmented Generation) improvement techniques. I built this to understand data processing, how different chunking strategies and query preprocessing methods affect performance, and to learn how evaluate with RAGAS.

**Important:** The data used in this project (19 books on neuroscience and philosophy) is **not included** in the repository due to intellectual property protection. The code is published for educational purposes.

You could use this project to:
- Learn how a complete RAG pipeline works
- Create your own dataset with your own documents
- Experiment with different RAG techniques


## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for Weaviate and Neo4j)
- **OpenRouter API key** (for embeddings and LLM calls) 



## Installation

```bash
# Clone repository
git clone https://github.com/crlsrmrlsz/raglab.git
cd raglab

# Create conda environment (recommended)
conda create -n raglab python=3.10
conda activate raglab

# Install dependencies
pip install -e .

# Install spaCy model (for sentence segmentation)
# Use scispaCy for scientific/medical text (better boundary detection):
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
# Or use standard model for general text:
# python -m spacy download en_core_web_sm

# Create .env file with your API key (code loads from src/)
cp .env.example src/.env
# Edit src/.env and add: OPENROUTER_API_KEY=your_key_here

# Start Docker services
docker compose up -d
```

---

## Database Setup

The project uses two databases, both running in Docker. No manual database installation required - everything is configured in `docker-compose.yml`:

**Weaviate** (vector database) - Required for all pipelines
- Stores chunk embeddings for semantic search
- Stores GraphRAG community summaries for global context retrieval
- Ports: 8080 (REST), 50051 (gRPC)

**Neo4j** (graph database) - Required only for GraphRAG
- Stores knowledge graph entities and relationships
- Includes GDS plugin for Leiden community detection
- Ports: 7474 (browser), 7687 (Bolt)

```bash
# Start both databases
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f weaviate
docker compose logs -f neo4j

# Verify Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# Access Neo4j browser (optional)
# Open http://localhost:7474 (user: neo4j, password: raglab_graphrag)

# Stop databases
docker compose down
```

See `docker-compose.yml` for detailed configuration options and comments.



## Configuration

Key settings are in `src/config.py`. Some important parameters:

<div align="center">

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CHUNK_TOKENS` | 800 | Target chunk size |
| `OVERLAP_SENTENCES` | 2 | Sentence overlap between chunks |
| `DEFAULT_TOP_K` | 10 | Chunks to retrieve per query |
| `PREPROCESSING_MODEL` | `deepseek/deepseek-v3.2` | Model for HyDE/decomposition |
| `GENERATION_MODEL` | `openai/gpt-5-mini` | Model for answer generation |
| `RERANK_MODEL` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Cross-encoder for reranking |

</div>

Database connections can be configured via environment variables in `.env`:
```
WEAVIATE_HOST=localhost
WEAVIATE_HTTP_PORT=8080
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=raglab_graphrag
```


## Project Structure

```
src/
├── content_preparation/    # PDF → clean text
├── rag_pipeline/           # chunking, embedding, retrieval, generation
│   ├── chunking/           # Section, contextual, semantic chunkers
│   ├── embedder.py         # OpenRouter embedding API (OpenAI-compatible)
│   ├── indexing/           # Weaviate upload
│   ├── retrieval/          # Search + preprocessing strategies
│   └── generation/         # Answer generation
├── graph/                  # GraphRAG: Neo4j, Leiden communities
├── evaluation/             # RAGAS evaluation framework
├── ui/                     # Streamlit application
└── stages/                 # CLI entry points for each stage

.data/                          # Database storage (Docker volumes, gitignored)
├── weaviate/                   # Weaviate vector database files
└── neo4j/                      # Neo4j graph database files

data/
├── raw/                    # Your source PDFs go here
│   └── {corpus}/           # Organize by topic (e.g., mybooks/)
└── processed/              # Pipeline outputs (auto-created)
```

---

## Data Folder Structure

When you run the pipeline, data flows through these folders:

<div align="center">

| Folder | Created By | Contains |
|--------|------------|----------|
| `data/raw/{corpus}/` | You | Your source PDF files |
| `data/processed/01_raw_extraction/` | Stage 1 | Markdown extracted from PDFs |
| `data/processed/02_manual_review/` | (Optional) | Reviewed markdown before cleaning |
| `data/processed/03_markdown_cleaning/` | Stage 2 | Cleaned markdown files |
| `data/processed/04_nlp_chunks/` | Stage 3 | Sentence-segmented JSON |
| `data/processed/05_final_chunks/{strategy}/` | Stage 4 | Chunks ready for embedding |
| `data/processed/06_embeddings/{strategy}/` | Stage 5 | Chunks with vector embeddings |
| `data/processed/05_final_chunks/graph/` | Stage 4c | GraphRAG entities and relationships |

</div>

---

## Pipeline Stages

<div align="center">

| Stage | Command | What It Does |
|-------|---------|--------------|
| 1 | `python -m src.stages.run_stage_1_extraction` | PDF → Markdown (using Docling) |
| 2 | `python -m src.stages.run_stage_2_processing` | Clean markdown (remove artifacts) |
| 3 | `python -m src.stages.run_stage_3_segmentation` | Sentence segmentation (spaCy NLP) |
| 4 | `python -m src.stages.run_stage_4_chunking` | Create chunks (800 tokens, 2-sentence overlap) |
| 4b | `python -m src.stages.run_stage_4b_raptor` | RAPTOR hierarchical tree (optional) |
| 4c | `python -m src.stages.run_stage_4c_graph_extract` | GraphRAG entity extraction (optional) |
| 5 | `python -m src.stages.run_stage_5_embedding` | Generate embeddings (OpenAI API) |
| 6 | `python -m src.stages.run_stage_6_weaviate` | Upload to Weaviate vector database |
| 6b | `python -m src.stages.run_stage_6b_neo4j` | Upload to Neo4j + Leiden communities (optional) |
| 7 | `python -m src.stages.run_stage_7_evaluation` | RAGAS evaluation |

</div>

**Note:** Stages 4b, 4c, and 6b are optional advanced techniques. The basic pipeline is stages 1-7.



## Advanced Pipelines

### RAPTOR (Hierarchical Summarization)

RAPTOR builds a tree of summaries enabling both detailed and thematic retrieval.

```bash
# After running stages 1-4
python -m src.stages.run_stage_4b_raptor      # Build summary tree
python -m src.stages.run_stage_5_embedding --strategy raptor
python -m src.stages.run_stage_6_weaviate --strategy raptor
```

### GraphRAG (Knowledge Graph + Communities)

GraphRAG extracts entities and relationships, detects communities using the Leiden algorithm, and enables cross-document reasoning.

```bash
# After running stages 1-4
python -m src.stages.run_stage_4c_graph_extract  # Extract entities + relationships
python -m src.stages.run_stage_6b_neo4j        # Upload to Neo4j + run Leiden

# Then use --preprocessing graphrag in evaluation or UI
```


## Quick Start

The fastest path to get a working RAG system with your own documents:

1. **Add your PDFs**
   ```bash
   mkdir -p data/raw/mybooks
   cp /path/to/your/*.pdf data/raw/mybooks/
   ```

2. **Run the pipeline** (stages 1-6)
   ```bash
   python -m src.stages.run_stage_1_extraction
   python -m src.stages.run_stage_2_processing
   python -m src.stages.run_stage_3_segmentation
   python -m src.stages.run_stage_4_chunking
   python -m src.stages.run_stage_5_embedding
   python -m src.stages.run_stage_6_weaviate
   ```

3. **Query your documents**
   ```bash
   streamlit run src/ui/app.py
   ```

---

## Navigation

**Next:** [Content Preparation](content-preparation/README.md) — PDF extraction and cleaning
