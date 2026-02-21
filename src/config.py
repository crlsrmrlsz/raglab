"""Central configuration for RAGLab pipeline.

Contains:
- Project paths for all pipeline stages (extraction through embedding)
- Text cleaning patterns (line removal, inline removal, substitutions)
- NLP settings (spaCy model, sentence filtering)
- Chunking parameters (token limits, overlap)
- Embedding settings (API configuration via .env)
- Weaviate vector database settings
"""
import re
from pathlib import Path

from dotenv import load_dotenv
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Processing pipeline directories
DIR_RAW_EXTRACT = DATA_DIR / "processed" / "01_raw_extraction"
DIR_MANUAL_REVIEW = DATA_DIR / "processed" / "02_manual_review"
DIR_DEBUG_CLEAN = DATA_DIR / "processed" / "03_markdown_cleaning"
DIR_NLP_CHUNKS = DATA_DIR / "processed" / "04_nlp_chunks"
DIR_FINAL_CHUNKS = DATA_DIR / "processed" / "05_final_chunks"
DIR_EMBEDDINGS = DATA_DIR / "processed" / "06_embeddings"

# Logging
DIR_CLEANING_LOGS = DATA_DIR / "logs"
DIR_CLEANING_LOGS.mkdir(parents=True, exist_ok=True)
CLEANING_LOG_FILE = DIR_CLEANING_LOGS / "cleaning_report.log"


# ============================================================================
# CLEANING PATTERNS
# ============================================================================

# Pattern format: (regex_pattern, pattern_name)
# All patterns use explicit case matching in regex (no re.IGNORECASE needed)

LINE_REMOVAL_PATTERNS: list[tuple[str, str]] = [
    # Figure/Table captions: "Figure 2. Model diagram", "Table 1-3: Flow Chart"
    # Matches: Figure/Fig/Table/Tab + number/identifier + UPPERCASE start
    # Preserves: "Figure 2 shows..." (lowercase after number)
    (
        r'^\s*(#+\s*)?([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s+[\w\.\-]+\s+[A-Z]',
        'FIGURE_TABLE_CAPTION'
    ),
    
    # Learning objectives: "LO 1.2", "LO 5"
    (
        r'^\s*(##\s*)?LO\s+\d',
        'LEARNING_OBJECTIVE'
    ),
    
    # Single character lines: isolated letters, numbers, symbols
    (
        r'^\s*[a-zA-Z0-9\.\|\-]\s*$',
        'SINGLE_CHAR'
    ),
    
    # Heading with only a number: "## 5"
    (
        r'^\s*##\s+\d+\s*$',
        'HEADING_SINGLE_NUMBER'
    ),
]


INLINE_REMOVAL_PATTERNS: list[tuple[str, str]] = [
    # Figure/table references in parentheses: "(Figure 2)", "(TABLE 1-3)"
    (
        r'\(\s*([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s*[\d\.\-]+[a-zA-Z]?\s*\)',
        'FIG_TABLE_REF'
    ),
    
    # Footnote markers: "fn3", "fn12" (typically appear mid-sentence)
    (
        r'\bfn\d+\b\s*',
        'FOOTNOTE_MARKER'
    ),
    
    # Standalone numbers after punctuation: ". 81 We" -> ". We"
    # Removes page numbers and footnote references
    (
        r'(?<=[.!?\"\'])\s+\d+\s+(?=[A-Z])',
        'TRAILING_NUMBER'
    ),
]


CHARACTER_SUBSTITUTIONS: list[tuple[str, str, str]] = [
    # Format: (old_string, new_string, substitution_name)
    ('/u2014.d', '--', 'EM_DASH_WITH_SUFFIX'),
    ('/u2014', '--', 'EM_DASH'),
    ('&', '&', 'HTML_AMPERSAND'),
]


# List marker pattern: used in special processing function
LIST_MARKER_PATTERN = r'^\s*\([a-z]\)\s+'

# Punctuation for paragraph merging decisions
TERMINAL_PUNCTUATION = ('.', '!', '?', ':', ';', '"', '"')
SENTENCE_ENDING_PUNCTUATION = ('.', '!', '?', '"', '"')

# Report formatting
REPORT_WIDTH = 100


# ============================================================================
# NLP SETTINGS
# ============================================================================

SPACY_MODEL = "en_core_sci_sm"

# Valid sentence endings for filtering
VALID_ENDINGS = ('.', '?', '!', '"', '"', ')', ']')

# Sentence filtering
MIN_SENTENCE_WORDS = 2

# Markdown header detection
HEADER_CHAPTER = '# '
HEADER_SECTION = '##'

# Context string formatting
CONTEXT_SEPARATOR = ' > '


# ============================================================================
# CHUNKING SETTINGS
# ============================================================================


# Chunking parameters
MAX_CHUNK_TOKENS = 800  # Target size for section chunking (sequential accumulation)

# Tokenizer model name (OpenAI compatible)
TOKENIZER_MODEL = "text-embedding-3-large"

# Embedding model input limit (safeguard for semantic chunking)
# text-embedding-3-large max context: 8191 tokens
# Semantic chunking uses this as safety ceiling, not optimization target
EMBEDDING_MAX_INPUT_TOKENS = 8191

# Configurable overlap: number of sentences to carry from previous chunk
OVERLAP_SENTENCES = 2  # Adjust this value (0 = no overlap, 2-3 recommended)

# Chunk ID formatting
CHUNK_ID_PREFIX = 'chunk'
CHUNK_ID_SEPARATOR = '::'

# Safety limits for chunking loops
MAX_LOOP_ITERATIONS = 1000


# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================


# Embedding model (OpenAI / OpenRouter compatible)
EMBEDDING_MODEL = TOKENIZER_MODEL  # "text-embedding-3-large"

# Safety limits
MAX_BATCH_TOKENS = 12_000   # conservative batch size
MAX_RETRIES = 3

# Load environment variables from the .env file (in src/ directory)
load_dotenv(Path(__file__).parent / ".env")

# Now you can access the variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')


def validate_api_key() -> None:
    """Validate that OPENROUTER_API_KEY is set.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.

    Note:
        Call this function at the start of any stage that requires API access.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your API key."
        )


# ============================================================================
# WEAVIATE SETTINGS
# ============================================================================

# Connection settings
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
WEAVIATE_HTTP_PORT = int(os.getenv('WEAVIATE_HTTP_PORT', '8080'))
WEAVIATE_GRPC_PORT = int(os.getenv('WEAVIATE_GRPC_PORT', '50051'))

# Batch upload settings
WEAVIATE_BATCH_SIZE = 100  # Objects per batch (Weaviate recommends 100-1000)

# Collection naming components (auto-generated)
# Format: RAG_{chunking_strategy}_{embedding_model_short}_{version}
CHUNKING_STRATEGY_NAME = "section800"  # Describes current chunking approach
EMBEDDING_MODEL_SHORT = "embed3large"  # Short name for embedding model
COLLECTION_VERSION = "v1"  # Increment when re-running with same strategy


def get_collection_name(chunking_strategy: str = None) -> str:
    """
    Generate collection name from pipeline configuration.

    Args:
        chunking_strategy: Chunking strategy name (e.g., "section", "semantic").
            If None, uses CHUNKING_STRATEGY_NAME from config.

    Returns:
        Collection name in format: RAG_{strategy}_{model}_{version}

    Example:
        >>> get_collection_name()  # Uses config default
        "RAG_section800_embed3large_v1"
        >>> get_collection_name("semantic")
        "RAG_semantic_embed3large_v1"
    """
    strategy = chunking_strategy if chunking_strategy else CHUNKING_STRATEGY_NAME
    strategy_safe = strategy.replace(".", "_")
    return f"RAG_{strategy_safe}_{EMBEDDING_MODEL_SHORT}_{COLLECTION_VERSION}"


def get_community_collection_name() -> str:
    """
    Generate community collection name for GraphRAG.

    Community embeddings are stored in Weaviate for efficient vector search.
    Uses GRAPHRAG_CHUNKING_STRATEGY since communities are GraphRAG-only.

    Returns:
        Collection name in format: Community_{strategy}_{version}

    Example:
        >>> get_community_collection_name()
        "Community_semantic_std2_v1"
    """
    strategy_safe = GRAPHRAG_CHUNKING_STRATEGY.replace(".", "_")
    return f"Community_{strategy_safe}_{COLLECTION_VERSION}"


# ============================================================================
# UI SETTINGS
# ============================================================================

# Search UI defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 20

# Reranking: Cross-encoder model for two-stage retrieval
#
# DECISION: mxbai-rerank-xsmall-v1 (Jan 2025)
# - Corpus is cross-domain (philosophy + neuroscience) requiring diverse training
# - MiniLM models trained only on MS MARCO (web search) underperform on BEIR (+16% gap)
# - mxbai-xsmall offers 8x speedup vs large-v1 while retaining diverse training
# - Expected: ~3s for 50 docs on CPU (vs ~1min with large-v1)
#
# Model Comparison:
# ┌─────────────────────────────────────┬─────────┬────────────┬──────────────────┐
# │ Model                               │ Params  │ BEIR NDCG  │ CPU Speed        │
# ├─────────────────────────────────────┼─────────┼────────────┼──────────────────┤
# │ cross-encoder/ms-marco-MiniLM-L-2   │ 15.6M   │ ~35*       │ Fastest (1x)     │
# │ cross-encoder/ms-marco-MiniLM-L-6   │ 22.7M   │ ~38*       │ Fast (2.4x)      │
# │ mixedbread-ai/mxbai-rerank-xsmall   │ 70.8M   │ 43.9       │ Moderate (7x)    │
# │ mixedbread-ai/mxbai-rerank-base     │ 200M    │ 46.9       │ Slow (20x)       │
# │ mixedbread-ai/mxbai-rerank-large    │ 560M    │ 48.8       │ Very slow (56x)  │
# └─────────────────────────────────────┴─────────┴────────────┴──────────────────┘
# * MiniLM excels on MS MARCO (74.3 NDCG) but underperforms on diverse BEIR domains
#
# Trade-off: MiniLM is faster but trained only on web search queries.
#            mxbai trained on diverse data (scientific, financial, etc.) - better for RAGLab.
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
RERANK_INITIAL_K = 50  # Retrieve more candidates than final top_k for reranking


# ============================================================================
# EVALUATION SETTINGS (RAGAS)
# ============================================================================

# Model selection: DeepSeek V3.2 for better reasoning
# GPT-5 Nano produced empty answers for cross-domain questions
# DeepSeek V3.2: $0.14/1M input - good balance of cost and capability
EVAL_GENERATION_MODEL = "deepseek/deepseek-v3.2"

# Evaluation model: Claude Haiku 4.5 for reliable structured output
# Claude 3 Haiku was returning prose instead of JSON, causing OutputParserException
# Claude Haiku 4.5 supports structured output with improved JSON compliance
EVAL_EVALUATION_MODEL = "anthropic/claude-haiku-4.5"

# Test questions file location
EVAL_TEST_QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation" / "questions" / "test_questions.json"

# Results output directory
EVAL_RESULTS_DIR = DATA_DIR / "evaluation" / "ragas_results"

# Default RAGAS metrics for evaluation
# Retrieval: context_precision, context_recall (requires reference)
# Generation: faithfulness, relevancy
# End-to-end: answer_correctness (requires reference)
EVAL_DEFAULT_METRICS = [
    "faithfulness",        # Generation: Is answer grounded in context?
    "relevancy",           # Generation: Does answer address question?
    "context_precision",   # Retrieval: Are retrieved chunks relevant?
    "context_recall",      # Retrieval: Did we get all needed info? (requires reference)
    "answer_correctness",  # End-to-end: Is answer factually correct? (requires reference)
]

# Reduced metrics for GraphRAG evaluation.
# GraphRAG returns community summaries (not document chunks) for global queries,
# making context_precision/recall meaningless. Faithfulness is also misleading
# because DRIFT synthesizes answers across multiple LLM rounds.
EVAL_GRAPHRAG_METRICS = [
    "relevancy",           # Generation: Does answer address question?
    "answer_correctness",  # End-to-end: Is answer factually correct? (requires reference)
]

# Trace output directory for evaluation runs
EVAL_TRACES_DIR = DATA_DIR / "evaluation" / "traces"

# Log output directory for comprehensive evaluation runs
EVAL_LOGS_DIR = DATA_DIR / "evaluation" / "logs"


# ============================================================================
# QUERY PREPROCESSING SETTINGS
# ============================================================================

# Model for query preprocessing (hyde, decomposition)
# DeepSeek V3.2: $0.14/1M input - strong reasoning at low cost
PREPROCESSING_MODEL = "deepseek/deepseek-v3.2"

# Fallback models for preprocessing (used if dynamic fetch fails)
# These are updated manually when OpenRouter availability changes
AVAILABLE_PREPROCESSING_MODELS = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("openai/gpt-5-mini", "Value: GPT-5 Mini"),
    ("google/gemini-3-flash-preview", "Quality: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Premium: Claude Haiku 4.5"),
]


# ============================================================================
# ANSWER GENERATION SETTINGS
# ============================================================================

# Default model for answer generation
# GPT-5-mini: improved reasoning for answer synthesis
GENERATION_MODEL = "openai/gpt-5-mini"

# Fallback models for generation (used if dynamic fetch fails)
AVAILABLE_GENERATION_MODELS = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("openai/gpt-5-mini", "Value: GPT-5 Mini"),
    ("google/gemini-3-flash-preview", "Quality: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Premium: Claude Haiku 4.5"),
]

# Enable/disable answer generation globally (can be overridden in UI)
ENABLE_ANSWER_GENERATION = True

# Enable/disable query preprocessing (strategy-based transformation)
ENABLE_QUERY_PREPROCESSING = True


# ============================================================================
# SEARCH TYPE SETTINGS
# ============================================================================

# Available search types (how chunks are retrieved from Weaviate)
# This is orthogonal to preprocessing strategies (which transform the query)
# Format: (search_type_id, display_label, description)
AVAILABLE_SEARCH_TYPES = [
    ("keyword", "Keyword (BM25)", "Pure BM25 keyword search, no embeddings"),
    ("hybrid", "Hybrid", "Combines vector similarity with BM25 keyword matching"),
]

# Default search type for evaluation
DEFAULT_SEARCH_TYPE = "hybrid"


# ============================================================================
# PREPROCESSING STRATEGY SETTINGS
# ============================================================================

# Available preprocessing strategies (query transformation before search)
# Note: These are ORTHOGONAL to search_type. Any strategy works with any search_type.
# Format: (strategy_id, display_label, description)
AVAILABLE_PREPROCESSING_STRATEGIES = [
    ("none", "None", "No preprocessing, use original query"),
    ("hyde", "HyDE", "Generate hypothetical answer for semantic matching (arXiv:2212.10496)"),
    ("decomposition", "Decomposition", "Break into sub-questions + union merge + rerank (arXiv:2507.00355)"),
    ("graphrag", "GraphRAG", "Pure graph retrieval with combined_degree ranking (arXiv:2404.16130)"),
]

# Default strategy for UI and preprocess_query() when not specified
DEFAULT_PREPROCESSING_STRATEGY = "hyde"

# Preprocessing compatibility by collection type for standard strategies
# Note: GraphRAG uses its own dedicated collection (semantic_std2) via StrategyConfig,
# so it's not included here. The StrategyConfig.collection_constraint handles it.
PREPROCESSING_COMPATIBILITY = {
    "section": ["none", "hyde", "decomposition"],
    "contextual": ["none", "hyde", "decomposition"],
    "semantic": ["none", "hyde", "decomposition"],
    "raptor": ["none", "hyde", "decomposition"],
}


def get_valid_preprocessing_strategies(collection_strategy: str) -> list:
    """Return valid preprocessing strategies for a collection type.

    GraphRAG requires matching chunk IDs between extraction and search.
    Only section and contextual collections have compatible IDs.

    Args:
        collection_strategy: The chunking strategy name (e.g., "section", "semantic_0.5").

    Returns:
        List of valid preprocessing strategy IDs.
    """
    # Handle semantic variants like "semantic_0.5"
    base_strategy = (
        collection_strategy.split("_")[0]
        if "_" in collection_strategy
        else collection_strategy
    )
    return PREPROCESSING_COMPATIBILITY.get(
        base_strategy, ["none", "hyde", "decomposition"]
    )


def list_search_types() -> list:
    """List all available search type IDs.

    Returns:
        List of search type IDs (e.g., ["keyword", "hybrid"]).
    """
    return [st[0] for st in AVAILABLE_SEARCH_TYPES]


# ============================================================================
# CHUNKING STRATEGY SETTINGS
# ============================================================================

# Available chunking strategies
# Format: (strategy_id, display_label, description)
AVAILABLE_CHUNKING_STRATEGIES = [
    ("section", "Section (Baseline)", "Sequential with sentence overlap, respects markdown sections"),
    ("semantic", "Semantic", "Embedding similarity-based boundaries for topic coherence"),
    ("contextual", "Contextual", "LLM-generated chunk context (Anthropic-style, +35% improvement)"),
    ("raptor", "RAPTOR", "Hierarchical summarization tree (+20% comprehension, arXiv:2401.18059)"),
]

# Default strategy for CLI when not specified
DEFAULT_CHUNKING_STRATEGY = "section"

# Semantic chunking parameters
# Standard deviation coefficient for detecting topic shifts
# Breakpoint occurs where: similarity < mean - (coefficient * std)
#
# Higher values = fewer splits (only extreme outliers trigger breakpoints)
# Lower values = more splits (more sensitive to similarity drops)
#
# Tuning notes:
# - 3.0: Default, conservative (statistically significant drops only)
# - 2.0: More sensitive, smaller chunks
# - 1.5: Aggressive splitting
# Note: EMBEDDING_MAX_INPUT_TOKENS (8191) is safeguard only; semantic boundaries drive chunking
SEMANTIC_STD_COEFFICIENT = 3.0  # Standard deviations below mean for breakpoint

# Contextual chunking parameters (Anthropic-style)
# Model for generating contextual snippets
# DeepSeek V3.2: $0.14/1M input - good balance of cost and capability
CONTEXTUAL_MODEL = "deepseek/deepseek-v3.2"

# Maximum tokens for the contextual snippet (output limit)
# Increased from 100 to allow room for complete sentences
CONTEXTUAL_MAX_SNIPPET_TOKENS = 150

# CONTEXTUAL_PROMPT is imported from src/prompts.py (see bottom of file)


def get_semantic_folder_name(std_coefficient: float = SEMANTIC_STD_COEFFICIENT) -> str:
    """Generate semantic chunking folder name with std coefficient.

    Creates folder names like 'semantic_std3' or 'semantic_std2.5' to distinguish
    outputs from different coefficient configurations.

    Args:
        std_coefficient: Standard deviation coefficient for breakpoint detection.

    Returns:
        Folder name in format 'semantic_std{coefficient}'.

    Example:
        >>> get_semantic_folder_name(3.0)
        'semantic_std3'
        >>> get_semantic_folder_name(2.5)
        'semantic_std2.5'
    """
    # Format coefficient: remove trailing zeros (3.00 -> 3, 2.50 -> 2.5)
    coef_str = f"{std_coefficient:.2f}".rstrip("0").rstrip(".")
    return f"semantic_std{coef_str}"


# ============================================================================
# STRATEGY METADATA REGISTRY
# ============================================================================
# Central registry for chunking strategy metadata, used for:
# - UI display (descriptions, labels)
# - Strategy-scoped embedding paths
# - Collection discovery and enrichment

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyMetadata:
    """Metadata for a chunking strategy.

    Attributes:
        key: Strategy identifier (e.g., "section", "semantic_0.5").
        display_name: Human-readable name for UI display.
        description: Short description of the strategy's approach.
    """
    key: str
    display_name: str
    description: str


# Registry of known chunking strategies with their metadata
STRATEGY_REGISTRY: dict[str, StrategyMetadata] = {
    "section": StrategyMetadata(
        key="section",
        display_name="Section-Based Chunking",
        description="Preserves document structure with sentence overlap",
    ),
    "contextual": StrategyMetadata(
        key="contextual",
        display_name="Contextual Chunking",
        description="LLM-generated context prepended (+35% improvement)",
    ),
    "raptor": StrategyMetadata(
        key="raptor",
        display_name="RAPTOR (Hierarchical)",
        description="Multi-level summary tree for theme + detail retrieval (+20%)",
    ),
    # Semantic strategies are generated dynamically based on threshold
}


def get_strategy_metadata(strategy: str) -> StrategyMetadata:
    """Get metadata for a chunking strategy.

    Args:
        strategy: Strategy key (e.g., "section", "semantic_0.5", "contextual").

    Returns:
        StrategyMetadata with display name and description.
        For unknown strategies, generates a generic fallback.

    Example:
        >>> get_strategy_metadata("section")
        StrategyMetadata(key='section', display_name='Section-Based Chunking', ...)
        >>> get_strategy_metadata("semantic_0.5")
        StrategyMetadata(key='semantic_0.5', display_name='Semantic Chunking (0.5)', ...)
    """
    # Check registry first
    if strategy in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy]

    # Handle semantic variants dynamically
    if strategy.startswith("semantic_"):
        suffix = strategy.split("_", 1)[1]
        # Handle stdX format (e.g., "semantic_std2" -> "2 std dev")
        if suffix.startswith("std") and suffix[3:].isdigit():
            std_val = suffix[3:]
            return StrategyMetadata(
                key=strategy,
                display_name=f"Semantic ({std_val} std)",
                description=f"Embedding similarity boundaries ({std_val} standard deviations)",
            )
        # Handle numeric threshold (e.g., "semantic_0.5")
        return StrategyMetadata(
            key=strategy,
            display_name=f"Semantic ({suffix})",
            description=f"Embedding similarity boundaries (threshold: {suffix})",
        )

    # Fallback for unknown strategies
    return StrategyMetadata(
        key=strategy,
        display_name=strategy.replace("_", " ").title(),
        description="Custom chunking strategy",
    )


def get_embedding_folder_path(strategy: str) -> Path:
    """Get strategy-scoped embedding folder path.

    Creates isolated embedding storage per strategy, enabling A/B testing
    of different chunking approaches without data overwrites.

    Args:
        strategy: Strategy key (e.g., "section", "semantic_0.5", "contextual").

    Returns:
        Path to embedding folder: data/processed/06_embeddings/{strategy}/

    Example:
        >>> get_embedding_folder_path("section")
        PosixPath('.../data/processed/06_embeddings/section')
        >>> get_embedding_folder_path("semantic_0.5")
        PosixPath('.../data/processed/06_embeddings/semantic_0.5')
    """
    # Sanitize strategy to prevent path traversal
    # Replace all path separators and multiple dots with underscores
    safe_strategy = re.sub(r'[/\\]+', '_', strategy)  # Replace path separators
    safe_strategy = re.sub(r'\.{2,}', '_', safe_strategy)  # Replace multiple dots
    return DIR_EMBEDDINGS / safe_strategy


# ============================================================================
# RAPTOR SETTINGS (Hierarchical Summarization)
# ============================================================================
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
# Paper: arXiv:2401.18059 (ICLR 2024)
# Builds a hierarchical tree of summaries enabling multi-level retrieval.

# Model for generating cluster summaries (reuse contextual model for consistency)
RAPTOR_SUMMARY_MODEL = CONTEXTUAL_MODEL  # -> deepseek/deepseek-v3.2

# Tree building constraints
RAPTOR_MAX_LEVELS = 4  # Maximum tree depth (0=leaves, 1-4=summaries)
RAPTOR_MIN_CLUSTER_SIZE = 3  # Minimum nodes required to attempt clustering

# UMAP dimensionality reduction parameters (from paper)
RAPTOR_UMAP_N_NEIGHBORS = 10  # Balance local/global structure
RAPTOR_UMAP_N_COMPONENTS = 10  # Target dimensions for GMM

# GMM clustering parameters
RAPTOR_MIN_CLUSTERS = 2  # Minimum K for BIC search
RAPTOR_MAX_CLUSTERS = 50  # Maximum K for BIC search
RAPTOR_CLUSTER_PROBABILITY_THRESHOLD = 0.3  # Soft assignment threshold

# Summarization parameters
# Increased from 150 to allow room for complete sentences
RAPTOR_MAX_SUMMARY_TOKENS = 200  # Output limit (paper avg: 131, buffer for sentence completion)
RAPTOR_MAX_CONTEXT_TOKENS = 8000  # Input context limit for LLM

# RAPTOR_SUMMARY_PROMPT is imported from src/prompts.py (see bottom of file)


# ============================================================================
# GRAPHRAG SETTINGS (Knowledge Graph + Leiden Communities)
# ============================================================================
# GraphRAG: Graph Retrieval-Augmented Generation (Microsoft Research)
# Paper: arXiv:2404.16130 (Apr 2024)
# Builds a knowledge graph of entities/relationships, detects communities
# via Leiden algorithm, and generates community summaries for global queries.

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "raglab_graphrag")

# Model for entity/relationship extraction
# Claude Haiku 4.5 for reliable structured output (JSON Schema)
# GPT-4o-mini produced 400 errors with structured output
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-haiku-4.5"

# Model for community summarization (same as extraction for consistency)
GRAPHRAG_SUMMARY_MODEL = CONTEXTUAL_MODEL  # -> deepseek/deepseek-v3.2

# Entity extraction parameters
GRAPHRAG_MAX_EXTRACTION_TOKENS = 8000  # Max tokens for extraction response
GRAPHRAG_MAX_ENTITIES = 10             # Max entities per chunk (reduced from 15 to prevent truncation)
GRAPHRAG_MAX_RELATIONSHIPS = 7         # Max relationships per chunk (reduced from 10 to prevent truncation)

# Gleaning: multi-pass extraction for improved recall (Microsoft GraphRAG paper)
# Each gleaning round asks the LLM to find entities missed in previous passes
# Set to 0 to disable gleaning (faster but lower recall)
GRAPHRAG_MAX_GLEANINGS = 1  # Number of additional extraction passes (0 = disabled)

# Entity types are now defined in src/graph/graphrag_types.yaml
# See: from src.graph.graphrag_types import get_entity_types

# Leiden community detection parameters
GRAPHRAG_LEIDEN_RESOLUTION = 1.0    # Higher = more, smaller communities
GRAPHRAG_LEIDEN_MAX_LEVELS = 10     # Maximum hierarchy depth
GRAPHRAG_MIN_COMMUNITY_SIZE = 3     # Minimum nodes per community
GRAPHRAG_LEIDEN_SEED = 42           # Fixed seed for deterministic results
GRAPHRAG_LEIDEN_CONCURRENCY = 1     # Single-threaded for reproducibility

# Community summarization parameters
# Microsoft GraphRAG (arXiv:2404.16130) does not cap summary output tokens
# GRAPHRAG_MAX_SUMMARY_TOKENS removed - allow complete summaries
GRAPHRAG_MAX_CONTEXT_TOKENS = 8000  # Max input tokens for summarization (matches Microsoft)

# GraphRAG prompts are imported from src/prompts.py (see bottom of file)

# Graph retrieval parameters
GRAPHRAG_TOP_COMMUNITIES = 3        # Number of communities to retrieve
GRAPHRAG_TRAVERSE_DEPTH = 2         # Hops for entity traversal

# Hierarchical community parameters (Microsoft GraphRAG paper)
# Level 0 (C0) = coarsest granularity (corpus themes), higher levels = finer
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3   # Number of levels: C0, C1, C2

# PageRank centrality parameters
GRAPHRAG_PAGERANK_DAMPING = 0.85    # Standard damping factor
GRAPHRAG_PAGERANK_ITERATIONS = 20   # Max iterations for convergence

# DRIFT Search parameters (replaces map-reduce for global queries)
# Simplified DRIFT: top-K community selection + primer + reduce (no follow-ups)
# Matches Microsoft defaults for K and folds, tuned for 19-book corpus (1000+ communities)
DRIFT_TOP_K_COMMUNITIES = 20     # Top-K communities by semantic similarity (Microsoft default: 20)
DRIFT_PRIMER_FOLDS = 4           # Parallel LLM calls in primer phase (5 communities per fold)
DRIFT_PRIMER_MAX_TOKENS = 1000   # Max tokens per primer intermediate answer
DRIFT_REDUCE_MAX_TOKENS = 800    # Max tokens for final synthesis

# Embedding-based entity extraction (Microsoft GraphRAG reference implementation)
GRAPHRAG_ENTITY_EXTRACTION_TOP_K = 10   # Max entities from embedding search
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3    # Minimum cosine similarity threshold

# Strict mode: discard entities with types not in graphrag_types.yaml
# Follows LangChain's approach to prevent graph fragmentation from type drift
GRAPHRAG_STRICT_MODE = True

# Recommended chunking for GraphRAG: semantic with std=2.0
# Lower std coefficient = more breakpoints = smaller, more cohesive chunks
# Better for entity extraction (each chunk covers fewer topics)
GRAPHRAG_SEMANTIC_STD_COEFFICIENT = 2.0

# GraphRAG uses semantic_std2 chunks exclusively for consistency between
# indexing (entity extraction) and query (chunk retrieval)
GRAPHRAG_CHUNKING_STRATEGY = "semantic_std2"


def get_entity_collection_name() -> str:
    """Generate entity collection name for GraphRAG embedding extraction.

    Uses GRAPHRAG_CHUNKING_STRATEGY for consistent entity-chunk mapping.

    Returns:
        Collection name like "Entity_semantic_std2_v1".
    """
    strategy_safe = GRAPHRAG_CHUNKING_STRATEGY.replace(".", "_")
    return f"Entity_{strategy_safe}_{COLLECTION_VERSION}"


def get_graphrag_chunk_collection_name() -> str:
    """Generate chunk collection name for GraphRAG retrieval.

    This is the collection used at query time to fetch chunks by ID after
    graph traversal. Must match the collection created during indexing.

    Returns:
        Collection name like "RAG_semantic_std2_embed3large_v1".
    """
    strategy_safe = GRAPHRAG_CHUNKING_STRATEGY.replace(".", "_")
    return f"RAG_{strategy_safe}_{EMBEDDING_MODEL_SHORT}_{COLLECTION_VERSION}"


# Output directory for graph data (under chunks since graph derives from chunks)
DIR_GRAPH_DATA = DIR_FINAL_CHUNKS / "graph"

# =============================================================================
# PROMPTS - Imported from prompts.py
# =============================================================================
# All LLM prompts are centralized in src/prompts.py for maintainability.
# Import them here for backward compatibility with existing code.

from src.prompts import (
    # Query preprocessing (HyDE split prompts for dual-domain corpus)
    HYDE_PROMPT_NEUROSCIENCE,
    HYDE_PROMPT_PHILOSOPHY,
    DECOMPOSITION_PROMPT,
    # Answer generation
    GENERATION_SYSTEM_PROMPT,
    # Contextual chunking
    CONTEXTUAL_PROMPT,
    # RAPTOR
    RAPTOR_SUMMARY_PROMPT,
    # GraphRAG
    GRAPHRAG_COMMUNITY_PROMPT,
    GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT,
    GRAPHRAG_CHUNK_EXTRACTION_PROMPT,
    # GraphRAG consolidation (Microsoft GraphRAG approach)
    GRAPHRAG_ENTITY_SUMMARIZE_PROMPT,
    GRAPHRAG_RELATIONSHIP_SUMMARIZE_PROMPT,
    # GraphRAG gleaning (multi-pass extraction)
    GRAPHRAG_LOOP_PROMPT,
    GRAPHRAG_CONTINUE_PROMPT,
)

# Number of hypothetical documents to generate for HyDE (total)
# Split evenly: 2 neuroscience + 2 philosophy for dual-domain corpus
HYDE_K = 4

# Max tokens per hypothetical passage
# Paper uses 100-150 tokens; prompt instructs "brief 2-3 sentences" for modern models
HYDE_MAX_TOKENS = 150