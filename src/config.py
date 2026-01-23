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


def get_community_collection_name(chunking_strategy: str = None) -> str:
    """
    Generate community collection name for GraphRAG.

    Community embeddings are stored in Weaviate for efficient vector search.
    Collection name includes strategy to match the chunks they were derived from.

    Args:
        chunking_strategy: Chunking strategy name (e.g., "section").
            If None, uses CHUNKING_STRATEGY_NAME from config.

    Returns:
        Collection name in format: Community_{strategy}_{version}

    Example:
        >>> get_community_collection_name()
        "Community_section800_v1"
        >>> get_community_collection_name("semantic")
        "Community_semantic_v1"
    """
    strategy = chunking_strategy if chunking_strategy else CHUNKING_STRATEGY_NAME
    strategy_safe = strategy.replace(".", "_")
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
# Model Comparison (see docs/preprocessing/reranking.md for full analysis):
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

# Evaluation model: GPT-4o-mini with JSON mode for reliable structured output
# Claude 3 Haiku was returning prose instead of JSON, causing OutputParserException
# GPT-4o-mini supports response_format: json_object for guaranteed JSON compliance
EVAL_EVALUATION_MODEL = "openai/gpt-4o-mini"

# Test questions file location
EVAL_TEST_QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "test_questions.json"

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

# Trace output directory for evaluation runs
EVAL_TRACES_DIR = DATA_DIR / "evaluation" / "traces"

# Log output directory for comprehensive evaluation runs
EVAL_LOGS_DIR = DATA_DIR / "evaluation" / "logs"


# ============================================================================
# QUERY PREPROCESSING SETTINGS
# ============================================================================

# Model for query preprocessing (hyde, decomposition)
# GPT-4o-mini: HyDE requires domain knowledge - nano models return empty responses
PREPROCESSING_MODEL = "openai/gpt-4o-mini"

# Fallback models for preprocessing (used if dynamic fetch fails)
# These are updated manually when OpenRouter availability changes
AVAILABLE_PREPROCESSING_MODELS = [
    ("openai/gpt-4o-mini", "Value: GPT-4o Mini"),
    ("deepseek/deepseek-v3.2", "Value: DeepSeek V3.2"),
    ("google/gemini-3-flash-preview", "Quality: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Premium: Claude Haiku 4.5"),
]


# ============================================================================
# ANSWER GENERATION SETTINGS
# ============================================================================

# Default model for answer generation
# GPT-4o-mini: $0.15/$0.60 per 1M tokens - consistent with eval/preprocessing
GENERATION_MODEL = "openai/gpt-4o-mini"

# Fallback models for generation (used if dynamic fetch fails)
AVAILABLE_GENERATION_MODELS = [
    ("openai/gpt-4o-mini", "Budget: GPT-4o-mini"),
    ("deepseek/deepseek-v3.2", "Value: DeepSeek V3.2"),
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
    ("decomposition", "Decomposition", "Break into sub-questions + RRF merge (arXiv:2507.00355)"),
    ("graphrag", "GraphRAG", "Hybrid graph + vector retrieval via RRF (arXiv:2404.16130)"),
]

# Default strategy for UI and preprocess_query() when not specified
DEFAULT_PREPROCESSING_STRATEGY = "hyde"

# Preprocessing compatibility by collection type
# GraphRAG requires chunk IDs to match between extraction and search.
# Only section collections have compatible IDs because:
# - section: Original chunk IDs used for entity extraction
# - contextual: Uses semantic_std2 as input, so has semantic chunk IDs (no match)
# - semantic: Different chunk boundaries = different IDs = no match
# - raptor: Summary nodes have different IDs; only leaf chunks match
GRAPHRAG_COMPATIBLE_COLLECTIONS = ["section"]

PREPROCESSING_COMPATIBILITY = {
    "section": ["none", "hyde", "decomposition", "graphrag"],
    "contextual": ["none", "hyde", "decomposition"],  # No graphrag - uses semantic chunk IDs
    "semantic": ["none", "hyde", "decomposition"],  # No graphrag - chunk ID mismatch
    "raptor": ["none", "hyde", "decomposition"],  # No graphrag - partial match only
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
# GPT-4o-mini: $0.15/$0.60 per 1M tokens - consistent across all tasks
CONTEXTUAL_MODEL = "openai/gpt-4o-mini"

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

    # Handle semantic_X.X variants dynamically
    if strategy.startswith("semantic_"):
        threshold = strategy.split("_", 1)[1]
        return StrategyMetadata(
            key=strategy,
            display_name=f"Semantic Chunking ({threshold})",
            description=f"Embedding similarity boundaries (threshold: {threshold})",
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
RAPTOR_SUMMARY_MODEL = CONTEXTUAL_MODEL  # -> openai/gpt-4o-mini

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
# Claude 3 Haiku for reliable structured output (JSON Schema)
# GPT-4o-mini produced 400 errors with structured output
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-3-haiku"

# Model for community summarization (same as extraction for consistency)
GRAPHRAG_SUMMARY_MODEL = CONTEXTUAL_MODEL  # -> openai/gpt-4o-mini

# Entity extraction parameters
GRAPHRAG_MAX_EXTRACTION_TOKENS = 4000  # Max tokens for extraction response
GRAPHRAG_MAX_ENTITIES = 10             # Max entities per chunk (reduced from 15 to prevent truncation)
GRAPHRAG_MAX_RELATIONSHIPS = 7         # Max relationships per chunk (reduced from 10 to prevent truncation)

# Domain-specific entity types for neuroscience/philosophy corpus
# These guide the LLM to extract relevant entities
GRAPHRAG_ENTITY_TYPES = [
    # Neuroscience entities
    "BRAIN_REGION",          # Prefrontal cortex, amygdala, hippocampus
    "NEUROTRANSMITTER",      # Dopamine, serotonin, cortisol
    "NEURAL_PROCESS",        # Synaptic plasticity, long-term potentiation
    "COGNITIVE_FUNCTION",    # Working memory, decision-making, attention
    "BEHAVIOR",              # Aggression, altruism, stress response
    # Philosophy entities
    "PHILOSOPHER",           # Marcus Aurelius, Schopenhauer, Confucius
    "PHILOSOPHICAL_CONCEPT", # Virtue ethics, will, Tao, Stoic acceptance
    "PHILOSOPHICAL_SCHOOL",  # Stoicism, Taoism, German Pessimism
    "TEXT_OR_WORK",          # Meditations, The Art of Worldly Wisdom
    # Research entities
    "RESEARCHER",            # Sapolsky, Kahneman, Tversky
    "STUDY_OR_EXPERIMENT",   # Stanford prison experiment, marshmallow test
    "COGNITIVE_BIAS",        # Confirmation bias, loss aversion
    # General entities
    "PERSON",                # Historical figures, case study subjects
    "ORGANIZATION",          # Universities, research institutions
    "BOOK_OR_CHAPTER",       # Source document references
]

# Relationship types for knowledge graph edges
GRAPHRAG_RELATIONSHIP_TYPES = [
    # Causal/mechanistic relationships
    "CAUSES",                # A causes B
    "INHIBITS",              # A inhibits/blocks B
    "MODULATES",             # A modulates/affects B
    "REGULATES",             # A regulates B
    # Associative relationships
    "ASSOCIATED_WITH",       # A is associated with B
    "PART_OF",               # A is part of B
    "LOCATED_IN",            # A is located in B
    # Philosophical relationships
    "PROPOSES",              # Philosopher proposes concept
    "INFLUENCES",            # A influences B
    "CONTRADICTS",           # A contradicts B
    "BUILDS_ON",             # A builds on B
    "ADVOCATES_FOR",         # A advocates for B
    # Research relationships
    "STUDIES",               # Researcher studies phenomenon
    "DEMONSTRATES",          # Study demonstrates finding
    "CITES",                 # A cites B
    # Attribution relationships
    "AUTHORED_BY",           # Work authored by person
    "AFFILIATED_WITH",       # Person affiliated with organization
]

# Leiden community detection parameters
GRAPHRAG_LEIDEN_RESOLUTION = 1.0    # Higher = more, smaller communities
GRAPHRAG_LEIDEN_MAX_LEVELS = 10     # Maximum hierarchy depth
GRAPHRAG_MIN_COMMUNITY_SIZE = 3     # Minimum nodes per community
GRAPHRAG_LEIDEN_SEED = 42           # Fixed seed for deterministic results
GRAPHRAG_LEIDEN_CONCURRENCY = 1     # Single-threaded for reproducibility

# Community summarization parameters
# Increased from 200 to allow room for complete sentences
GRAPHRAG_MAX_SUMMARY_TOKENS = 300   # Max tokens per community summary (buffer for sentence completion)
GRAPHRAG_MAX_CONTEXT_TOKENS = 6000  # Max input tokens for summarization

# GraphRAG prompts are imported from src/prompts.py (see bottom of file):
# - GRAPHRAG_COMMUNITY_PROMPT
# - GRAPHRAG_QUERY_EXTRACTION_PROMPT

# Graph retrieval parameters
GRAPHRAG_TOP_COMMUNITIES = 3        # Number of communities to retrieve
GRAPHRAG_TRAVERSE_DEPTH = 2         # Hops for entity traversal
GRAPHRAG_RRF_K = 60                 # RRF constant for score fusion

# Hierarchical community parameters (Microsoft GraphRAG paper)
# Level 0 (C0) = finest granularity, Level 2 (C2) = coarsest (corpus themes)
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3   # Number of levels: C0, C1, C2

# PageRank centrality parameters
GRAPHRAG_PAGERANK_DAMPING = 0.85    # Standard damping factor
GRAPHRAG_PAGERANK_ITERATIONS = 20   # Max iterations for convergence

# Map-reduce for global queries (Microsoft GraphRAG paper)
GRAPHRAG_MAP_REDUCE_TOP_K = 5       # Communities for map-reduce
GRAPHRAG_MAP_MAX_TOKENS = 300       # Max tokens per map response
GRAPHRAG_REDUCE_MAX_TOKENS = 500    # Max tokens for reduce response

# Embedding-based entity extraction (Microsoft GraphRAG reference implementation)
# Replaces LLM extraction with embedding similarity for faster queries
GRAPHRAG_ENTITY_EXTRACTION_TOP_K = 10   # Max entities from embedding search
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3    # Minimum cosine similarity threshold
GRAPHRAG_USE_EMBEDDING_EXTRACTION = True  # Use embedding-based extraction (fallback to LLM)


def get_entity_collection_name() -> str:
    """Generate entity collection name for GraphRAG embedding extraction.

    Uses same naming pattern as community collection for consistency.

    Returns:
        Collection name like "Entity_section800_v1".
    """
    strategy_safe = CHUNKING_STRATEGY_NAME.replace(".", "_")
    return f"Entity_{strategy_safe}_{COLLECTION_VERSION}"


# Output directory for graph data (under chunks since graph derives from chunks)
DIR_GRAPH_DATA = DIR_FINAL_CHUNKS / "graph"

# =============================================================================
# CORPUS MAPPING FOR STRATIFIED ENTITY TYPE CONSOLIDATION
# =============================================================================
# Maps book filenames (stems) to corpus types for balanced entity type discovery.
# Used by auto_tuning.py to select top entity types proportionally from each corpus,
# preventing larger corpora from dominating the final entity type taxonomy.

CORPUS_BOOK_MAPPING = {
    # Neuroscience books (8 books, ~4440 chunks, ~48.6K entities)
    "neuroscience": [
        "Cognitive Biology , Evolutionary and Developmental Perspectives on Mind Brain and Behavior (Luca Tommasi, Mary A.Peterson, Lynn Nadel)",
        "Determined, a science of life without free will (Robert M. Sapolsky)",
        "Cognitive Neuroscience, The Biology of the Mind (Michael Gazzaniga)",
        "Biopsychology (John Pinel, Steven Barnes)",
        "Fundamentals of Cognitive Neuroscience,  A_Beginners Guide(Nicole M. Gage Bernard)",
        "Behave, The_Biology of Humans at Our Best Worst (Robert M. Sapolsky)",
        "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)",
        "Psychobiology of Behaviour (Konstanthos N,Fountoulakis, Loannis Nimatoudis)",
    ],
    # Philosophy books (11 books, ~1809 chunks, ~15.6K entities)
    "philosophy": [
        "The Enchiridion (Epictetus)",
        "Thinking Fast and Slow (Daniel Kahneman)",
        "Letters from a Stoic (Seneca)",
        "The Meditations (Marcus Aurelius)",
        "Wisdom of Life (Schopenhauer)",
        "The essays, counsels and maxims (Arthur Schopenhauer)",
        "Essays and Aphorisms (Arthur Schopenhauer)",
        "The Art of Living ,The Classical Manual on Virtue Happiness and Effectiveness (Epictetus)",
        "Tao te ching Lao_tzu (Lao Tzu)",
        "The Pocket Oracle and Art of Prudence (Baltasar Gracian)",
        "The Analects Conclusions and Conversations (Confucius)",
    ],
}

# Stratified consolidation parameters
GRAPHRAG_TYPES_PER_CORPUS = 12      # Top entity types to select from each corpus
GRAPHRAG_MIN_CORPUS_PERCENTAGE = 1.0  # Minimum % within corpus to be considered (filters noise)


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
    GRAPHRAG_QUERY_EXTRACTION_PROMPT,
    GRAPHRAG_COMMUNITY_PROMPT,
    # Auto-tuning
    GRAPHRAG_OPEN_EXTRACTION_PROMPT,
    GRAPHRAG_GLOBAL_CONSOLIDATION_PROMPT,
    GRAPHRAG_STRATIFIED_CONSOLIDATION_PROMPT,
)

# Number of hypothetical documents to generate for HyDE (total)
# Split evenly: 2 neuroscience + 2 philosophy for dual-domain corpus
HYDE_K = 4

# Max tokens per hypothetical passage
# Paper uses ~100-150 tokens, but RAGLab's dense corpus needs more for complete thoughts
HYDE_MAX_TOKENS = 250