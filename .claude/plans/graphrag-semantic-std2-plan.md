# Plan: GraphRAG Uses semantic_std2 Exclusively

## Summary

GraphRAG will use `semantic_std2` chunking exclusively, configured at indexing time. At query time, GraphRAG operates independently of the normal collection/search_type selection - it uses its own dedicated semantic_std2 index.

**Key principle**: All constraints are defined in `StrategyConfig` (single source of truth), following the same pattern as HyDE's `alpha_constraint` and `requires_reranking`.

---

## Current State (Problems)

| File | Issue |
|------|-------|
| `src/config.py:419` | `GRAPHRAG_COMPATIBLE_COLLECTIONS = ["section"]` - wrong |
| `src/config.py:421-425` | `PREPROCESSING_COMPATIBILITY` says graphrag only works with section |
| `src/config.py:757-766` | `get_entity_collection_name()` hardcodes `CHUNKING_STRATEGY_NAME` ("section800") |
| `src/config.py:751-754` | Comment says "Recommended: semantic std=2.0" but code uses section |
| `strategy_config.py:248` | `compatible_collections={"section"}` |
| `src/ui/app.py:398-401` | Forces section collection for graphrag |
| `src/stages/run_stage_4_5_graph_extract.py:66` | `--strategy` defaults to "section" |
| Documentation | Shows `Entity_section800_v1` instead of `Entity_semantic_std2_v1` |

---

## Phase 1: Extend StrategyConfig Class (Single Source of Truth)

**File: `src/rag_pipeline/retrieval/preprocessing/strategy_config.py`**

### 1a. Add new constraint classes (following AlphaConstraint pattern)

```python
@dataclass
class CollectionConstraint:
    """Defines collection requirements for a strategy.

    Attributes:
        mode: How collection is constrained:
            - "any": Strategy works with any collection
            - "dedicated": Strategy uses its own dedicated index (ignores selection)
        dedicated_collection: Collection name when mode="dedicated"
    """
    mode: Literal["any", "dedicated"]
    dedicated_collection: Optional[str] = None  # e.g., "RAG_semantic_std2_embed3large_v1"

    def __post_init__(self):
        if self.mode == "dedicated" and self.dedicated_collection is None:
            raise ValueError("dedicated mode requires dedicated_collection")

    def uses_dedicated_index(self) -> bool:
        return self.mode == "dedicated"


@dataclass
class SearchTypeConstraint:
    """Defines search type requirements for a strategy.

    Attributes:
        mode: How search type is constrained:
            - "any": Strategy works with any search type (keyword, hybrid)
            - "fixed": Strategy requires specific search type
            - "internal": Strategy performs its own retrieval (no external search)
        fixed_value: Required search type when mode="fixed"
    """
    mode: Literal["any", "fixed", "internal"]
    fixed_value: Optional[str] = None  # e.g., "hybrid"

    def __post_init__(self):
        if self.mode == "fixed" and self.fixed_value is None:
            raise ValueError("fixed mode requires fixed_value")

    def is_internal(self) -> bool:
        return self.mode == "internal"
```

### 1b. Update StrategyConfig dataclass

```python
@dataclass
class StrategyConfig:
    """Declarative configuration for a preprocessing strategy.

    This is the SINGLE SOURCE OF TRUTH for strategy constraints.
    All UI rendering, evaluation validation, and retrieval logic
    consult this configuration.

    Attributes:
        strategy_id: Unique identifier (e.g., "hyde", "graphrag").
        display_name: Human-readable name for UI.
        description: Short description for UI help text.
        alpha_constraint: How alpha (search balance) is constrained.
        collection_constraint: How collection selection is constrained.
        search_type_constraint: How search type is constrained.
        includes_original_in_embedding: Whether original query should be
            included when averaging embeddings (HyDE paper requirement).
        requires_reranking: If True, cross-encoder reranking is mandatory.
    """

    strategy_id: str
    display_name: str
    description: str
    alpha_constraint: AlphaConstraint = field(
        default_factory=lambda: AlphaConstraint(mode="any")
    )
    collection_constraint: CollectionConstraint = field(
        default_factory=lambda: CollectionConstraint(mode="any")
    )
    search_type_constraint: SearchTypeConstraint = field(
        default_factory=lambda: SearchTypeConstraint(mode="any")
    )
    includes_original_in_embedding: bool = False
    requires_reranking: bool = False

    # Convenience methods
    def uses_dedicated_index(self) -> bool:
        """Check if strategy uses dedicated index (ignores collection selection)."""
        return self.collection_constraint.uses_dedicated_index()

    def has_internal_search(self) -> bool:
        """Check if strategy performs its own retrieval."""
        return self.search_type_constraint.is_internal()
```

### 1c. Update STRATEGY_CONFIGS registry

```python
STRATEGY_CONFIGS: dict[str, StrategyConfig] = {
    "none": StrategyConfig(
        strategy_id="none",
        display_name="None",
        description="No preprocessing, use original query",
        alpha_constraint=AlphaConstraint(mode="any"),
        collection_constraint=CollectionConstraint(mode="any"),
        search_type_constraint=SearchTypeConstraint(mode="any"),
    ),
    "hyde": StrategyConfig(
        strategy_id="hyde",
        display_name="HyDE",
        description="Hypothetical Document Embeddings (arXiv:2212.10496)",
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        collection_constraint=CollectionConstraint(mode="any"),
        search_type_constraint=SearchTypeConstraint(mode="fixed", fixed_value="hybrid"),
        includes_original_in_embedding=True,
    ),
    "decomposition": StrategyConfig(
        strategy_id="decomposition",
        display_name="Decomposition",
        description="Break into sub-questions + rerank (arXiv:2507.00355)",
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        collection_constraint=CollectionConstraint(mode="any"),
        search_type_constraint=SearchTypeConstraint(mode="fixed", fixed_value="hybrid"),
        requires_reranking=True,
    ),
    "graphrag": StrategyConfig(
        strategy_id="graphrag",
        display_name="GraphRAG",
        description="Knowledge graph + community retrieval (arXiv:2404.16130)",
        # GraphRAG uses dedicated semantic_std2 index
        collection_constraint=CollectionConstraint(
            mode="dedicated",
            dedicated_collection="RAG_semantic_std2_embed3large_v1",
        ),
        # GraphRAG performs its own hybrid retrieval (graph + vector via RRF)
        search_type_constraint=SearchTypeConstraint(mode="internal"),
        # Alpha is N/A for internal search, but set to 1.0 for consistency
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
    ),
}
```

---

## Phase 2: Update config.py (Minimal Changes)

**File: `src/config.py`**

### 2a. Add GraphRAG chunking strategy constant

```python
# GraphRAG uses semantic_std2 exclusively (fixed at indexing time)
GRAPHRAG_CHUNKING_STRATEGY = "semantic_std2"
```

### 2b. Update collection name functions

```python
def get_entity_collection_name() -> str:
    """Generate entity collection name for GraphRAG.

    Uses GRAPHRAG_CHUNKING_STRATEGY (semantic_std2).
    """
    strategy_safe = GRAPHRAG_CHUNKING_STRATEGY.replace(".", "_")
    return f"Entity_{strategy_safe}_{COLLECTION_VERSION}"


def get_graphrag_chunk_collection_name() -> str:
    """Generate chunk collection name for GraphRAG retrieval.

    Used at query time for vector search over semantic_std2 chunks.
    """
    strategy_safe = GRAPHRAG_CHUNKING_STRATEGY.replace(".", "_")
    return f"RAG_{strategy_safe}_{EMBEDDING_MODEL_SHORT}_{COLLECTION_VERSION}"
```

### 2c. Remove deprecated compatibility structures

```python
# DELETE these (now handled by StrategyConfig):
# GRAPHRAG_COMPATIBLE_COLLECTIONS = ["section"]
# Remove graphrag from PREPROCESSING_COMPATIBILITY dict
```

---

## Phase 3: Update UI

**File: `src/ui/app.py`**

### 3a. Use StrategyConfig for rendering decisions

```python
from src.rag_pipeline.retrieval.preprocessing.strategy_config import get_strategy_config

strategy_config = get_strategy_config(selected_strategy)

# Collection selection
if strategy_config.uses_dedicated_index():
    st.sidebar.info(f"Uses dedicated index: {strategy_config.collection_constraint.dedicated_collection}")
    selected_collection = strategy_config.collection_constraint.dedicated_collection
else:
    # Show collection selector
    ...

# Search type / alpha
if strategy_config.has_internal_search():
    st.sidebar.info("Performs internal hybrid retrieval (no search type selection)")
    # Don't show alpha slider
elif strategy_config.search_type_constraint.mode == "fixed":
    st.sidebar.info(f"Search type: {strategy_config.search_type_constraint.fixed_value}")
    # Don't show search type selector
else:
    # Show search type selector
    ...
```

---

## Phase 4: Update Evaluation Code

**File: `src/stages/run_stage_7_evaluation.py`**

### 4a. Use StrategyConfig in comprehensive mode

```python
from src.rag_pipeline.retrieval.preprocessing.strategy_config import (
    get_strategy_config,
    list_strategy_configs,
)

# Separate strategies by constraint type
standard_strategies = []
dedicated_strategies = []

for config in list_strategy_configs():
    if config.uses_dedicated_index() or config.has_internal_search():
        dedicated_strategies.append(config)
    else:
        standard_strategies.append(config)

# Standard grid: collections × alphas × strategies
for collection in collections:
    for strategy_config in standard_strategies:
        for alpha in strategy_config.alpha_constraint.get_allowed_values():
            # Run evaluation...

# Dedicated strategies: single run each
for strategy_config in dedicated_strategies:
    collection = strategy_config.collection_constraint.dedicated_collection
    alpha = strategy_config.alpha_constraint.get_default()
    # Run evaluation once...
```

---

## Phase 5: Update Extraction Stage

**File: `src/stages/run_stage_4_5_graph_extract.py`**

```python
from src.config import GRAPHRAG_CHUNKING_STRATEGY

parser.add_argument(
    "--strategy", type=str, default=GRAPHRAG_CHUNKING_STRATEGY,
    help=f"Chunking strategy (default: {GRAPHRAG_CHUNKING_STRATEGY})",
)
```

---

## Phase 6: Update Documentation

### Files to update:
- `CLAUDE.md` - Update GraphRAG commands
- `memory-bank/graphrag/graphrag_raglab_implementation.md` - Fix collection names
- `docs/preprocessing/graphrag.md` - Update if exists

### Collection names:
- `Entity_semantic_std2_v1`
- `Community_semantic_std2_v1`
- `RAG_semantic_std2_embed3large_v1`

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/rag_pipeline/retrieval/preprocessing/strategy_config.py` | Add CollectionConstraint, SearchTypeConstraint, update StrategyConfig |
| `src/config.py` | Add GRAPHRAG_CHUNKING_STRATEGY, update collection functions, remove deprecated |
| `src/ui/app.py` | Use StrategyConfig for UI rendering decisions |
| `src/stages/run_stage_7_evaluation.py` | Use StrategyConfig for grid separation |
| `src/stages/run_stage_4_5_graph_extract.py` | Default to GRAPHRAG_CHUNKING_STRATEGY |
| `CLAUDE.md` | Update GraphRAG commands |
| `memory-bank/graphrag/graphrag_raglab_implementation.md` | Fix collection names |

---

## Verification

1. **StrategyConfig**: Verify constraints are correct
   ```python
   from src.rag_pipeline.retrieval.preprocessing.strategy_config import get_strategy_config

   config = get_strategy_config("graphrag")
   assert config.uses_dedicated_index() == True
   assert config.has_internal_search() == True
   assert config.collection_constraint.dedicated_collection == "RAG_semantic_std2_embed3large_v1"
   ```

2. **UI**: Select graphrag and verify constraints are respected

3. **Evaluation**: Run comprehensive and verify graphrag runs once
