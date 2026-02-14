"""RAGLab Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.
Features:
- Query preprocessing (HyDE, decomposition strategies)
- Hybrid/vector search with optional cross-encoder reranking
- LLM-based answer generation
- Pipeline logging with full prompt visibility

Run with:
    streamlit run src/ui/app.py

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
"""

import logging

# Suppress noisy HTTP logs from Weaviate client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st
import pandas as pd

from src.config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    AVAILABLE_PREPROCESSING_STRATEGIES,
    GENERATION_MODEL,
    PREPROCESSING_MODEL,
    get_valid_preprocessing_strategies,
)
from src.rag_pipeline.retrieval.query_strategy_config import (
    get_strategy_config,
)
from src.ui.services.search import search_chunks, get_available_collections, CollectionInfo
from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query
from src.rag_pipeline.generation.answer_generator import generate_answer
from src.graph.query import format_graph_context_for_generation


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAGLab Search",
    page_icon="books",
    layout="wide",
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit reruns the entire script on every interaction.
# Session state persists data across reruns.

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "connection_error" not in st.session_state:
    st.session_state.connection_error = None

if "preprocessed_query" not in st.session_state:
    st.session_state.preprocessed_query = None

if "generated_answer" not in st.session_state:
    st.session_state.generated_answer = None

if "rerank_data" not in st.session_state:
    st.session_state.rerank_data = None

if "rrf_data" not in st.session_state:
    st.session_state.rrf_data = None

if "graph_metadata" not in st.session_state:
    st.session_state.graph_metadata = None

if "retrieval_settings" not in st.session_state:
    st.session_state.retrieval_settings = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _format_config_summary() -> str:
    """Format compact config summary for display."""
    settings = st.session_state.retrieval_settings
    prep = st.session_state.preprocessed_query
    ans = st.session_state.generated_answer

    # Collection display name (fallback to raw name)
    coll = settings.get("collection_display_name", settings.get("collection_name", "unknown"))

    # Search type
    search = settings.get("search_type", "hybrid")
    alpha = settings.get("alpha", 0.5)
    search_str = f"{search}" if search == "vector" else f"hybrid (α={alpha})"

    # Preprocessing
    prep_str = "none"
    if prep:
        prep_str = getattr(prep, "strategy_used", "none")
        sub_q = getattr(prep, "sub_queries", None)
        if sub_q:
            prep_str += f" ({len(sub_q)} sub-q)"

    # Results count
    top_k = settings.get("top_k", 5)

    # Total time
    total_ms = 0
    if prep:
        total_ms += getattr(prep, "preprocessing_time_ms", 0)
    if ans:
        total_ms += getattr(ans, "generation_time_ms", 0)
    time_str = f"{total_ms:,.0f}ms" if total_ms else ""

    return f"{coll} | {search_str} | {prep_str} | {top_k} results | {time_str}"




def _display_chunks(chunks, show_indices=True):
    """Display chunk results with expandable details."""
    for i, chunk in enumerate(chunks, 1):
        # Extract author for cleaner display
        book_parts = chunk["book_id"].rsplit("(", 1)
        book_title = book_parts[0].strip()
        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""

        # RAPTOR summary indicator
        is_summary = chunk.get("is_summary", False)
        tree_level = chunk.get("tree_level", 0)
        summary_badge = " [SUMMARY]" if is_summary else ""

        prefix = f"[{i}] " if show_indices else ""
        with st.expander(
            f"**{prefix}**{book_title[:50]}...{summary_badge} | Score: {chunk['similarity']:.3f}",
            expanded=(i <= 3 and not st.session_state.generated_answer),
        ):
            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Similarity", f"{chunk['similarity']:.4f}")
            col2.metric("Tokens", chunk["token_count"])
            col3.markdown(f"**Author:** {author}")
            # Show tree level for RAPTOR chunks
            if tree_level > 0 or is_summary:
                col4.metric("Tree Level", tree_level)

            # Section info
            st.markdown(f"**Section:** {chunk['section']}")
            st.caption(f"Context: {chunk['context']}")

            # Main text
            st.markdown("---")
            st.markdown(chunk["text"])


def _render_preprocessing_stage(prep) -> None:
    """Render preprocessing stage details."""
    strategy = getattr(prep, 'strategy_used', 'N/A')
    model = getattr(prep, 'model', 'N/A')

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Strategy:** `{strategy}`")
    col2.markdown(f"**Model:** `{model}`")
    col3.metric("Time", f"{prep.preprocessing_time_ms:.0f}ms")

    # HyDE output - show all hypotheticals with domain labels
    generated_queries = getattr(prep, 'generated_queries', None)
    if strategy == "hyde" and generated_queries:
        # Filter to only hyde passages (exclude original query)
        hyde_passages = [q for q in generated_queries if q.get("type", "").startswith("hyde")]
        if hyde_passages:
            st.markdown(f"**Hypothetical Passages ({len(hyde_passages)}):**")
            # Group by domain for display
            neuro_count = phil_count = 0
            for q in hyde_passages:
                domain = q.get("domain", "unknown").title()
                if domain.lower() == "neuroscience":
                    neuro_count += 1
                    label = f"Neuroscience #{neuro_count}"
                else:
                    phil_count += 1
                    label = f"Philosophy #{phil_count}"
                with st.expander(label, expanded=False):
                    st.info(q.get("query", ""))

    # Decomposition output
    sub_queries = getattr(prep, 'sub_queries', None)
    if sub_queries:
        st.markdown("**Sub-Questions:**")
        for i, sq in enumerate(sub_queries, 1):
            st.markdown(f"{i}. {sq}")


def _render_retrieval_stage(settings: dict, results: list, prep) -> None:
    """Render retrieval stage details."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Type", settings.get("search_type", "N/A"))
    col2.metric("Alpha", settings.get("alpha", "N/A"))
    col3.metric("Top-K", settings.get("top_k", "N/A"))
    col4.metric("Found", len(results))

    search_q = prep.search_query if prep else st.session_state.last_query
    st.markdown("**Search Query:**")
    st.code(search_q, language="text")

    # Collapsible chunks section
    if results:
        # Score explanation based on retrieval method
        reranked = st.session_state.get("rerank_data") is not None
        if reranked:
            score_info = "Scores: cross-encoder semantic relevance (0.0-1.0+)"
        elif st.session_state.get("rrf_data") is not None:
            score_info = "Scores: RRF (Reciprocal Rank Fusion, k=60)"
        else:
            score_info = "Scores: cosine similarity (0.0-1.0)"

        with st.expander(f"Retrieved Chunks ({len(results)})", expanded=False):
            st.caption(score_info)
            _display_chunks(results)


def _render_rrf_stage(rrf, prep) -> None:
    """Render RRF merging stage details."""
    col1, col2, col3 = st.columns(3)
    num_chunks = len(rrf.query_contributions) if hasattr(rrf, 'query_contributions') and rrf.query_contributions else 0
    num_queries = len(prep.generated_queries) if prep and hasattr(prep, 'generated_queries') else 0
    col1.metric("Queries Merged", num_queries)
    col2.metric("Unique Chunks", num_chunks)
    col3.metric("Time", f"{rrf.merge_time_ms:.0f}ms")

    if hasattr(rrf, 'query_contributions') and rrf.query_contributions:
        contrib_data = [
            {"Chunk": cid[:30] + "..." if len(cid) > 30 else cid, "Found By": ", ".join(qt)}
            for cid, qt in list(rrf.query_contributions.items())[:5]
        ]
        if contrib_data:
            st.dataframe(pd.DataFrame(contrib_data), width="stretch")


def _render_graph_stage(graph_meta: dict) -> None:
    """Render GraphRAG stage details."""
    extracted = graph_meta.get("extracted_entities", [])
    matched = graph_meta.get("query_entities", [])
    communities = graph_meta.get("community_context", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Extracted", len(extracted))
    col2.metric("Matched", len(matched))
    col3.metric("Graph Chunks", graph_meta.get("graph_chunk_count", 0))
    col4.metric("Communities", len(communities))

    # Show extracted vs matched entities
    if extracted:
        st.markdown(f"**LLM Extracted:** {', '.join(extracted)}")
        if matched:
            st.markdown(f"**Found in Graph:** {', '.join(matched)}")
        else:
            st.caption("None of the extracted entities exist in the knowledge graph.")
    else:
        st.caption("LLM did not extract any entities from the query.")

    # Show community summaries (full text in expanders)
    if communities:
        st.markdown("**Relevant Communities:**")
        for i, comm in enumerate(communities, 1):
            with st.expander(f"Community {i}: {comm['summary'][:80]}...", expanded=(i == 1)):
                st.markdown(comm['summary'])


def _render_drift_stage(graph_meta: dict) -> None:
    """Render DRIFT global search stage details."""
    dr = graph_meta.get("drift_result", {})
    communities = dr.get("communities_used", [])
    intermediates = dr.get("intermediate_answers", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Communities", len(communities))
    col2.metric("LLM Calls", dr.get("total_llm_calls", 0))
    col3.metric("Primer", f"{dr.get('primer_time_ms', 0):.0f}ms")
    col4.metric("Reduce", f"{dr.get('reduce_time_ms', 0):.0f}ms")

    if intermediates:
        st.markdown("**Intermediate Answers (Primer Folds):**")
        for i, answer in enumerate(intermediates, 1):
            preview = answer[:80] + "..." if len(answer) > 80 else answer
            with st.expander(f"Fold {i}: {preview}", expanded=False):
                st.markdown(answer)


def _render_rerank_stage(rerank) -> None:
    """Render reranking stage details."""
    col1, col2 = st.columns(2)
    col1.markdown(f"**Model:** `{rerank.model}`")
    col2.metric("Time", f"{rerank.rerank_time_ms:.0f}ms")

    if rerank.order_changes:
        df = pd.DataFrame(rerank.order_changes)
        df = df[["before_rank", "after_rank", "before_score", "after_score", "text_preview"]]
        df.columns = ["Before", "After", "Old Score", "New Score", "Preview"]
        df["Old Score"] = df["Old Score"].round(3)
        df["New Score"] = df["New Score"].round(3)
        st.dataframe(df, width="stretch")


def _render_generation_stage(ans) -> None:
    """Render generation stage details."""
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Model:** `{ans.model}`")
    col2.metric("Time", f"{ans.generation_time_ms:.0f}ms")
    col3.markdown(f"**Sources:** {ans.sources_used}")

    with st.expander("Show Prompt", expanded=False):
        st.code(f"[System]\n{ans.system_prompt_used}\n\n[User]\n{ans.user_prompt_used}", language="text")


def _render_pipeline_log():
    """Render executed pipeline stages (only shows stages that were used)."""
    prep = st.session_state.preprocessed_query
    settings = st.session_state.retrieval_settings
    results = st.session_state.search_results
    rrf = st.session_state.rrf_data
    graph_meta = st.session_state.graph_metadata
    rerank = st.session_state.rerank_data
    ans = st.session_state.generated_answer

    # Preprocessing (only if used)
    if prep and getattr(prep, 'strategy_used', 'none') != 'none':
        with st.expander("Preprocessing", expanded=True):
            _render_preprocessing_stage(prep)

    # Retrieval (always shown when we have results)
    if settings:
        with st.expander("Retrieval", expanded=True):
            _render_retrieval_stage(settings, results, prep)

    # RRF (only for decomposition strategy - HyDE uses embedding averaging instead)
    if rrf:
        with st.expander("RRF Merging", expanded=False):
            _render_rrf_stage(rrf, prep)

    # GraphRAG (only if used and successful)
    if graph_meta and not graph_meta.get("error"):
        if graph_meta.get("drift_result"):
            with st.expander("DRIFT Global Search", expanded=False):
                _render_drift_stage(graph_meta)
        else:
            with st.expander("Graph Enrichment", expanded=False):
                _render_graph_stage(graph_meta)
    elif graph_meta and graph_meta.get("error"):
        st.warning(f"GraphRAG failed: {graph_meta['error']}")

    # Reranking (only if enabled)
    if rerank:
        with st.expander("Reranking", expanded=False):
            _render_rerank_stage(rerank)

    # Generation (always shown when we have an answer)
    if ans:
        with st.expander("Generation", expanded=True):
            _render_generation_stage(ans)


# ============================================================================
# SIDEBAR - Simple flow: Preprocessing → Collection → Retrieval → Reranking
# ============================================================================

st.sidebar.title("Settings")

# Load collections once (cached)
try:
    collection_infos = get_available_collections()
    st.session_state.connection_error = None
except Exception as e:
    collection_infos = []
    st.session_state.connection_error = str(e)

# -----------------------------------------------------------------------------
# Query Preprocessing
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Query Preprocessing")

strategy_options = {s[0]: (s[1], s[2]) for s in AVAILABLE_PREPROCESSING_STRATEGIES}
selected_strategy = st.sidebar.selectbox(
    "Strategy",
    options=list(strategy_options.keys()),
    format_func=lambda x: f"{strategy_options[x][0]} - {strategy_options[x][1]}",
    help="How to transform the query before searching.",
    key="preprocessing_strategy",
)
enable_preprocessing = selected_strategy != "none"

# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Collection")

# Get strategy config for constraint checking
strategy_config = get_strategy_config(selected_strategy)

if strategy_config.uses_dedicated_index():
    # Strategy uses dedicated collection (e.g., GraphRAG uses semantic_std2)
    dedicated_name = strategy_config.collection_constraint.dedicated_collection
    st.sidebar.selectbox(
        "Chunking Strategy",
        options=[dedicated_name],
        disabled=True,
        help="This strategy uses a dedicated index.",
    )
    st.sidebar.caption(f"Uses dedicated index: {dedicated_name}")
    # Check if dedicated collection exists
    if dedicated_name in [c.collection_name for c in collection_infos]:
        selected_collection = dedicated_name
    else:
        st.sidebar.error(
            f"Collection '{dedicated_name}' not found. Run the chunking and upload pipeline:\n"
            "1. Stage 4: python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0\n"
            "2. Stage 5: python -m src.stages.run_stage_5_embedding --strategy semantic_std2\n"
            "3. Stage 6: python -m src.stages.run_stage_6_weaviate --strategy semantic_std2"
        )
        selected_collection = None
elif collection_infos:
    # Standard strategies: filter to compatible collections
    compatible = [
        info for info in collection_infos
        if selected_strategy in get_valid_preprocessing_strategies(info.strategy)
    ]
    if compatible:
        selected_collection = st.sidebar.selectbox(
            "Chunking Strategy",
            options=[c.collection_name for c in compatible],
            format_func=lambda x: next(c.display_name for c in compatible if c.collection_name == x),
            help="Which chunking method to search.",
        )
    else:
        st.sidebar.warning("No compatible collections for this strategy.")
        selected_collection = None
else:
    st.sidebar.warning("No collections found. Is Weaviate running?")
    selected_collection = None

# -----------------------------------------------------------------------------
# Retrieval (constraint-aware based on selected strategy)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Retrieval")

# strategy_config was already fetched above for collection constraint
alpha_constraint = strategy_config.alpha_constraint

# Render alpha control based on constraint
if strategy_config.has_internal_search():
    # Strategy performs its own retrieval (e.g., GraphRAG)
    st.sidebar.slider(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        disabled=True,
        help="Search type is controlled by the strategy.",
    )
    st.sidebar.caption("Search: **Internal** (graph + vector retrieval)")
    alpha = 1.0  # Default for internal strategies
elif alpha_constraint.mode == "fixed":
    # Strategy requires specific alpha (e.g., HyDE requires alpha=1.0)
    alpha = alpha_constraint.fixed_value
    if alpha == 1.0:
        alpha_label = "Semantic"
    elif alpha == 0.0:
        alpha_label = "Keyword"
    else:
        alpha_label = f"Hybrid ({alpha})"
    st.sidebar.slider(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=alpha,
        disabled=True,
        help="Alpha is fixed for this strategy.",
    )
    st.sidebar.caption(f"Search: **{alpha_label}** (required by {selected_strategy})")
elif alpha_constraint.mode == "range":
    # Strategy allows alpha within range
    alpha = st.sidebar.slider(
        "Alpha",
        min_value=alpha_constraint.min_value,
        max_value=alpha_constraint.max_value,
        value=alpha_constraint.get_default(),
        step=0.1,
        help="0 = keyword only, 1 = semantic only",
    )
else:
    # Strategy allows any alpha
    alpha = st.sidebar.slider(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = keyword only, 1 = semantic only",
    )

# Always use hybrid search type (alpha controls the balance)
search_type = "hybrid"

top_k = st.sidebar.slider(
    "Results",
    min_value=1,
    max_value=MAX_TOP_K,
    value=DEFAULT_TOP_K,
    help="Number of chunks to retrieve.",
)

# -----------------------------------------------------------------------------
# Reranking (constraint-aware based on selected strategy)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Reranking")

reranking_constraint = strategy_config.reranking_constraint

if reranking_constraint.mode == "required":
    # Strategy requires reranking (e.g., decomposition per arXiv:2507.00355)
    use_reranking = True
    st.sidebar.checkbox(
        "Enable Cross-Encoder",
        value=True,
        disabled=True,
        help="Required by this preprocessing strategy.",
    )
    st.sidebar.caption(f"Required by {selected_strategy}")
elif reranking_constraint.mode == "forbidden":
    # Strategy forbids reranking (e.g., graphrag uses combined_degree ranking)
    use_reranking = False
    st.sidebar.checkbox(
        "Enable Cross-Encoder",
        value=False,
        disabled=True,
        help="Not compatible with this preprocessing strategy.",
    )
    st.sidebar.caption(f"Disabled for {selected_strategy}")
else:
    # Strategy allows optional reranking (none, hyde)
    use_reranking = st.sidebar.checkbox(
        "Enable Cross-Encoder",
        value=False,
        help="Re-score results for higher accuracy (slow on CPU).",
    )
    if use_reranking:
        st.sidebar.caption("~2 min/query on CPU")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header with logo
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.title("RAGLab Search")
    st.markdown("Search across 19 books combining neuroscience and philosophy.")
    # Show connection error if any
    if st.session_state.connection_error:
        st.error(f"Connection Error: {st.session_state.connection_error}")
        st.info("Make sure Weaviate is running: `docker compose up -d`")
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the relationship between emotions and decision-making?",
    )
    # Search button
    search_clicked = st.button("Search", type="primary", disabled=not query)
with header_col2:
    st.image("assets/raglab_logo.png", width="stretch")

# Execute search
if search_clicked and query:
    if not selected_collection:
        st.error("No collection available. Please run `docker compose up -d` and run Stage 6.")
    else:
        # Step 1: Query Preprocessing (optional)
        preprocessed = None
        search_query = query

        if enable_preprocessing:
            with st.spinner("Stage 1: Analyzing query..."):
                try:
                    preprocessed = preprocess_query(
                        query, model=PREPROCESSING_MODEL, strategy=selected_strategy
                    )
                    search_query = preprocessed.search_query
                    st.session_state.preprocessed_query = preprocessed
                except Exception as e:
                    st.warning(f"Preprocessing failed: {e}. Using original query.")
                    st.session_state.preprocessed_query = None
        else:
            st.session_state.preprocessed_query = None

        # Step 2 & 3: Search (with optional reranking)
        # Strategy determines retrieval method: HyDE=embedding averaging, decomposition=union+rerank

        # Build spinner message based on strategy-specific retrieval method
        spinner_msg = "Stage 2: Searching..."
        if selected_strategy == "hyde":
            spinner_msg = "Stage 2: Searching (HyDE embedding averaging)..."
        elif selected_strategy == "decomposition":
            spinner_msg = "Stage 2: Searching (sub-queries + rerank)..."
        elif selected_strategy == "graphrag":
            spinner_msg = "Stage 2: Searching (graph + vector)..."
        # Add reranking indicator for non-decomposition strategies (decomposition always reranks)
        if use_reranking and selected_strategy != "decomposition":
            spinner_msg = spinner_msg.replace("...", " + reranking...")

        with st.spinner(spinner_msg):
            try:
                search_output = search_chunks(
                    query=search_query,
                    top_k=top_k,
                    search_type=search_type,
                    alpha=alpha,
                    collection_name=selected_collection,
                    use_reranking=use_reranking,
                    strategy=selected_strategy,
                )
                st.session_state.search_results = search_output.results
                st.session_state.rerank_data = search_output.rerank_data
                st.session_state.rrf_data = search_output.rrf_data
                st.session_state.graph_metadata = search_output.graph_metadata
                st.session_state.last_query = query
                # Get display name for the selected collection
                display_name = next(
                    (c.display_name for c in collection_infos if c.collection_name == selected_collection),
                    selected_collection,
                )
                st.session_state.retrieval_settings = {
                    "search_type": search_type,
                    "alpha": alpha,
                    "top_k": top_k,
                    "collection_name": selected_collection,
                    "collection_display_name": display_name,
                }
                st.session_state.connection_error = None
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []
                st.session_state.generated_answer = None
                st.session_state.rerank_data = None
                st.session_state.rrf_data = None
                st.session_state.graph_metadata = None
                preprocessed = None

        # Step 4: Answer Generation
        if st.session_state.search_results:
            with st.spinner("Stage 4: Generating answer..."):
                try:
                    # Format graph context if GraphRAG was used successfully
                    graph_context = None
                    graph_meta = st.session_state.graph_metadata
                    if graph_meta and not graph_meta.get("error"):
                        graph_context = format_graph_context_for_generation(graph_meta)

                    answer = generate_answer(
                        query=query,
                        chunks=st.session_state.search_results,
                        model=GENERATION_MODEL,
                        graph_context=graph_context,
                    )
                    st.session_state.generated_answer = answer
                except Exception as e:
                    st.warning(f"Answer generation failed: {e}")
                    st.session_state.generated_answer = None
        elif (st.session_state.graph_metadata
              and st.session_state.graph_metadata.get("drift_result")):
            # Global query: DRIFT already produced the answer (no chunks)
            mr = st.session_state.graph_metadata["drift_result"]
            from src.rag_pipeline.generation.answer_generator import GeneratedAnswer
            n_communities = len(mr.get("communities_used", []))
            n_folds = len(mr.get("intermediate_answers", []))
            drift_summary = (
                f"DRIFT Global Search\n"
                f"Communities: {n_communities} | "
                f"Primer folds: {n_folds} | "
                f"LLM calls: {mr.get('total_llm_calls', 0)}\n"
                f"Primer: {mr.get('primer_time_ms', 0):.0f}ms | "
                f"Reduce: {mr.get('reduce_time_ms', 0):.0f}ms | "
                f"Total: {mr.get('total_time_ms', 0):.0f}ms"
            )
            st.session_state.generated_answer = GeneratedAnswer(
                answer=mr["final_answer"],
                model=GENERATION_MODEL,
                generation_time_ms=mr.get("total_time_ms", 0),
                sources_used=[],
                system_prompt_used=drift_summary,
                user_prompt_used=query,
            )
        else:
            # No results to generate from
            st.session_state.generated_answer = None

        # Auto-save successful queries to log
        if st.session_state.search_results:
            from src.shared.query_logger import log_query
            log_query(
                query=query,
                preprocessed=preprocessed,
                retrieval_settings=st.session_state.retrieval_settings,
                search_results=st.session_state.search_results,
                rerank_data=st.session_state.rerank_data,
                generated_answer=st.session_state.generated_answer,
                collection_name=selected_collection,
                rrf_data=st.session_state.rrf_data,
            )


# ============================================================================
# RESULTS DISPLAY - Tabs: Answer | Pipeline Log
# ============================================================================

if st.session_state.search_results or st.session_state.generated_answer:
    #st.divider()
    #st.subheader(f"Results for: \"{st.session_state.last_query}\"")

    # Create tabs for different views
    tab_answer, tab_log = st.tabs(["Answer", "Pipeline Log"])

    # -------------------------------------------------------------------------
    # TAB 1: Answer
    # -------------------------------------------------------------------------
    with tab_answer:
        # Compact config summary
        st.caption(f"config: {_format_config_summary()}")

        # Generated Answer Section
        if st.session_state.generated_answer:
            ans = st.session_state.generated_answer
            st.markdown("### Generated Answer")

            # Display the answer
            st.markdown(ans.answer)

            # Show metadata
            graph_meta = st.session_state.graph_metadata
            is_global = (graph_meta and graph_meta.get("drift_result"))
            col1, col2, col3 = st.columns(3)
            col1.caption(f"Model: {ans.model}")
            if is_global:
                dr = graph_meta["drift_result"]
                n_comm = len(dr.get("communities_used", []))
                col2.caption(f"Communities: {n_comm} | LLM calls: {dr.get('total_llm_calls', 0)}")
            else:
                col2.caption(f"Sources cited: {ans.sources_used}")
            col3.caption(f"Generated in {ans.generation_time_ms:.0f}ms")

            # Display formatted references for cited sources
            results = st.session_state.search_results
            if ans.sources_used and results:
                st.markdown("---")
                st.caption("References")
                for idx in sorted(ans.sources_used):
                    if 1 <= idx <= len(results):
                        chunk = results[idx - 1]  # Convert 1-based to 0-based
                        book_parts = chunk["book_id"].rsplit("(", 1)
                        book_title = book_parts[0].strip()
                        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""
                        section = chunk.get("section", "")
                        ref_text = f"[{idx}] {book_title}"
                        if author:
                            ref_text += f" — {author}"
                        if section:
                            ref_text += f", Section: {section}"
                        st.caption(ref_text)

        else:
            st.warning("Answer generation failed. Check the Pipeline Log tab for details.")

    # -------------------------------------------------------------------------
    # TAB 2: Pipeline Log
    # -------------------------------------------------------------------------
    with tab_log:
        st.markdown("### Pipeline Execution Log")
        st.caption("Full visibility into what happened at each stage of the RAG pipeline.")
        _render_pipeline_log()

elif query and not st.session_state.search_results and not st.session_state.generated_answer:
    st.info("No results found. Try a different query.")

else:
    st.info("Enter a query above to search the knowledge base.")


