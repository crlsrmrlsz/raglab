"""LLM prompt templates for RAGLab pipeline.

Contains all prompts used for:
- Query preprocessing (HyDE, decomposition)
- Answer generation
- GraphRAG entity extraction and community summarization
- Auto-tuning type discovery and consolidation
- Contextual chunking
- RAPTOR hierarchical summarization
"""


# =============================================================================
# QUERY PREPROCESSING PROMPTS
# =============================================================================

HYDE_PROMPT = """Please write a passage from a neuroscience textbook or classical wisdom essay to answer the question.

Question: {query}

Passage:"""

DECOMPOSITION_PROMPT = """Break down this question for a knowledge base on cognitive science and philosophy.

If the question is simple enough to answer directly, keep it as a single question.
Otherwise, create 3-5 sub-questions that can be answered independently and together cover all aspects of the original.

Question: {query}

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation"
}}"""


# =============================================================================
# ANSWER GENERATION PROMPTS
# =============================================================================

GENERATION_SYSTEM_PROMPT = """You are a knowledgeable assistant that synthesizes information from diverse sources.

Your context may include:
- Scientific sources (neuroscience, cognitive science, psychology)
- Philosophical and wisdom literature (Stoics, Eastern philosophy, etc.)

When relevant, distinguish between empirical findings and philosophical insights,
but structure your answer naturally based on what the question needs.

Cite sources by number [1], [2], etc. so users can explore further."""


# =============================================================================
# CONTEXTUAL CHUNKING PROMPTS
# =============================================================================

# Context generation adapted from Anthropic's approach.
# Instead of passing the full document (impractical for books), we pass:
# - Book title (LLM may recognize well-known books/authors)
# - Section title (contains disambiguation terms like "Ventral Striatum: Pleasure and Reward")
# - The chunk text
CONTEXTUAL_PROMPT = """<book>
{book_title}
</book>

<section>
{section_title}
</section>

<chunk>
{chunk_text}
</chunk>

Write a brief context (1-2 sentences, ~50-80 words) to situate this chunk within the book for search retrieval.
Use key terms from the section title. Ensure sentences are complete - do not end mid-sentence.
Answer only with the context, nothing else."""


# =============================================================================
# RAPTOR PROMPTS
# =============================================================================

RAPTOR_SUMMARY_PROMPT = """Write a concise summary (~100-150 words) of the following content.
Include key details but ensure all sentences are complete. Do not end mid-sentence.

Content:
{context}

Summary:"""


# =============================================================================
# GRAPHRAG PROMPTS
# =============================================================================

# Query-time entity extraction (simpler than chunk extraction)
GRAPHRAG_QUERY_EXTRACTION_PROMPT = """Identify entities mentioned or implied in this query.

Entity types: {entity_types}

Query: {query}

Extract all relevant entities, including:
- Explicitly named entities (e.g., "Sapolsky", "dopamine")
- Implied concepts (e.g., "why we procrastinate" implies "procrastination")
- Domain concepts (e.g., "self-control", "consciousness", "happiness")

Be concise - extract only the key entities (typically 1-5 per query).

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{"entities": [{{"name": "entity_name", "entity_type": "TYPE"}}]}}

Example response for "How does stress affect memory?":
{{"entities": [{{"name": "stress", "entity_type": "CONCEPT"}}, {{"name": "memory", "entity_type": "COGNITIVE_PROCESS"}}]}}"""

# Community summarization prompt
GRAPHRAG_COMMUNITY_PROMPT = """You are analyzing a community of related entities from a knowledge graph.
This community was detected via the Leiden algorithm and contains semantically related concepts.

Community entities and their relationships:
{community_context}

Write a summary (2-3 short paragraphs, ~150-200 words) that:
1. Identifies the main theme or topic connecting these entities
2. Explains the key relationships and how concepts interact
3. Highlights important details, names, and specific findings

Ensure all sentences are complete. Do not end mid-sentence.

Summary:"""


# =============================================================================
# AUTO-TUNING PROMPTS (GraphRAG Entity Type Discovery)
# =============================================================================

# Open-ended entity extraction (discovers types from corpus)
GRAPHRAG_OPEN_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

For each entity, assign the MOST APPROPRIATE TYPE (use UPPERCASE_SNAKE_CASE).
Common types: BRAIN_REGION, NEUROTRANSMITTER, CONCEPT, PHILOSOPHER, RESEARCHER, BEHAVIOR, EMOTION, BOOK, STUDY.
You may create NEW types if none fit well.

LIMITS: Up to {max_entities} entities and {max_relationships} relationships.
Keep descriptions under 15 words. Focus on significant concepts.

Text:
{text}

IMPORTANT: Respond ONLY with valid JSON:
{{"entities": [{{"name": "...", "entity_type": "...", "description": "..."}}], "relationships": [{{"source_entity": "...", "target_entity": "...", "relationship_type": "...", "description": "...", "weight": 1.0}}]}}"""

# Global consolidation (single pass, may favor larger corpora)
GRAPHRAG_GLOBAL_CONSOLIDATION_PROMPT = """Consolidate these discovered entity/relationship types into a clean taxonomy.

ENTITY TYPES (with counts):
{entity_types}

RELATIONSHIP TYPES (with counts):
{relationship_types}

Rules:
1. Merge similar types (e.g., BRAIN_REGION + NEURAL_STRUCTURE)
2. Remove types with count=1 unless clearly important
3. Target: 15-25 entity types, 10-20 relationship types

Respond with JSON: {{"entity_types": [...], "relationship_types": [...], "rationale": "..."}}"""

# Stratified consolidation (balances across domains)
GRAPHRAG_STRATIFIED_CONSOLIDATION_PROMPT = """Consolidate entity types from TWO domains with BALANCED representation.

DOMAIN 1: {corpus1_name}
{corpus1_types}

DOMAIN 2: {corpus2_name}
{corpus2_types}

SHARED TYPES:
{shared_types}

RELATIONSHIP TYPES:
{relationship_types}

Rules:
1. Keep domain-specific types even if low global count
2. Merge obviously similar types across domains
3. Target: 20-25 entity types, 12-18 relationship types
4. Ensure BOTH domains are well-represented

Respond with JSON: {{"entity_types": [...], "relationship_types": [...], "rationale": "..."}}"""


# =============================================================================
# MAP-REDUCE PROMPTS (GraphRAG Global Query Handling)
# =============================================================================

# Query classification: local (entity-specific) vs global (thematic)
GRAPHRAG_CLASSIFICATION_PROMPT = """Classify this query as 'local' or 'global'.

LOCAL queries ask about specific entities, concepts, or facts:
- "What is dopamine?"
- "How does the prefrontal cortex affect decision-making?"
- "What did Sapolsky say about stress?"

GLOBAL queries ask about themes, patterns, or overviews across topics:
- "What are the main themes in this corpus?"
- "How do neuroscience and philosophy approaches differ?"
- "What are the key insights about human behavior?"

Query: {query}

Respond with ONLY the word 'local' or 'global'."""

# Map phase: Generate partial answer from one community
GRAPHRAG_MAP_PROMPT = """Answer this question using ONLY the community context below.

Community Theme:
{community_summary}

Key Entities (ranked by importance):
{top_entities}

Key Relationships:
{relationships}

Question: {query}

Provide a concise answer (2-3 sentences) based ONLY on this community's themes and entities.
If this community is not relevant to the question, respond with "Not relevant to this community."

Answer:"""

# Reduce phase: Synthesize partial answers into final response
GRAPHRAG_REDUCE_PROMPT = """Synthesize these partial answers from different thematic communities into a comprehensive response.

Question: {query}

Partial Answers from Communities:
{partial_answers}

Create a unified, coherent answer that:
1. Integrates insights from relevant communities
2. Identifies common themes across communities
3. Notes any contrasting perspectives if present
4. Ignores "Not relevant" responses

Synthesized Answer:"""
