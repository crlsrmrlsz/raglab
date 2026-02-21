"""LLM prompt templates for RAGLab pipeline.

Contains all prompts used for:
- Query preprocessing (HyDE, decomposition)
- Answer generation
- GraphRAG entity extraction and community summarization
- Contextual chunking
- RAPTOR hierarchical summarization
"""


# =============================================================================
# QUERY PREPROCESSING PROMPTS
# =============================================================================

# Split prompts for dual-domain corpus (2 neuroscience + 2 philosophy hypotheticals)
# Explicit length constraint for modern models (paper used InstructGPT which was naturally concise)
HYDE_PROMPT_NEUROSCIENCE = """Write a brief 2-3 sentence passage from a neuroscience textbook to answer the question.

Question: {query}

Passage:"""

HYDE_PROMPT_PHILOSOPHY = """Write a brief 2-3 sentence passage from a classical wisdom/philosophy essay to answer the question.

Question: {query}

Passage:"""

DECOMPOSITION_PROMPT = """Break down this question for a knowledge base spanning neuroscience research and classical wisdom/philosophy.

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

# Hierarchical community summarization prompt (Microsoft GraphRAG bottom-up approach)
# Used when a community aggregates multiple sub-communities and child summaries
# are substituted for raw entity data due to token limits
GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT = """You are analyzing a community of related entities from a knowledge graph.
This community aggregates multiple sub-communities detected via the Leiden algorithm.

The context below includes:
- Sub-community summaries (marked with [Sub-Community]) representing clusters of related entities
- Cross-community relationships connecting entities across sub-communities

Community context:
{community_context}

Write a comprehensive summary (3-4 paragraphs) that:
1. Identifies the overarching theme connecting these sub-communities
2. Synthesizes insights across sub-community themes
3. Highlights important cross-cutting relationships and patterns

Ensure all sentences are complete.

Summary:"""


# =============================================================================
# GRAPHRAG CHUNK EXTRACTION PROMPT
# =============================================================================

# Constrained entity extraction using curated types from graphrag_types.yaml
# Per Microsoft GraphRAG: entity types are predefined, relationships are open-ended
GRAPHRAG_CHUNK_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

ENTITY TYPES (use ONLY these): {entity_types}

For each entity:
- name: The entity name as it appears in text
- entity_type: One of the types above (choose the BEST match)
- description: Brief description (under 15 words)

For each relationship:
- source_entity / target_entity: Entity names from above
- relationship_type: Free-form type (e.g., CAUSES, MODULATES, PROPOSES, INFLUENCES)
- description: Brief description of the relationship
- weight: 0.0-1.0 (strength/importance)

LIMITS: Up to {max_entities} entities and {max_relationships} relationships.
Focus on significant concepts mentioned in the text.

Text:
{text}

IMPORTANT: Respond ONLY with valid JSON:
{{"entities": [{{"name": "...", "entity_type": "...", "description": "..."}}], "relationships": [{{"source_entity": "...", "target_entity": "...", "relationship_type": "...", "description": "...", "weight": 1.0}}]}}"""


# =============================================================================
# GRAPHRAG GLEANING PROMPTS (Multi-pass extraction)
# =============================================================================
# From Microsoft GraphRAG paper (arXiv:2404.16130):
# "We use multiple rounds of 'gleanings' to encourage the LLM to detect
# any additional entities it may have missed on prior extraction rounds."

# Loop check: Ask if more entities remain (expects Y/N)
GRAPHRAG_LOOP_PROMPT = """Based on the text and your previous extraction, are there any important entities or relationships you may have missed?

Consider:
- Key people, concepts, or locations not yet captured
- Relationships between entities not yet documented
- Implicit entities that are important for understanding the text

Answer with ONLY 'Y' if there are missed entities to extract, or 'N' if extraction is complete."""

# Continue prompt: Encourage extraction of missed entities
GRAPHRAG_CONTINUE_PROMPT = """MANY entities and relationships were missed in the previous extraction.

Original text:
{text}

Previously extracted (DO NOT repeat these):
Entities: {previous_entities}
Relationships: {previous_relationships}

ENTITY TYPES (use ONLY these): {entity_types}

Extract ADDITIONAL entities and relationships that were missed. Focus on:
- Entities that were implied but not explicitly extracted
- Secondary characters, concepts, or locations
- Relationships between previously extracted entities
- Any important details overlooked

IMPORTANT: Respond ONLY with valid JSON:
{{"entities": [{{"name": "...", "entity_type": "...", "description": "..."}}], "relationships": [{{"source_entity": "...", "target_entity": "...", "relationship_type": "...", "description": "...", "weight": 1.0}}]}}"""


# =============================================================================
# GRAPHRAG CONSOLIDATION PROMPTS (Microsoft GraphRAG approach)
# =============================================================================

# Entity description summarization
# Used during indexing to merge duplicate entities into coherent descriptions
GRAPHRAG_ENTITY_SUMMARIZE_PROMPT = """Summarize these descriptions of "{entity_name}" ({entity_type}) into one coherent description.

Descriptions:
{descriptions}

Write a single description (2-3 sentences) capturing the key information. Do not mention "sources" or "descriptions".

Summary:"""

# Relationship description summarization
# Used during indexing to merge duplicate relationships
GRAPHRAG_RELATIONSHIP_SUMMARIZE_PROMPT = """Summarize these descriptions of the relationship between "{source}" and "{target}" into one sentence.

Descriptions:
{descriptions}

Write a clear, concise description (1-2 sentences).

Summary:"""


# =============================================================================
# MAP-REDUCE PROMPTS (DEPRECATED — replaced by DRIFT prompts in src/graph/drift.py)
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

# ---------------------------------------------------------------------------
# DRIFT Search prompts (simplified: primer + reduce, no follow-ups)
# ---------------------------------------------------------------------------

# Primer phase: Process one fold of community reports, generate intermediate answer
DRIFT_PRIMER_PROMPT = """You are analyzing community reports from a knowledge graph to answer a query.

Query: {query}

Community Reports (ranked by relevance):
{community_reports}

Based on these community reports:
1. Provide an intermediate answer to the query using ONLY information from the reports above.
2. Rate the relevance of these communities to the query on a scale of 0-10.

If the communities are not relevant, say "Not relevant" and give a score of 0.

Format your response as:
[Score: X/10]
[Answer]
Your intermediate answer here."""

# Reduce phase: Synthesize scored intermediate answers into final response
DRIFT_REDUCE_PROMPT = """Synthesize these intermediate answers into a comprehensive final response.

Query: {query}

Intermediate Answers (from different community groups, ranked by relevance score):
{intermediate_answers}

Create a unified, coherent answer that:
1. Prioritizes insights from higher-scored intermediate answers
2. Integrates complementary information across groups
3. Identifies common themes and contrasting perspectives
4. Omits information marked as "Not relevant"

Synthesized Answer:"""
