"""GraphRAG entity extraction with curated entity types and consolidation.

## What This Module Does

1. Constrained Extraction: LLM extracts entities using predefined types from graphrag_types.yaml
2. Relationship Extraction: Relationships are extracted with open-ended types (per GraphRAG paper)
3. Entity Consolidation: Merge duplicates by (normalized_name, entity_type) with LLM summarization
4. Relationship Consolidation: Merge duplicates by (source, target, type) with LLM summarization
5. Save Results: Stores extraction_results.json for Neo4j upload in Stage 6b

## Key Design Decisions

Entity types are curated in graphrag_types.yaml rather than auto-discovered.
This provides:
- Consistent taxonomy across extraction runs
- Better control over entity granularity

Microsoft GraphRAG consolidation approach:
- Entities merged by (normalized_name, entity_type) - allows same name with different types
- Descriptions are LLM-summarized when duplicates have multiple unique descriptions
- source_chunk_ids tracked as list for provenance (all source chunks preserved)
"""

from typing import Any, Optional
from pathlib import Path
from collections import Counter
import json
import time

from pydantic import BaseModel, Field

from src.config import (
    GRAPHRAG_EXTRACTION_MODEL,
    GRAPHRAG_MAX_EXTRACTION_TOKENS,
    GRAPHRAG_MAX_ENTITIES,
    GRAPHRAG_MAX_RELATIONSHIPS,
    DIR_GRAPH_DATA,
    DIR_FINAL_CHUNKS,
    GRAPHRAG_CHUNK_EXTRACTION_PROMPT,
    GRAPHRAG_SUMMARY_MODEL,
    GRAPHRAG_STRICT_MODE,
    GRAPHRAG_MAX_GLEANINGS,
)
from src.prompts import (
    GRAPHRAG_ENTITY_SUMMARIZE_PROMPT,
    GRAPHRAG_RELATIONSHIP_SUMMARIZE_PROMPT,
    GRAPHRAG_LOOP_PROMPT,
    GRAPHRAG_CONTINUE_PROMPT,
)
from src.graph.graphrag_types import get_entity_types, get_entity_types_string
from src.shared.openrouter_client import call_structured_completion, call_chat_completion
from src.shared.files import setup_logging, OverwriteContext

logger = setup_logging(__name__)


# ============================================================================
# Pydantic Schemas
# ============================================================================

class ExtractedEntity(BaseModel):
    """Entity extracted from a chunk."""
    name: str = Field(..., min_length=1)
    entity_type: str = Field(...)
    description: str = Field(default="")


class ExtractedRelationship(BaseModel):
    """Relationship extracted from a chunk."""
    source_entity: str = Field(...)
    target_entity: str = Field(...)
    relationship_type: str = Field(...)
    description: str = Field(default="")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ChunkExtractionResult(BaseModel):
    """Result of entity/relationship extraction from a chunk."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ============================================================================
# Strict Mode Filtering
# ============================================================================

def filter_entities_strict(
    entities: list[ExtractedEntity],
    allowed_types: set[str],
) -> tuple[list[ExtractedEntity], int]:
    """Filter entities to only those with allowed types (case-insensitive).

    Follows LangChain's approach: when the LLM ignores type constraints and
    invents new types, discard those entities to prevent graph fragmentation.

    Args:
        entities: List of extracted entities to filter.
        allowed_types: Set of allowed entity type names.

    Returns:
        Tuple of (filtered_entities, discarded_count).
    """
    allowed_lower = {t.lower() for t in allowed_types}
    filtered = []
    discarded = 0

    for entity in entities:
        if entity.entity_type.lower() in allowed_lower:
            filtered.append(entity)
        else:
            discarded += 1

    return filtered, discarded


# ============================================================================
# Core Extraction Functions
# ============================================================================

def extract_chunk(
    chunk: dict[str, Any],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    max_gleanings: int = GRAPHRAG_MAX_GLEANINGS,
) -> ChunkExtractionResult:
    """Extract entities/relationships from a single chunk using curated types.

    Uses Microsoft GraphRAG's gleaning mechanism for improved recall:
    1. Initial extraction pass
    2. Check if entities were missed (LOOP_PROMPT)
    3. If yes, continue extracting (CONTINUE_PROMPT)
    4. Repeat up to max_gleanings times

    Args:
        chunk: Chunk dict with 'text' field.
        model: LLM model for extraction.
        max_gleanings: Maximum additional extraction passes (0 = disabled).

    Returns:
        ChunkExtractionResult with merged entities and relationships.
    """
    entity_types = get_entity_types_string()
    text = chunk["text"]

    # --- Initial extraction ---
    prompt = GRAPHRAG_CHUNK_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        text=text,
        max_entities=GRAPHRAG_MAX_ENTITIES,
        max_relationships=GRAPHRAG_MAX_RELATIONSHIPS,
    )
    result = call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ChunkExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )

    all_entities = list(result.entities)
    all_relationships = list(result.relationships)

    # --- Gleaning loop ---
    for i in range(max_gleanings):
        # Check if more entities to extract
        if not _should_continue_gleaning(text, all_entities, all_relationships, model):
            logger.debug(f"Gleaning stopped after {i} rounds (LLM said complete)")
            break

        # Extract additional entities
        additional = _glean_additional_entities(
            text=text,
            previous_entities=all_entities,
            previous_relationships=all_relationships,
            entity_types=entity_types,
            model=model,
        )

        if not additional.entities and not additional.relationships:
            logger.debug(f"Gleaning stopped after {i+1} rounds (no new entities)")
            break

        all_entities.extend(additional.entities)
        all_relationships.extend(additional.relationships)
        logger.debug(
            f"Gleaning round {i+1}: +{len(additional.entities)} entities, "
            f"+{len(additional.relationships)} relationships"
        )

    # Deduplicate within chunk
    final_entities = _deduplicate_entities(all_entities)
    final_relationships = _deduplicate_relationships(all_relationships)

    result = ChunkExtractionResult(entities=final_entities, relationships=final_relationships)

    # Apply strict mode filtering if enabled
    if GRAPHRAG_STRICT_MODE:
        allowed = set(get_entity_types())
        filtered, discarded = filter_entities_strict(result.entities, allowed)
        if discarded > 0:
            logger.debug(f"Strict mode: discarded {discarded} entities with non-matching types")
        result = ChunkExtractionResult(entities=filtered, relationships=result.relationships)

    return result


def _should_continue_gleaning(
    text: str,
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
    model: str,
) -> bool:
    """Ask LLM if more entities remain to extract.

    Args:
        text: Original chunk text.
        entities: Previously extracted entities.
        relationships: Previously extracted relationships.
        model: LLM model.

    Returns:
        True if LLM indicates more entities to extract.
    """
    # Build context of what was extracted
    entity_summary = ", ".join(e.name for e in entities[:10])  # First 10 for context
    rel_summary = ", ".join(
        f"{r.source_entity}->{r.target_entity}" for r in relationships[:5]
    )

    context = f"""Text excerpt: {text[:500]}...

Previously extracted entities: {entity_summary}
Previously extracted relationships: {rel_summary}

{GRAPHRAG_LOOP_PROMPT}"""

    response = call_chat_completion(
        messages=[{"role": "user", "content": context}],
        model=model,
        temperature=0.0,
        max_tokens=5,
    )

    return response.strip().upper().startswith("Y")


def _glean_additional_entities(
    text: str,
    previous_entities: list[ExtractedEntity],
    previous_relationships: list[ExtractedRelationship],
    entity_types: str,
    model: str,
) -> ChunkExtractionResult:
    """Extract additional entities missed in previous passes.

    Args:
        text: Original chunk text.
        previous_entities: Already extracted entities.
        previous_relationships: Already extracted relationships.
        entity_types: Allowed entity types string.
        model: LLM model.

    Returns:
        ChunkExtractionResult with additional entities/relationships.
    """
    prev_entity_str = ", ".join(f"{e.name} ({e.entity_type})" for e in previous_entities)
    prev_rel_str = ", ".join(
        f"{r.source_entity} -> {r.target_entity}" for r in previous_relationships
    )

    prompt = GRAPHRAG_CONTINUE_PROMPT.format(
        text=text,
        previous_entities=prev_entity_str or "None",
        previous_relationships=prev_rel_str or "None",
        entity_types=entity_types,
    )

    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ChunkExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )


def _deduplicate_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Deduplicate entities within a chunk by normalized name.

    Args:
        entities: List of extracted entities.

    Returns:
        Deduplicated list, keeping first occurrence.
    """
    from src.graph.schemas import GraphEntity

    seen: set[tuple[str, str]] = set()
    unique: list[ExtractedEntity] = []
    for e in entities:
        ge = GraphEntity(name=e.name, entity_type=e.entity_type)
        key = (ge.normalized_name(), e.entity_type.lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _deduplicate_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Deduplicate relationships within a chunk.

    Args:
        relationships: List of extracted relationships.

    Returns:
        Deduplicated list, keeping first occurrence.
    """
    from src.graph.schemas import GraphEntity

    seen: set[tuple[str, str, str]] = set()
    unique: list[ExtractedRelationship] = []
    for r in relationships:
        source_ge = GraphEntity(name=r.source_entity, entity_type="")
        target_ge = GraphEntity(name=r.target_entity, entity_type="")
        key = (
            source_ge.normalized_name(),
            target_ge.normalized_name(),
            r.relationship_type.lower(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def extract_book(
    book_path: Path,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> dict[str, Any]:
    """Extract entities/relationships from all chunks in a book.

    Args:
        book_path: Path to book's chunk JSON file.
        model: LLM model for extraction.

    Returns:
        Dict with entities, relationships, type counts, and stats.
    """
    book_name = book_path.stem

    # Load chunks
    with open(book_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data if isinstance(data, list) else data.get("chunks", [])

    all_entities: list[dict[str, Any]] = []
    all_relationships: list[dict[str, Any]] = []
    entity_type_counter: Counter = Counter()
    relationship_type_counter: Counter = Counter()
    failed_chunks = 0

    logger.info(f"Extracting from {book_name}: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        try:
            result = extract_chunk(chunk, model=model)
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")

            for entity in result.entities:
                entity_dict = entity.model_dump()
                entity_dict["source_chunk_id"] = chunk_id
                all_entities.append(entity_dict)
                entity_type_counter[entity.entity_type] += 1

            for rel in result.relationships:
                rel_dict = rel.model_dump()
                rel_dict["source_chunk_id"] = chunk_id
                all_relationships.append(rel_dict)
                relationship_type_counter[rel.relationship_type] += 1

            if (i + 1) % 20 == 0:
                logger.info(f"  [{book_name}] {i + 1}/{len(chunks)} chunks")

            time.sleep(1.5)  # Rate limit: 40 RPM

        except Exception as e:
            logger.warning(f"Failed chunk {chunk.get('chunk_id', i)}: {e}")
            failed_chunks += 1

    logger.info(
        f"  [{book_name}] Done: {len(chunks) - failed_chunks}/{len(chunks)} chunks, "
        f"{len(all_entities)} entities"
    )

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "relationship_type_counts": dict(relationship_type_counter.most_common()),
        "stats": {
            "book_name": book_name,
            "total_chunks": len(chunks),
            "processed_chunks": len(chunks) - failed_chunks,
            "failed_chunks": failed_chunks,
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
        },
    }


# ============================================================================
# File I/O Functions
# ============================================================================

def load_book_files(strategy: str = "section") -> list[Path]:
    """Get list of book chunk files.

    Args:
        strategy: Chunking strategy subfolder (default: "section").

    Returns:
        List of paths to book chunk JSON files.
    """
    chunk_dir = DIR_FINAL_CHUNKS / strategy
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    return sorted(chunk_dir.glob("*.json"))


def merge_extractions(
    extractions_dir: Path,
    model: str = GRAPHRAG_SUMMARY_MODEL,
) -> dict[str, Any]:
    """Merge all per-book extraction files and consolidate duplicates.

    Microsoft GraphRAG approach:
    1. Merge entities/relationships from all book extraction files
    2. Consolidate entities by (normalized_name, entity_type)
    3. Consolidate relationships by (source, target, type)
    4. LLM-summarize descriptions for duplicates

    Args:
        extractions_dir: Directory containing per-book extraction JSON files.
        model: LLM model for description summarization.

    Returns:
        Dict with consolidated entities, relationships, counts, and stats.
    """
    all_entities: list[dict[str, Any]] = []
    all_relationships: list[dict[str, Any]] = []
    entity_counter: Counter = Counter()
    relationship_counter: Counter = Counter()
    stats = {"total_books": 0, "processed_chunks": 0, "failed_chunks": 0}

    for f in sorted(extractions_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        all_entities.extend(data.get("entities", []))
        all_relationships.extend(data.get("relationships", []))
        for t, c in data.get("entity_type_counts", {}).items():
            entity_counter[t] += c
        for t, c in data.get("relationship_type_counts", {}).items():
            relationship_counter[t] += c
        book_stats = data.get("stats", {})
        stats["total_books"] += 1
        stats["processed_chunks"] += book_stats.get("processed_chunks", 0)
        stats["failed_chunks"] += book_stats.get("failed_chunks", 0)

    stats["raw_entities"] = len(all_entities)
    stats["raw_relationships"] = len(all_relationships)
    stats["unique_entity_types"] = len(entity_counter)
    stats["unique_relationship_types"] = len(relationship_counter)

    # Consolidate duplicates with LLM summarization (Microsoft GraphRAG approach)
    logger.info("Consolidating entities...")
    consolidated_entities = consolidate_entities(all_entities, model=model)

    logger.info("Consolidating relationships...")
    consolidated_relationships = consolidate_relationships(all_relationships, model=model)

    stats["total_entities"] = len(consolidated_entities)
    stats["total_relationships"] = len(consolidated_relationships)

    return {
        "entities": consolidated_entities,
        "relationships": consolidated_relationships,
        "entity_type_counts": dict(entity_counter.most_common()),
        "relationship_type_counts": dict(relationship_counter.most_common()),
        "stats": stats,
    }


# ============================================================================
# Entity/Relationship Consolidation (Microsoft GraphRAG approach)
# ============================================================================

def summarize_descriptions(
    descriptions: list[str],
    prompt_template: str,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    **format_kwargs,
) -> str:
    """Summarize multiple descriptions into one using LLM.

    Args:
        descriptions: List of description strings to summarize.
        prompt_template: The prompt template with {descriptions} placeholder.
        model: LLM model for summarization.
        **format_kwargs: Additional format arguments for the prompt template.

    Returns:
        Summarized description string.
    """
    # Format descriptions as numbered list
    desc_text = "\n".join(f"- {d}" for d in descriptions if d.strip())

    prompt = prompt_template.format(descriptions=desc_text, **format_kwargs)

    return call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=200,
    ).strip()


def consolidate_entities(
    entities: list[dict[str, Any]],
    model: str = GRAPHRAG_SUMMARY_MODEL,
) -> list[dict[str, Any]]:
    """Consolidate duplicate entities by (normalized_name, entity_type).

    Microsoft GraphRAG approach:
    - Group entities by (normalized_name, entity_type)
    - For groups with multiple descriptions: LLM summarize
    - Track all source_chunk_ids as a list

    Args:
        entities: List of entity dicts from extraction.
        model: LLM model for description summarization.

    Returns:
        List of consolidated entity dicts with unique (name, type) pairs.
    """
    from src.graph.schemas import GraphEntity

    # Group by (normalized_name, entity_type)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for entity in entities:
        ge = GraphEntity(
            name=entity["name"],
            entity_type=entity.get("entity_type", ""),
        )
        key = (ge.normalized_name(), entity.get("entity_type", ""))
        if key not in groups:
            groups[key] = []
        groups[key].append(entity)

    consolidated: list[dict[str, Any]] = []
    summarized_count = 0

    for (normalized_name, entity_type), group in groups.items():
        # Collect all unique descriptions
        descriptions = list({e.get("description", "") for e in group if e.get("description", "").strip()})

        # Collect all source chunk IDs
        source_chunk_ids = [e.get("source_chunk_id") for e in group if e.get("source_chunk_id")]

        # Use first entity's name (preserve original casing)
        name = group[0]["name"]

        # Decide on description
        if len(descriptions) > 1:
            # Multiple descriptions: LLM summarize
            description = summarize_descriptions(
                descriptions,
                GRAPHRAG_ENTITY_SUMMARIZE_PROMPT,
                model=model,
                entity_name=name,
                entity_type=entity_type,
            )
            summarized_count += 1
        elif descriptions:
            description = descriptions[0]
        else:
            description = ""

        consolidated.append({
            "name": name,
            "normalized_name": normalized_name,
            "entity_type": entity_type,
            "description": description,
            "source_chunk_ids": source_chunk_ids,
        })

    logger.info(
        f"Consolidated {len(entities)} entities -> {len(consolidated)} unique "
        f"({summarized_count} LLM summarizations)"
    )

    return consolidated


def consolidate_relationships(
    relationships: list[dict[str, Any]],
    model: str = GRAPHRAG_SUMMARY_MODEL,
) -> list[dict[str, Any]]:
    """Consolidate duplicate relationships by (source, target, type).

    Microsoft GraphRAG approach:
    - Group by (source_normalized, target_normalized, relationship_type)
    - For groups with multiple descriptions: LLM summarize
    - Average weights from duplicates
    - Track all source_chunk_ids as a list

    Args:
        relationships: List of relationship dicts from extraction.
        model: LLM model for description summarization.

    Returns:
        List of consolidated relationship dicts.
    """
    from src.graph.schemas import GraphEntity

    # Group by (source_normalized, target_normalized, relationship_type)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for rel in relationships:
        source_ge = GraphEntity(name=rel["source_entity"], entity_type="")
        target_ge = GraphEntity(name=rel["target_entity"], entity_type="")
        key = (
            source_ge.normalized_name(),
            target_ge.normalized_name(),
            rel.get("relationship_type", "RELATED"),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(rel)

    consolidated: list[dict[str, Any]] = []
    summarized_count = 0

    for (source_norm, target_norm, rel_type), group in groups.items():
        # Collect unique descriptions
        descriptions = list({r.get("description", "") for r in group if r.get("description", "").strip()})

        # Collect all source chunk IDs
        source_chunk_ids = [r.get("source_chunk_id") for r in group if r.get("source_chunk_id")]

        # Average weights
        weights = [r.get("weight", 1.0) for r in group]
        avg_weight = sum(weights) / len(weights) if weights else 1.0

        # Use first relationship's entity names (preserve original casing)
        source_entity = group[0]["source_entity"]
        target_entity = group[0]["target_entity"]

        # Decide on description
        if len(descriptions) > 1:
            description = summarize_descriptions(
                descriptions,
                GRAPHRAG_RELATIONSHIP_SUMMARIZE_PROMPT,
                model=model,
                source=source_entity,
                target=target_entity,
            )
            summarized_count += 1
        elif descriptions:
            description = descriptions[0]
        else:
            description = ""

        consolidated.append({
            "source_entity": source_entity,
            "target_entity": target_entity,
            "source_normalized": source_norm,
            "target_normalized": target_norm,
            "relationship_type": rel_type,
            "description": description,
            "weight": avg_weight,
            "source_chunk_ids": source_chunk_ids,
        })

    logger.info(
        f"Consolidated {len(relationships)} relationships -> {len(consolidated)} unique "
        f"({summarized_count} LLM summarizations)"
    )

    return consolidated


# ============================================================================
# Main Entry Point
# ============================================================================

def run_extraction(
    strategy: str = "section",
    overwrite_context: Optional[OverwriteContext] = None,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> dict[str, Any]:
    """Run entity extraction with curated types from graphrag_types.yaml.

    Args:
        strategy: Chunking strategy subfolder (default: "section").
        overwrite_context: Controls whether to skip already-processed books.
        model: LLM model for extraction.

    Returns:
        Dict with extraction results and file paths.
    """
    book_files = load_book_files(strategy)
    logger.info(f"Found {len(book_files)} books")

    extractions_dir = DIR_GRAPH_DATA / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    processed, skipped = [], []

    for book_path in book_files:
        book_name = book_path.stem
        output_path = extractions_dir / f"{book_name}.json"

        if overwrite_context and not overwrite_context.should_overwrite(output_path, logger):
            skipped.append(book_name)
            continue

        results = extract_book(book_path, model=model)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        processed.append(book_name)
        logger.info(f"Saved: {output_path.name}")

    logger.info(f"Extraction complete: {len(processed)} processed, {len(skipped)} skipped")

    # Merge all extractions
    merged = merge_extractions(extractions_dir)

    # Save merged results
    extraction_path = DIR_GRAPH_DATA / "extraction_results.json"
    with open(extraction_path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": merged["entities"],
            "relationships": merged["relationships"],
            "stats": merged["stats"],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved merged results to {extraction_path}")

    merged["extraction_path"] = str(extraction_path)
    merged["extractions_dir"] = str(extractions_dir)
    merged["processed_books"] = processed
    merged["skipped_books"] = skipped

    return merged
