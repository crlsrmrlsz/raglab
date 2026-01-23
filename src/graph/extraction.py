"""GraphRAG entity extraction with curated entity types.

## What This Module Does

1. Constrained Extraction: LLM extracts entities using predefined types from graphrag_types.yaml
2. Relationship Extraction: Relationships are extracted with open-ended types (per GraphRAG paper)
3. Save Results: Stores extraction_results.json for Neo4j upload in Stage 6b

## Key Design Decision

Entity types are curated in graphrag_types.yaml rather than auto-discovered.
This provides:
- Consistent taxonomy across extraction runs
- Better control over entity granularity
- Lower indexing cost (no consolidation LLM calls)
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
)
from src.graph.graphrag_types import get_entity_types_string
from src.shared.openrouter_client import call_structured_completion
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


class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction from a chunk."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ============================================================================
# Core Extraction Functions
# ============================================================================

def extract_chunk(
    chunk: dict[str, Any],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> ExtractionResult:
    """Extract entities/relationships from a single chunk using curated types.

    Args:
        chunk: Chunk dict with 'text' field.
        model: LLM model for extraction.

    Returns:
        ExtractionResult with entities and relationships.
    """
    entity_types = get_entity_types_string()

    prompt = GRAPHRAG_CHUNK_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        text=chunk["text"],
        max_entities=GRAPHRAG_MAX_ENTITIES,
        max_relationships=GRAPHRAG_MAX_RELATIONSHIPS,
    )
    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )


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


def merge_extractions(extractions_dir: Path) -> dict[str, Any]:
    """Merge all per-book extraction files into aggregated results.

    Args:
        extractions_dir: Directory containing per-book extraction JSON files.

    Returns:
        Dict with merged entities, relationships, counts, and stats.
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

    stats["total_entities"] = len(all_entities)
    stats["total_relationships"] = len(all_relationships)
    stats["unique_entity_types"] = len(entity_counter)
    stats["unique_relationship_types"] = len(relationship_counter)

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_counter.most_common()),
        "relationship_type_counts": dict(relationship_counter.most_common()),
        "stats": stats,
    }


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
