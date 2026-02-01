"""Stage 6b: Neo4j Upload and Leiden Community Detection.

## RAG Theory: Knowledge Graph Construction

This stage uploads extracted entities and relationships to Neo4j,
then runs Leiden community detection to identify clusters of
related entities. Community summaries enable global queries.

## Pipeline Phases

1. **Upload**: Load entities/relationships from extraction_results.json → Neo4j
2. **Leiden**: Run Leiden algorithm → assign community IDs + compute PageRank
3. **Summaries**: Generate community summaries → upload to Weaviate
4. **Entities**: Embed entity descriptions → upload to Weaviate

## Crash-Proof Design

- **Deterministic Leiden**: Uses seed=42 + concurrency=1 (same results on re-run)
- **Checkpoint**: Hierarchy saved to leiden_checkpoint.json for resume
- **Atomic uploads**: Each community uploaded to Weaviate immediately
- **Resume mode**: Skips existing communities in Weaviate

## Usage

```bash
# Full pipeline (all 4 phases)
python -m src.stages.run_stage_6b_neo4j

# Resume from Leiden (graph already uploaded)
python -m src.stages.run_stage_6b_neo4j --from leiden

# Resume from summaries (Leiden done, regenerate summaries)
python -m src.stages.run_stage_6b_neo4j --from summaries

# Resume from embeddings only (communities done, just upload entity embeddings)
python -m src.stages.run_stage_6b_neo4j --from embeddings

# Skip entity embeddings (faster if already done)
python -m src.stages.run_stage_6b_neo4j --skip-entity-embeddings

# Clear graph and start fresh
python -m src.stages.run_stage_6b_neo4j --clear
```
"""

import argparse
import json
import time
from pathlib import Path

from src.shared.files import setup_logging
from src.config import (
    DIR_GRAPH_DATA,
    GRAPHRAG_SUMMARY_MODEL,
    get_entity_collection_name,
)
from src.graph.neo4j_client import (
    get_driver,
    get_gds_client,
    verify_connection,
    clear_graph,
    upload_extraction_results,
    get_graph_stats,
)
from src.graph.community import (
    detect_and_summarize_communities,
    get_community_ids_from_neo4j,
)

logger = setup_logging(__name__)


def load_extraction_results(
    input_name: str = "extraction_results.json",
) -> dict:
    """Load extraction results from JSON file.

    Args:
        input_name: Input filename.

    Returns:
        Dict with entities and relationships.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    input_path = DIR_GRAPH_DATA / input_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Extraction results not found: {input_path}\n"
            "Run Stage 4c first: python -m src.stages.run_stage_4c_graph_extract"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_entity_embeddings_to_weaviate(driver, chunk_size: int = 5000) -> int:
    """Upload entity description embeddings to Weaviate for embedding-based extraction.

    Uses streaming chunks to avoid memory exhaustion when processing large entity
    counts. Each chunk is embedded and uploaded before the next is loaded.

    Args:
        driver: Neo4j driver for reading entities.
        chunk_size: Number of entities to process per chunk (default 5000).

    Returns:
        Number of entities uploaded.
    """
    from src.rag_pipeline.embedder import embed_texts
    from src.rag_pipeline.indexing.weaviate_client import (
        get_client as get_weaviate_client,
        create_entity_collection,
        upload_entity_descriptions,
        get_entity_collection_count,
        delete_collection,
    )

    collection_name = get_entity_collection_name()
    logger.info(f"Entity collection: {collection_name}")

    # Get total count first
    count_query = """
    MATCH (e:Entity)
    WHERE e.description IS NOT NULL AND e.description <> ''
    RETURN count(e) as total
    """
    count_result = driver.execute_query(count_query)
    total_entities = count_result.records[0]["total"]

    if total_entities == 0:
        logger.warning("No entities with descriptions found in Neo4j")
        return 0

    logger.info(f"Found {total_entities} entities with descriptions")
    logger.info(f"Processing in chunks of {chunk_size} to manage memory")

    # Prepare Weaviate collection
    client = get_weaviate_client()

    try:
        # Delete existing collection and recreate
        if client.collections.exists(collection_name):
            logger.info(f"Deleting existing collection {collection_name}")
            delete_collection(client, collection_name)

        logger.info(f"Creating collection {collection_name}")
        create_entity_collection(client, collection_name)

        # Process entities in streaming chunks
        total_uploaded = 0
        offset = 0
        chunk_num = 0
        overall_start = time.time()

        while offset < total_entities:
            chunk_num += 1
            chunk_start = time.time()

            # Fetch chunk from Neo4j
            chunk_query = """
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND e.description <> ''
            RETURN
                e.name as entity_name,
                e.normalized_name as normalized_name,
                e.entity_type as entity_type,
                e.description as description
            ORDER BY e.normalized_name
            SKIP $offset LIMIT $limit
            """
            result = driver.execute_query(
                chunk_query,
                parameters_={"offset": offset, "limit": chunk_size}
            )
            chunk_entities = [dict(record) for record in result.records]

            if not chunk_entities:
                break

            # Embed this chunk
            descriptions = [e["description"] for e in chunk_entities]
            embeddings = embed_texts(descriptions)

            # Add embeddings to entities
            for entity, embedding in zip(chunk_entities, embeddings):
                entity["embedding"] = embedding

            # Upload this chunk to Weaviate
            uploaded = upload_entity_descriptions(client, collection_name, chunk_entities)
            total_uploaded += uploaded

            chunk_time = time.time() - chunk_start
            logger.info(
                f"Chunk {chunk_num}: {uploaded} entities "
                f"({offset + 1}-{offset + len(chunk_entities)} of {total_entities}) "
                f"in {chunk_time:.1f}s"
            )

            # Release memory before next chunk
            del chunk_entities, descriptions, embeddings
            offset += chunk_size

        overall_time = time.time() - overall_start
        logger.info(f"Upload complete in {overall_time:.1f}s")
        logger.info(f"  Total uploaded: {total_uploaded} entities")

        # Verify
        final_count = get_entity_collection_count(client, collection_name)
        logger.info(f"  Verified: {final_count} entities in collection")

        return total_uploaded

    finally:
        client.close()


def main():
    """Run Neo4j upload and Leiden community detection."""
    parser = argparse.ArgumentParser(
        description="Stage 6b: Upload to Neo4j and run Leiden community detection"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=["upload", "leiden", "summaries", "embeddings"],
        default="upload",
        help="Start from phase: upload (default), leiden, summaries, or embeddings",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing graph before upload",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GRAPHRAG_SUMMARY_MODEL,
        help=f"LLM model for community summarization (default: {GRAPHRAG_SUMMARY_MODEL})",
    )
    parser.add_argument(
        "--skip-entity-embeddings",
        action="store_true",
        help="Skip entity embedding upload to Weaviate",
    )

    args = parser.parse_args()

    # Determine which phases to run
    run_upload = args.start_from == "upload"
    run_leiden = args.start_from in ("upload", "leiden")
    skip_leiden = args.start_from == "summaries"
    skip_to_embeddings = args.start_from == "embeddings"

    logger.info("=" * 60)
    logger.info("STAGE 6b: NEO4J UPLOAD + LEIDEN COMMUNITIES")
    logger.info("=" * 60)
    logger.info(f"Starting from: {args.start_from}")

    start_time = time.time()

    # Connect to Neo4j
    try:
        driver = get_driver()
        verify_connection(driver)
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Ensure Neo4j is running: docker compose up -d neo4j")
        raise

    try:
        # Validate prerequisites for non-upload starts
        if not run_upload:
            logger.info("-" * 60)
            logger.info("VALIDATING PREREQUISITES")
            logger.info("-" * 60)

            stats = get_graph_stats(driver)
            if stats["node_count"] == 0:
                raise ValueError(
                    f"Cannot start from {args.start_from}: no entities in Neo4j. "
                    "Run full pipeline first: python -m src.stages.run_stage_6b_neo4j"
                )

            if skip_leiden and not skip_to_embeddings:
                # Check community_ids exist for summaries-only mode
                community_ids = get_community_ids_from_neo4j(driver)
                if not community_ids:
                    raise ValueError(
                        "Cannot start from summaries: no community_ids in Neo4j. "
                        "Run with --from leiden first."
                    )
                logger.info(f"Found {len(community_ids)} existing communities")

            logger.info(f"Validation passed: {stats['node_count']} entities")

        # Phase 1: Upload to Neo4j
        if run_upload and not skip_to_embeddings:
            logger.info("-" * 60)
            logger.info("PHASE 1: UPLOAD TO NEO4J")
            logger.info("-" * 60)

            # Load extraction results
            results = load_extraction_results()
            logger.info(
                f"Loaded {len(results['entities'])} entities, "
                f"{len(results['relationships'])} relationships"
            )

            # Clear graph if requested
            if args.clear:
                clear_graph(driver)

            # Upload
            upload_start = time.time()
            counts = upload_extraction_results(driver, results)
            upload_time = time.time() - upload_start

            logger.info(f"Upload complete in {upload_time:.1f}s")
            logger.info(f"  Entities: {counts['entity_count']}")
            logger.info(f"  Relationships: {counts['relationship_count']}")

        # Get graph stats
        stats = get_graph_stats(driver)
        logger.info(f"Graph stats: {stats['node_count']} nodes, {stats['relationship_count']} relationships")

        # Phase 2 & 3: Leiden + Summaries (skip if going directly to embeddings)
        if not skip_to_embeddings:
            logger.info("-" * 60)
            if skip_leiden:
                logger.info("PHASE 3: COMMUNITY SUMMARIES (using checkpoint)")
            else:
                logger.info("PHASE 2 & 3: LEIDEN + COMMUNITY SUMMARIES")
            logger.info("-" * 60)

            if stats["node_count"] == 0:
                logger.warning("Graph is empty, skipping Leiden")
            else:
                # Get GDS client
                gds = get_gds_client(driver)

                # Run Leiden and generate summaries
                leiden_start = time.time()
                communities = detect_and_summarize_communities(
                    driver,
                    gds,
                    model=args.model,
                    resume=skip_leiden,
                    skip_leiden=skip_leiden,
                )
                leiden_time = time.time() - leiden_start

                logger.info(f"Complete in {leiden_time:.1f}s")
                logger.info(f"  Communities: {len(communities)}")
                logger.info(f"  Total members: {sum(c.member_count for c in communities)}")

        # Phase 4: Entity Embedding Upload
        if not args.skip_entity_embeddings:
            logger.info("-" * 60)
            logger.info("PHASE 4: ENTITY EMBEDDINGS")
            logger.info("-" * 60)

            upload_entity_embeddings_to_weaviate(driver)

        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("STAGE 6b COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f}s")

        # Print final graph stats
        final_stats = get_graph_stats(driver)
        logger.info(f"Final graph: {final_stats['node_count']} nodes, {final_stats['relationship_count']} relationships")

        if final_stats.get("entity_types"):
            logger.info("Entity types:")
            for etype, count in list(final_stats["entity_types"].items())[:5]:
                logger.info(f"  {etype}: {count}")

    finally:
        driver.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    main()
