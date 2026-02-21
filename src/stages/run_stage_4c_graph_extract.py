"""Stage 4c: Entity Extraction for GraphRAG.

Extracts entities and relationships from chunked documents using curated
entity types from graphrag_types.yaml. Results are saved to extraction_results.json
for Neo4j upload in Stage 6b.

## Usage

```bash
# Full extraction (default: semantic_std2 chunks)
python -m src.stages.run_stage_4c_graph_extract

# Use different chunking strategy
python -m src.stages.run_stage_4c_graph_extract --strategy section

# Resume after interruption (skip processed books)
python -m src.stages.run_stage_4c_graph_extract --overwrite skip

# List books to be processed
python -m src.stages.run_stage_4c_graph_extract --list-books
```

## Next Step

After extraction, upload to Neo4j:
```bash
python -m src.stages.run_stage_6b_neo4j
```
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.graph.extraction_utils import run_extraction, load_book_files
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.config import GRAPHRAG_EXTRACTION_MODEL, DIR_CLEANING_LOGS, GRAPHRAG_CHUNKING_STRATEGY

logger = setup_logging(__name__)


def setup_file_logging() -> Path:
    """Setup dual console + file logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DIR_CLEANING_LOGS / f"extraction_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)

    return log_file


def main():
    """Run entity extraction stage."""
    parser = argparse.ArgumentParser(
        description="Stage 4c: Extract entities/relationships for GraphRAG"
    )
    parser.add_argument(
        "--strategy", type=str, default=GRAPHRAG_CHUNKING_STRATEGY,
        help=f"Chunking strategy (default: {GRAPHRAG_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--overwrite", type=str, default="prompt",
        choices=["prompt", "skip", "all"],
        help="Overwrite mode: prompt (default), skip existing, or overwrite all",
    )
    parser.add_argument(
        "--model", type=str, default=GRAPHRAG_EXTRACTION_MODEL,
        help=f"LLM model (default: {GRAPHRAG_EXTRACTION_MODEL})",
    )
    parser.add_argument(
        "--list-books", action="store_true",
        help="List books to be processed and exit",
    )

    args = parser.parse_args()

    # List books
    if args.list_books:
        try:
            book_files = load_book_files(args.strategy)
            logger.info(f"=== Books ({len(book_files)}) ===")
            for i, path in enumerate(book_files, 1):
                logger.info(f"  {i:2}. {path.stem}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        return

    # Run extraction
    log_file = setup_file_logging()

    logger.info("=" * 60)
    logger.info("Stage 4c: Entity Extraction")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info(f"Model: {args.model}")

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    try:
        results = run_extraction(
            strategy=args.strategy,
            overwrite_context=overwrite_context,
            model=args.model,
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Resume with: --overwrite skip")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

    # Summary
    logger.info("=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)
    logger.info(f"Books: {len(results['processed_books'])} processed, {len(results['skipped_books'])} skipped")
    logger.info(f"Entities: {results['stats']['total_entities']:,}")
    logger.info(f"Relationships: {results['stats']['total_relationships']:,}")
    logger.info(f"Entity types used: {results['stats']['unique_entity_types']}")
    logger.info(f"Output: {results['extraction_path']}")


if __name__ == "__main__":
    main()
