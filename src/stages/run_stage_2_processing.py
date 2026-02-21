"""Stage 2: Clean and process manually reviewed markdown files."""

import argparse
from pathlib import Path

from src.config import DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN
from src.shared import (
    setup_logging,
    get_file_list,
    get_output_path,
    OverwriteContext,
    parse_overwrite_arg,
)
from src.content_preparation.cleaning import run_structural_cleaning, setup_cleaning_logger

logger = setup_logging("Stage2_Processing")


def main():
    """Run markdown cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Clean and process manually reviewed markdown files"
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite behavior: prompt (default), skip, all",
    )
    args = parser.parse_args()

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    logger.info("Starting Stage 2: Markdown Cleaning")

    cleaning_logger = setup_cleaning_logger()

    # Find reviewed markdown files
    input_files = get_file_list(DIR_MANUAL_REVIEW, "md")
    logger.info(f"Found {len(input_files)} reviewed Markdown files.")

    if not input_files:
        logger.warning(f"No files found in {DIR_MANUAL_REVIEW}. Did you move files after review?")
        return

    success_count = 0
    skipped_count = 0
    for md_path in input_files:
        # Check overwrite decision
        debug_path = get_output_path(md_path, DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN)
        if not overwrite_context.should_overwrite(debug_path, logger):
            skipped_count += 1
            continue

        logger.info(f"Processing: {md_path.name}")

        # Read and clean
        raw_text = md_path.read_text(encoding="utf-8")
        book_name = md_path.stem
        cleaned_text, cleaning_log = run_structural_cleaning(
            raw_text,
            book_name=book_name,
            enable_logging=True
        )

        # Log cleaning report
        if cleaning_log:
            cleaning_logger.info(cleaning_log.generate_report())
            logger.info(
                f"Cleaning changes: {len(cleaning_log.lines_removed)} lines removed, "
                f"{len(cleaning_log.inline_removals)} inline removals, "
                f"{cleaning_log.paragraphs_merged} paragraphs merged"
            )

        # Save cleaned file
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(cleaned_text, encoding="utf-8")

        success_count += 1
        logger.info(f"Finished cleaning {md_path.name}")

    logger.info(f"Stage 2 complete. {success_count} processed, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
