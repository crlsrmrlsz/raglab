"""Stage 3: NLP segmentation of cleaned markdown files."""

import argparse
import json
from pathlib import Path

from src.config import DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS
from src.shared import (
    setup_logging,
    get_file_list,
    get_output_path,
    OverwriteContext,
    parse_overwrite_arg,
)
from src.content_preparation.segmentation import segment_document

logger = setup_logging("Stage3_Segmentation")


def main():
    """Run NLP segmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 3: NLP segmentation of cleaned markdown files"
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

    logger.info("Starting Stage 3: NLP Segmentation")

    # Find cleaned markdown files
    input_files = get_file_list(DIR_DEBUG_CLEAN, "md")
    logger.info(f"Found {len(input_files)} cleaned Markdown files.")

    if not input_files:
        logger.warning(f"No files found in {DIR_DEBUG_CLEAN}. Run Stage 2 first.")
        return

    success_count = 0
    skipped_count = 0
    for md_path in input_files:
        # Check overwrite decision (check JSON output)
        json_path = get_output_path(md_path, DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS, ".json")
        if not overwrite_context.should_overwrite(json_path, logger):
            skipped_count += 1
            continue

        logger.info(f"Processing: {md_path.name}")

        # Read and segment
        cleaned_text = md_path.read_text(encoding="utf-8")
        book_name = md_path.stem.replace("_debug", "")

        chunks = segment_document(cleaned_text, book_name)

        # Save JSON output
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # Save Markdown output (human readable)
        md_out_path = get_output_path(md_path, DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS)

        md_lines = [f"# Analyzed Content: {book_name}\n"]
        for i, chunk in enumerate(chunks):
            md_lines.append("---")
            md_lines.append(f"### Chunk {i+1}")
            md_lines.append(f"**Context:** `{chunk['context']}`")
            md_lines.append(f"**Sentences:** {chunk['num_sentences']}")
            for sent in chunk['sentences']:
                md_lines.append(f"- {sent}")
            md_lines.append("\n")

        md_out_path.write_text("\n".join(md_lines), encoding="utf-8")

        success_count += 1
        logger.info(f"Finished {md_path.name} -> {len(chunks)} chunks generated.")

    logger.info(f"Stage 3 complete. {success_count} processed, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
