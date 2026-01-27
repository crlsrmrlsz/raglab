"""PDF extraction module using Docling.

Provides PDF-to-markdown conversion with artifact removal
(captions, footnotes, tables, pictures).
"""

from pathlib import Path
from typing import Optional

from docling.datamodel.document import InputFormat, DocItemLabel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

from src.shared import setup_logging

logger = setup_logging(__name__)

# Module-level converter (lazy initialized)
_converter: Optional[DocumentConverter] = None


def _get_converter() -> DocumentConverter:
    """Get or create the document converter singleton.

    Returns:
        Configured DocumentConverter instance.
    """
    global _converter
    if _converter is None:
        logger.info("Initializing Docling converter...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    return _converter


def _get_all_descendants(item) -> list:
    """Recursively collect all children of a document item.

    Args:
        item: Document item to traverse.

    Returns:
        List of all descendant items.
    """
    descendants = []
    if hasattr(item, "children") and item.children:
        for child in item.children:
            descendants.append(child)
            descendants.extend(_get_all_descendants(child))
    return descendants


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF to markdown.

    Removes captions, footnotes, tables, page headers/footers,
    and pictures with their children.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        Extracted text as markdown string.

    Raises:
        Exception: If PDF conversion fails.
    """
    converter = _get_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    # First removal: captions, footnotes, headers, footers, tables
    labels_to_remove = {
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.TABLE
    }

    items_to_remove = [
        item for item, level in doc.iterate_items()
        if hasattr(item, "label") and item.label in labels_to_remove
    ]

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)

    # Second removal: pictures and all children
    items_to_remove = []
    seen_ids = set()

    for item, level in doc.iterate_items():
        if hasattr(item, "label") and item.label == DocItemLabel.PICTURE:
            if id(item) not in seen_ids:
                items_to_remove.append(item)
                seen_ids.add(id(item))

            for child in _get_all_descendants(item):
                if id(child) not in seen_ids:
                    items_to_remove.append(child)
                    seen_ids.add(id(child))

    if items_to_remove:
        logger.info(f"Removing {len(items_to_remove)} picture items")
        doc.delete_items(node_items=items_to_remove)

    return doc.export_to_markdown()
