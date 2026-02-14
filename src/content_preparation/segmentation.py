"""NLP segmentation module using spaCy.

Segments markdown documents into structured paragraphs with
sentence-level granularity and context tracking.
"""

import re

import spacy

from src.config import (
    SPACY_MODEL,
    VALID_ENDINGS,
    MIN_SENTENCE_WORDS,
    HEADER_CHAPTER,
    HEADER_SECTION,
    CONTEXT_SEPARATOR,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)

# Module-level NLP model (lazy initialized)
_nlp = None


def _get_nlp():
    """Get or create the spaCy NLP model singleton.

    Returns:
        Loaded spaCy model.

    Raises:
        OSError: If the specified model is not installed.
    """
    global _nlp
    if _nlp is None:
        logger.info(f"Loading spaCy model: {SPACY_MODEL}")
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=["ner"])
        except OSError:
            raise OSError(
                f"Model '{SPACY_MODEL}' not found. "
                f"Install with: python -m spacy download {SPACY_MODEL}"
            )
    return _nlp


def _get_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy.

    Args:
        text: Input text to segment.

    Returns:
        List of sentence strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def _filter_sentences(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Filter out noise sentences (fragments, lowercase starts).

    Args:
        sentences: List of sentence strings.

    Returns:
        Tuple of (kept_sentences, removed_sentences).
    """
    kept = []
    removed = []

    for sent in sentences:
        reason = None
        if len(sent.split()) < MIN_SENTENCE_WORDS:
            reason = "Too short"
        elif sent and sent[0].islower():
            reason = "Starts lowercase"
        elif sent and not sent.endswith(VALID_ENDINGS):
            reason = "No terminal punctuation"

        if reason:
            removed.append(f"[{reason}] {sent}")
        else:
            kept.append(sent)

    return kept, removed


def _build_context_string(book_name: str, chapter: str, section: str = "") -> str:
    """Build hierarchical context string from components.

    Args:
        book_name: Source book identifier.
        chapter: Chapter title.
        section: Optional section title.

    Returns:
        Context string in format "book > chapter" or "book > chapter > section".
    """
    parts = [book_name, chapter]
    if section:
        parts.append(section)
    return CONTEXT_SEPARATOR.join(parts)


def segment_document(clean_text: str, book_name: str) -> list[dict]:
    """Segment document into structured chunks with context.

    Splits text by headers, extracts sentences per paragraph,
    and applies quality filtering.

    Args:
        clean_text: Cleaned markdown text.
        book_name: Identifier for the source book.

    Returns:
        List of chunk dictionaries containing:
        - context: Hierarchical path (book > chapter > section)
        - text: Reconstructed chunk text
        - sentences: List of individual sentences
        - num_sentences: Count of sentences
    """
    sections = re.split(r'(^#+\s+.*$)', clean_text, flags=re.MULTILINE)

    processed_chunks = []
    current_chapter = "Unknown Chapter"
    current_section = ""

    for segment in sections:
        segment = segment.strip()
        if not segment:
            continue

        # Update context from headers
        if segment.startswith('#'):
            clean_header = segment.lstrip('#').strip()
            if segment.startswith(HEADER_CHAPTER):
                current_chapter = clean_header
                current_section = ""
            elif segment.startswith(HEADER_SECTION):
                current_section = clean_header
            continue

        # Process body paragraphs
        paragraphs = segment.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            raw_sentences = _get_sentences(paragraph)
            valid_sentences, _ = _filter_sentences(raw_sentences)

            if not valid_sentences:
                continue

            context_str = _build_context_string(
                book_name, current_chapter, current_section
            )

            chunk_data = {
                "context": context_str,
                "text": " ".join(valid_sentences),
                "sentences": valid_sentences,
                "num_sentences": len(valid_sentences)
            }
            processed_chunks.append(chunk_data)

    return processed_chunks
