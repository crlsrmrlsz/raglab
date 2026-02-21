"""
Text cleaning module for markdown documents.
Removes artifacts, consolidates paragraphs, and standardizes formatting.
"""
import re
from typing import Optional
from dataclasses import dataclass, field

from src.config import (
    LINE_REMOVAL_PATTERNS,
    INLINE_REMOVAL_PATTERNS,
    CHARACTER_SUBSTITUTIONS,
    LIST_MARKER_PATTERN,
    TERMINAL_PUNCTUATION,
    SENTENCE_ENDING_PUNCTUATION,
    REPORT_WIDTH,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


# ============================================================================
# LOGGING DATACLASS
# ============================================================================

@dataclass
class CleaningLog:
    """Tracks all cleaning operations for comprehensive reporting."""
    
    book_name: str
    lines_removed: list[tuple[str, str]] = field(default_factory=list)
    inline_removals: list[tuple[str, str]] = field(default_factory=list)
    paragraphs_merged: int = 0
    substitutions: list[tuple[str, int]] = field(default_factory=list)
    list_markers_removed: int = 0

    def log_line_removal(self, pattern_name: str, line: str):
        """Record a removed line."""
        self.lines_removed.append((pattern_name, line.strip()))

    def log_inline_removal(self, pattern_name: str, removed_text: str):
        """Record removed inline text."""
        self.inline_removals.append((pattern_name, removed_text))

    def log_substitution(self, sub_name: str, count: int):
        """Record character substitutions."""
        if count > 0:
            self.substitutions.append((sub_name, count))

    def generate_report(self) -> str:
        """Generate comprehensive cleaning report."""
        report = []
        report.append("=" * REPORT_WIDTH)
        report.append(f"CLEANING REPORT FOR: {self.book_name}")
        report.append("=" * REPORT_WIDTH)
        report.append("")

        # Summary
        report.append("SUMMARY:")
        report.append("-" * REPORT_WIDTH)
        report.append(f"  Lines removed: {len(self.lines_removed)}")
        report.append(f"  Inline removals: {len(self.inline_removals)}")
        report.append(f"  Paragraphs merged: {self.paragraphs_merged}")
        report.append(f"  List markers removed: {self.list_markers_removed}")
        total_subs = sum(count for _, count in self.substitutions)
        report.append(f"  Character substitutions: {total_subs}")
        report.append("")

        # Lines removed details
        if self.lines_removed:
            report.append(f"LINES REMOVED ({len(self.lines_removed)} total):")
            report.append("-" * REPORT_WIDTH)
            
            pattern_groups = {}
            for pattern_name, line in self.lines_removed:
                pattern_groups.setdefault(pattern_name, []).append(line)

            for pattern_name, lines in pattern_groups.items():
                report.append(f"\n  [{pattern_name}] - {len(lines)} lines:")
                for line in lines:
                    report.append(f"    • {line}")
            report.append("")

        # Inline removals details
        if self.inline_removals:
            report.append(f"INLINE REMOVALS ({len(self.inline_removals)} total):")
            report.append("-" * REPORT_WIDTH)
            
            pattern_groups = {}
            for pattern_name, removed_text in self.inline_removals:
                pattern_groups.setdefault(pattern_name, []).append(removed_text)

            for pattern_name, removals in pattern_groups.items():
                report.append(f"\n  [{pattern_name}] - {len(removals)} occurrences:")
                for removed_text in removals:
                    report.append(f"    REMOVED: '{removed_text}'")
            report.append("")

        # Substitutions details
        if self.substitutions:
            report.append("CHARACTER SUBSTITUTIONS:")
            report.append("-" * REPORT_WIDTH)
            for sub_name, count in self.substitutions:
                report.append(f"  [{sub_name}]: {count} replacements")
            report.append("")

        # Paragraph merging
        if self.paragraphs_merged > 0:
            report.append("PARAGRAPH CONSOLIDATION:")
            report.append("-" * REPORT_WIDTH)
            report.append(f"  {self.paragraphs_merged} paragraph breaks removed")
            report.append("")

        # List markers
        if self.list_markers_removed > 0:
            report.append("LIST MARKERS REMOVED:")
            report.append("-" * REPORT_WIDTH)
            report.append(f"  {self.list_markers_removed} list marker lines removed")
            report.append("")

        report.append("=" * REPORT_WIDTH)
        return "\n".join(report)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _should_merge_paragraphs(ends_terminal: bool, starts_lower: bool, ends_connector: bool) -> bool:
    """Determine if consecutive paragraphs should be merged.

    Args:
        ends_terminal: Whether first paragraph ends with terminal punctuation.
        starts_lower: Whether second paragraph starts with lowercase.
        ends_connector: Whether first paragraph ends with comma or hyphen.

    Returns:
        True if paragraphs should be merged.
    """
    return (not ends_terminal) or starts_lower or ends_connector


# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def remove_artifact_lines(text: str, log: Optional[CleaningLog] = None) -> str:
    """
    Remove entire lines matching artifact patterns.
    Examples: figure captions, learning objectives, single characters.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        should_remove = False
        
        for pattern, pattern_name in LINE_REMOVAL_PATTERNS:
            if re.search(pattern, line):
                if log:
                    log.log_line_removal(pattern_name, line)
                should_remove = True
                break

        if not should_remove:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_list_markers(text: str, log: Optional[CleaningLog] = None) -> str:
    """
    Remove list marker lines like "(a)", "(b)" that follow terminal punctuation.
    Only removes markers that appear after sentence-ending punctuation.
    """
    lines = text.split('\n')
    cleaned_lines = []
    removed_count = 0

    for i, line in enumerate(lines):
        if re.match(LIST_MARKER_PATTERN, line):
            # Check if previous line ends with terminal punctuation
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line and prev_line[-1] in SENTENCE_ENDING_PUNCTUATION:
                    if log:
                        log.log_line_removal('LIST_MARKER', line)
                    removed_count += 1
                    continue

        cleaned_lines.append(line)

    if log and removed_count > 0:
        log.list_markers_removed = removed_count

    return '\n'.join(cleaned_lines)


def clean_inline_content(text: str, log: Optional[CleaningLog] = None) -> str:
    """
    Clean inline formatting: remove inline artifacts and apply substitutions.
    """
    # Character substitutions
    for old, new, sub_name in CHARACTER_SUBSTITUTIONS:
        count = text.count(old)
        if count > 0:
            text = text.replace(old, new)
            if log:
                log.log_substitution(sub_name, count)

    # Inline pattern removals
    for pattern, pattern_name in INLINE_REMOVAL_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches and log:
            for match in matches:
                log.log_inline_removal(pattern_name, match.group())
        text = re.sub(pattern, '', text)

    # Fix hyphenated words with spaces: "mu- opioid" -> "mu-opioid"
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)

    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)

    return text.strip()


def consolidate_paragraphs(paragraphs: list[str], log: Optional[CleaningLog] = None) -> list[str]:
    """
    Merge paragraphs that were incorrectly split by formatting errors.
    
    Merge conditions:
    - P1 doesn't end with terminal punctuation AND P2 exists
    - P2 starts with lowercase
    - P1 ends with comma or hyphen
    """
    if not paragraphs:
        return []

    consolidated = []
    buffer = paragraphs[0].strip()
    merge_count = 0

    for i in range(1, len(paragraphs)):
        current = paragraphs[i].strip()
        if not current:
            continue

        ends_terminal = buffer.endswith(TERMINAL_PUNCTUATION)
        starts_lower = current[0].islower() if current else False
        ends_connector = buffer.endswith((',', '-'))

        if _should_merge_paragraphs(ends_terminal, starts_lower, ends_connector):
            buffer += " " + current
            merge_count += 1
        else:
            consolidated.append(buffer)
            buffer = current

    if buffer:
        consolidated.append(buffer)

    if log:
        log.paragraphs_merged += merge_count

    return consolidated


# ============================================================================
# MAIN CLEANING ORCHESTRATION
# ============================================================================

def run_structural_cleaning(
    md_content: str,
    book_name: str = "Unknown",
    enable_logging: bool = True
) -> tuple[str, Optional[CleaningLog]]:
    """
    Complete markdown cleaning pipeline.
    
    Process flow:
    1. Split by headers to preserve structure
    2. Consolidate broken paragraphs within each section
    3. Remove list markers
    4. Remove artifact lines
    5. Clean inline content
    
    Args:
        md_content: Markdown text to clean
        book_name: Book identifier for logging
        enable_logging: Whether to track changes
        
    Returns:
        Tuple of (cleaned_text, cleaning_log)
    """
    log = CleaningLog(book_name=book_name) if enable_logging else None

    # Split by headers to preserve document structure
    sections = re.split(r'(^#+\s+.*$)', md_content, flags=re.MULTILINE)

    # Process each section
    processed_sections = []
    for segment in sections:
        segment = segment.strip()
        if not segment:
            continue

        # Keep headers unchanged
        if segment.startswith('#'):
            processed_sections.append(segment)
            continue

        # Consolidate paragraphs in body text
        raw_paragraphs = segment.split('\n\n')
        consolidated = consolidate_paragraphs(raw_paragraphs, log)
        processed_sections.append('\n\n'.join(consolidated))

    md_content = '\n\n'.join(processed_sections)

    # Apply line-level cleaning
    md_content = remove_list_markers(md_content, log)
    md_content = remove_artifact_lines(md_content, log)

    # Apply inline cleaning
    cleaned_text = clean_inline_content(md_content, log)

    return cleaned_text, log

