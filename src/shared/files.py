import logging
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# ============================================================================
# OVERWRITE HANDLING
# ============================================================================


class OverwriteMode(Enum):
    """Overwrite behavior for file operations.

    Used with --overwrite CLI argument across all pipeline stages.
    """

    PROMPT = auto()  # Ask interactively (default)
    SKIP = auto()  # Never overwrite, skip existing files
    ALL = auto()  # Overwrite all files without asking


class OverwriteContext:
    """Tracks overwrite decisions across multiple files in a pipeline run.

    Maintains session-level state for "all" and "none" responses so users
    don't have to answer the same question for every file.

    Args:
        mode: Initial overwrite mode from CLI argument.

    Example:
        >>> context = OverwriteContext(OverwriteMode.PROMPT)
        >>> if context.should_overwrite(Path("output.json")):
        ...     write_file(data, "output.json")
    """

    def __init__(self, mode: OverwriteMode = OverwriteMode.PROMPT):
        self.mode = mode
        self._session_decision: Optional[str] = None  # "all" or "none" from user

    def should_overwrite(self, output_path: Path, logger: logging.Logger = None) -> bool:
        """Determine if file should be written (overwriting if exists).

        Args:
            output_path: Target file path to potentially overwrite.
            logger: Optional logger for skip messages.

        Returns:
            True if file should be written, False to skip.
        """
        # File doesn't exist - always write
        if not output_path.exists():
            return True

        # CLI flag: skip all existing files
        if self.mode == OverwriteMode.SKIP:
            if logger:
                logger.info(f"Skipping {output_path.name} (output exists, --overwrite=skip)")
            return False

        # CLI flag: overwrite all files
        if self.mode == OverwriteMode.ALL:
            return True

        # Session decision from previous prompt
        if self._session_decision == "all":
            return True
        if self._session_decision == "none":
            if logger:
                logger.info(f"Skipping {output_path.name} (user chose 'none')")
            return False

        # Interactive prompt (default mode)
        return self._prompt_user(output_path, logger)

    def _prompt_user(self, output_path: Path, logger: logging.Logger = None) -> bool:
        """Prompt user for overwrite decision.

        Options:
            yes  - Overwrite this file only
            all  - Overwrite all remaining files (sets session decision)
            no   - Skip this file only
            none - Skip all remaining files (sets session decision)

        Args:
            output_path: Path to existing file.
            logger: Optional logger for messages.

        Returns:
            True to overwrite, False to skip.
        """
        while True:
            response = input(
                f"\nFile exists: {output_path}\n"
                "Overwrite? [y]es / [a]ll / [n]o / no[ne]: "
            ).lower().strip()

            if response in ("y", "yes"):
                return True
            elif response in ("a", "all"):
                self._session_decision = "all"
                return True
            elif response in ("n", "no"):
                if logger:
                    logger.info(f"Skipping {output_path.name} (user chose 'no')")
                return False
            elif response == "none":
                self._session_decision = "none"
                if logger:
                    logger.info(f"Skipping {output_path.name} (user chose 'none')")
                return False
            else:
                if logger:
                    logger.warning("Invalid response. Enter: yes, all, no, or none")


def parse_overwrite_arg(value: str) -> OverwriteMode:
    """Parse --overwrite CLI argument value to OverwriteMode.

    Args:
        value: One of "prompt", "skip", "all".

    Returns:
        Corresponding OverwriteMode enum value.

    Raises:
        ValueError: If value is not recognized.

    Example:
        >>> mode = parse_overwrite_arg("skip")
        >>> context = OverwriteContext(mode)
    """
    mapping = {
        "prompt": OverwriteMode.PROMPT,
        "skip": OverwriteMode.SKIP,
        "all": OverwriteMode.ALL,
    }
    if value not in mapping:
        raise ValueError(f"Invalid overwrite mode: '{value}'. Use: prompt, skip, all")
    return mapping[value]


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(name: str) -> logging.Logger:
    """Configures a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)

def get_file_list(source_dir: Path, extension: str) -> list[Path]:
    """Recursively finds all files with specific extension."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return list(source_dir.rglob(f"*.{extension}"))

def get_output_path(source_path: Path, source_root: Path, output_root: Path, new_extension: str = None) -> Path:
    """
    Calculates the mirror output path.
    Example: raw/neuroscience/book.pdf -> processed/neuroscience/book.md
    """
    relative_path = source_path.relative_to(source_root)
    destination = output_root / relative_path
    
    if new_extension:
        destination = destination.with_suffix(new_extension)
        
    return destination
