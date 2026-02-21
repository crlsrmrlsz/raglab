# Utils package for RAGLab

# Import all utility functions to make them available for import
from .files import (
    setup_logging,
    get_file_list,
    get_output_path,
    OverwriteMode,
    OverwriteContext,
    parse_overwrite_arg,
)

# OpenRouter API client
from .openrouter_client import (
    call_chat_completion,
    call_simple_prompt,
    OpenRouterError,
    RateLimitError,
    APIError,
)

# Optional: Import token utilities if they exist
try:
    from .tokens import count_tokens
except ImportError:
    pass
