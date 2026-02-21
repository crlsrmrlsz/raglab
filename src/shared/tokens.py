import tiktoken
from src.config import TOKENIZER_MODEL

_encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding.encode(text))