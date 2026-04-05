"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from bhodi_platform.answering.runtime import get_tokenizer
from bhodi_platform.interfaces.tui.chat import _save_log as save_log

__all__ = [
    "count_tokens",
    "fast_tokenize",
    "save_log",
]


def fast_tokenize(texts, max_length: int = 1024):
    """
    Tokenizes the input texts quickly using the configured tokenizer.

    Args:
        texts (str or List[str]): The text or list of texts to tokenize.
        max_length (int, optional): Maximum length of the tokenized sequence. Defaults to 1024.

    Returns:
        Tensor: The tokenized output, padded and truncated to the longest sequence.
    """
    tokenizer = get_tokenizer()
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the provided text.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: The number of tokens in the text.
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
