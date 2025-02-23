"""
Utility functions for Bhodi.

This module provides helper functions for tokenization, token counting,
and logging with timestamps.
"""

import datetime
from bhodi_doc_analyzer.config import tokenizer

# =============================================================================
# TOKENIZE/INDEXING UTILS
# =============================================================================

def fast_tokenize(texts, max_length: int = 1024):
    """
    Tokenizes the input texts quickly using the configured tokenizer.

    Args:
        texts (str or List[str]): The text or list of texts to tokenize.
        max_length (int, optional): Maximum length of the tokenized sequence. Defaults to 1024.

    Returns:
        Tensor: The tokenized output, padded and truncated to the longest sequence.
    """
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the provided text.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenizer.encode(text))

# =============================================================================
# LOGGING
# =============================================================================

def save_log(log_text: str) -> None:
    """
    Appends a log entry to 'assistant_logs.txt' with a timestamp.

    Args:
        log_text (str): The message to log.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("assistant_logs.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {log_text}\n")
