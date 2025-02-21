import datetime
from bhodi_doc_analyzer.config import tokenizer

# =============================================================================
# TOKENIZE/INDEXING UTILS
# =============================================================================

def fast_tokenize(texts, max_length: int = 1024):
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# =============================================================================
# LOGGING
# =============================================================================

def save_log(log_text: str) -> None:
    """
    Appends a log entry to 'assistant_logs.txt' with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("assistant_logs.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {log_text}\n")
