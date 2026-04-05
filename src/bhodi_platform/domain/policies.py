"""Domain policies for Bhodi platform.

These encapsulate business rules extracted from DefaultRetrievalCollaborator
and other service logic. Policies are stateless and contain pure business logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Policy governing document retrieval behavior.

    Encapsulates rules for reranking, summarization thresholds,
    and truncation decisions.
    """

    reranker_max_length: int = 512
    document_summary_token_limit: int = 300
    reranker_score_threshold: float | None = None

    def should_summarize(self, token_count: int) -> bool:
        """Determine if a document should be summarized based on token count."""
        return token_count > self.document_summary_token_limit

    def get_reranker_max_length(self) -> int:
        """Get the max length parameter for reranker calls."""
        return self.reranker_max_length


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Policy governing answer generation behavior.

    Encapsulates rules for role mapping, prompt truncation,
    and summary thresholds.
    """

    role_mapping: dict[str, str] | None = None
    prompt_summary_token_limit: int = 1200
    raw_summary_char_limit: int = 2500
    summarizer_max_length: int = 1500
    summarizer_min_length: int = 500

    def __post_init__(self) -> None:
        if self.role_mapping is None:
            object.__setattr__(
                self,
                "role_mapping",
                {
                    "question": "user",
                    "human": "user",
                    "user": "user",
                    "answer": "assistant",
                    "assistant": "assistant",
                    "ai": "assistant",
                },
            )

    def map_role(self, role: str) -> str:
        """Map a legacy role to a model-native role."""
        return self.role_mapping.get(role, "user")  # type: ignore[arg-type]

    def should_summarize_prompt(self, token_count: int) -> bool:
        """Determine if the prompt should be summarized based on token count."""
        return token_count > self.prompt_summary_token_limit

    def should_summarize_text(self, char_count: int) -> bool:
        """Determine if text should be summarized based on character count."""
        return char_count < self.raw_summary_char_limit


@dataclass(frozen=True, slots=True)
class ContextAssemblyPolicy:
    """Policy governing how context is assembled from retrieved documents.

    Encapsulates token budget rules for context window management.
    """

    context_token_limit: int = 2000
    document_separator: str = "\n"

    def compute_available_tokens(
        self,
        total_used: int,
        max_tokens: int | None = None,
    ) -> int:
        """Compute remaining tokens available after accounting for used tokens."""
        effective_max = (
            max_tokens if max_tokens is not None else self.context_token_limit
        )
        return max(0, effective_max - total_used)

    def should_truncate(self, token_count: int) -> bool:
        """Determine if content should be truncated based on token count."""
        return token_count > self.context_token_limit


@dataclass(frozen=True, slots=True)
class IndexingPolicy:
    """Policy governing document indexing behavior.

    Encapsulates rules for path validation and indexing constraints.
    """

    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".doc", ".docx")
    max_file_size_mb: int = 100
    require_absolute_path: bool = True

    def is_valid_path(self, path: str) -> bool:
        """Validate that a document path meets indexing requirements.

        For directories, only checks existence and absolute path requirement.
        For files, also checks the allowed extensions.
        """
        from pathlib import Path

        p = Path(path)
        if self.require_absolute_path and not p.is_absolute():
            return False
        if not p.exists():
            return False
        # Directories are always valid (they contain files, not single file)
        if p.is_dir():
            return True
        if p.suffix.lower() not in self.allowed_extensions:
            return False
        return True

    def validate_file_size(self, path: str) -> bool:
        """Check if a file is within the allowed size limit."""
        from pathlib import Path

        p = Path(path)
        if not p.is_file():
            return True  # Directories don't have a single file size
        size_mb = p.stat().st_size / (1024 * 1024)
        return size_mb <= self.max_file_size_mb
