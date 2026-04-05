"""Domain services for Bhodi platform.

Stateless policy orchestrators that coordinate domain logic.
These are not the application services but pure domain logic coordinators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bhodi_platform.domain.exceptions import (
    DocumentIntegrityError,
    PolicyViolationError,
)
from bhodi_platform.domain.policies import (
    ContextAssemblyPolicy,
    IndexingPolicy,
    RetrievalPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path


class RetrievalDomainService:
    """Coordinates retrieval-related domain logic.

    This is a stateless service that applies RetrievalPolicy and
    ContextAssemblyPolicy to make business decisions about document
    retrieval and context assembly.
    """

    def __init__(
        self,
        retrieval_policy: RetrievalPolicy | None = None,
        context_policy: ContextAssemblyPolicy | None = None,
    ) -> None:
        self._retrieval_policy = retrieval_policy or RetrievalPolicy()
        self._context_policy = context_policy or ContextAssemblyPolicy()

    @property
    def retrieval_policy(self) -> RetrievalPolicy:
        return self._retrieval_policy

    @property
    def context_policy(self) -> ContextAssemblyPolicy:
        return self._context_policy

    def check_document_integrity(self, page_content: str | None) -> None:
        """Validate that a document has the required fields for citation support."""
        if not isinstance(page_content, str):
            raise DocumentIntegrityError(
                "Retrieved documents must expose string page_content to preserve "
                "content and metadata for future citation support."
            )

    def decide_summarization(self, token_count: int) -> bool:
        """Decide whether a document should be summarized."""
        return self._retrieval_policy.should_summarize(token_count)

    def compute_context_budget(
        self,
        total_used_tokens: int,
    ) -> int:
        """Compute available token budget for context assembly."""
        return self._context_policy.compute_available_tokens(total_used_tokens)

    def validate_assembly_result(
        self,
        assembled_tokens: int,
        assembled_content: str,
    ) -> None:
        """Validate that assembled context is within policy bounds."""
        if assembled_tokens > self._context_policy.context_token_limit:
            raise PolicyViolationError(
                f"Assembled context ({assembled_tokens} tokens) exceeds "
                f"policy limit ({self._context_policy.context_token_limit} tokens)"
            )
        if not assembled_content:
            raise PolicyViolationError("Assembled context cannot be empty")


class IndexingDomainService:
    """Coordinates indexing-related domain logic.

    This is a stateless service that applies IndexingPolicy to make
    business decisions about document indexing.
    """

    def __init__(
        self,
        indexing_policy: IndexingPolicy | None = None,
    ) -> None:
        self._indexing_policy = indexing_policy or IndexingPolicy()

    @property
    def indexing_policy(self) -> IndexingPolicy:
        return self._indexing_policy

    def validate_path(self, path: str | Path) -> None:
        """Validate that a document path meets indexing policy requirements.

        Raises:
            PolicyViolationError: If the path does not meet policy requirements.
        """
        path_str = str(path)
        if not self._indexing_policy.is_valid_path(path_str):
            raise PolicyViolationError(
                f"Document path does not meet indexing policy requirements: {path_str}. "
                f"Allowed extensions: {self._indexing_policy.allowed_extensions}"
            )

    def validate_file_size(self, path: str | Path) -> None:
        """Validate that a file is within the allowed size limit.

        Raises:
            PolicyViolationError: If the file exceeds size limits.
        """
        path_str = str(path)
        if not self._indexing_policy.validate_file_size(path_str):
            raise PolicyViolationError(
                f"File exceeds maximum size limit of "
                f"{self._indexing_policy.max_file_size_mb}MB: {path_str}"
            )

    def validate_index_request(self, path: str | Path) -> None:
        """Perform all indexing policy validations for a path.

        Raises:
            PolicyViolationError: If any policy requirement is not met.
        """
        self.validate_path(path)
        self.validate_file_size(path)
