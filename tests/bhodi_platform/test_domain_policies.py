"""Tests for domain policies."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from bhodi_platform.domain import (
    ContextAssemblyPolicy,
    GenerationPolicy,
    IndexingPolicy,
    RetrievalPolicy,
)
from bhodi_platform.domain.exceptions import PolicyViolationError


class RetrievalPolicyTest(TestCase):
    def test_retrieval_policy_defaults(self) -> None:
        policy = RetrievalPolicy()
        self.assertEqual(policy.reranker_max_length, 512)
        self.assertEqual(policy.document_summary_token_limit, 300)

    def test_should_summarize(self) -> None:
        policy = RetrievalPolicy(document_summary_token_limit=100)
        self.assertFalse(policy.should_summarize(50))
        self.assertFalse(policy.should_summarize(100))
        self.assertTrue(policy.should_summarize(101))

    def test_retrieval_policy_is_frozen(self) -> None:
        policy = RetrievalPolicy()
        with self.assertRaises(AttributeError):
            policy.reranker_max_length = 1024


class GenerationPolicyTest(TestCase):
    def test_generation_policy_defaults(self) -> None:
        policy = GenerationPolicy()
        self.assertEqual(policy.prompt_summary_token_limit, 1200)
        self.assertEqual(policy.raw_summary_char_limit, 2500)
        self.assertIsNotNone(policy.role_mapping)

    def test_map_role_user(self) -> None:
        policy = GenerationPolicy()
        self.assertEqual(policy.map_role("question"), "user")
        self.assertEqual(policy.map_role("human"), "user")
        self.assertEqual(policy.map_role("user"), "user")

    def test_map_role_assistant(self) -> None:
        policy = GenerationPolicy()
        self.assertEqual(policy.map_role("answer"), "assistant")
        self.assertEqual(policy.map_role("assistant"), "assistant")
        self.assertEqual(policy.map_role("ai"), "assistant")

    def test_map_role_unknown_defaults_to_user(self) -> None:
        policy = GenerationPolicy()
        self.assertEqual(policy.map_role("unknown"), "user")

    def test_should_summarize_prompt(self) -> None:
        policy = GenerationPolicy(prompt_summary_token_limit=100)
        self.assertFalse(policy.should_summarize_prompt(50))
        self.assertFalse(policy.should_summarize_prompt(100))
        self.assertTrue(policy.should_summarize_prompt(101))

    def test_should_summarize_text_by_char_count(self) -> None:
        policy = GenerationPolicy(raw_summary_char_limit=1000)
        self.assertTrue(policy.should_summarize_text(500))
        self.assertFalse(policy.should_summarize_text(1000))  # < not <=
        self.assertFalse(policy.should_summarize_text(1001))

    def test_generation_policy_is_frozen(self) -> None:
        policy = GenerationPolicy()
        with self.assertRaises(AttributeError):
            policy.prompt_summary_token_limit = 2000


class ContextAssemblyPolicyTest(TestCase):
    def test_context_assembly_policy_defaults(self) -> None:
        policy = ContextAssemblyPolicy()
        self.assertEqual(policy.context_token_limit, 2000)
        self.assertEqual(policy.document_separator, "\n")

    def test_compute_available_tokens(self) -> None:
        policy = ContextAssemblyPolicy(context_token_limit=1000)
        self.assertEqual(policy.compute_available_tokens(0), 1000)
        self.assertEqual(policy.compute_available_tokens(300), 700)
        self.assertEqual(policy.compute_available_tokens(1000), 0)
        self.assertEqual(policy.compute_available_tokens(1500), 0)

    def test_compute_available_tokens_custom_max(self) -> None:
        policy = ContextAssemblyPolicy(context_token_limit=2000)
        self.assertEqual(policy.compute_available_tokens(500, max_tokens=800), 300)

    def test_should_truncate(self) -> None:
        policy = ContextAssemblyPolicy(context_token_limit=1000)
        self.assertFalse(policy.should_truncate(500))
        self.assertFalse(policy.should_truncate(1000))
        self.assertTrue(policy.should_truncate(1001))

    def test_context_assembly_policy_is_frozen(self) -> None:
        policy = ContextAssemblyPolicy()
        with self.assertRaises(AttributeError):
            policy.context_token_limit = 5000


class IndexingPolicyTest(TestCase):
    def test_indexing_policy_defaults(self) -> None:
        policy = IndexingPolicy()
        self.assertIn(".txt", policy.allowed_extensions)
        self.assertIn(".md", policy.allowed_extensions)
        self.assertEqual(policy.max_file_size_mb, 100)
        self.assertTrue(policy.require_absolute_path)

    def test_is_valid_path_with_valid_extension(self) -> None:
        policy = IndexingPolicy(allowed_extensions=(".txt", ".md"))
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.touch()
            self.assertTrue(policy.is_valid_path(str(path)))

    def test_is_valid_path_with_invalid_extension(self) -> None:
        policy = IndexingPolicy(allowed_extensions=(".txt", ".md"))
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.exe"
            path.touch()
            self.assertFalse(policy.is_valid_path(str(path)))

    def test_is_valid_path_nonexistent(self) -> None:
        policy = IndexingPolicy()
        self.assertFalse(policy.is_valid_path("/nonexistent/path/file.txt"))

    def test_is_valid_path_relative_when_required(self) -> None:
        policy = IndexingPolicy(require_absolute_path=True)
        self.assertFalse(policy.is_valid_path("relative/path/file.txt"))

    def test_is_valid_path_relative_when_not_required(self) -> None:
        policy = IndexingPolicy(require_absolute_path=False)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.touch()
            # Path must exist and have valid extension when relative paths allowed
            self.assertTrue(policy.is_valid_path(str(path)))

    def test_validate_file_size_within_limit(self) -> None:
        policy = IndexingPolicy(max_file_size_mb=100)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("small content")
            self.assertTrue(policy.validate_file_size(str(path)))

    def test_validate_file_size_exceeds_limit(self) -> None:
        policy = IndexingPolicy(max_file_size_mb=1)  # 1MB limit
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            # Create a file larger than 1MB (write 2MB of data)
            path.write_bytes(b"x" * (2 * 1024 * 1024))
            self.assertFalse(policy.validate_file_size(str(path)))

    def test_validate_file_size_directory(self) -> None:
        policy = IndexingPolicy(max_file_size_mb=1)
        with TemporaryDirectory() as tmpdir:
            self.assertTrue(policy.validate_file_size(tmpdir))

    def test_indexing_policy_is_frozen(self) -> None:
        policy = IndexingPolicy()
        with self.assertRaises(AttributeError):
            policy.max_file_size_mb = 200
