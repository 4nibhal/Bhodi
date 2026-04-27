"""Unit tests for Bhodi CLI entry points."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from bhodi_platform.interfaces.cli.main import main as cli_main


class TestCliMain:
    """Tests for the main CLI dispatcher."""

    def test_health_command(self, capsys):
        """bhodi health should print OK."""
        with patch.object(sys, "argv", ["bhodi", "health"]):
            cli_main()
        captured = capsys.readouterr()
        assert "Health check: OK" in captured.out

    def test_index_help(self, capsys):
        """bhodi index --help should show usage."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["bhodi", "index", "--help"]):
                cli_main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "index" in captured.out

    def test_query_help(self, capsys):
        """bhodi query --help should show usage."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["bhodi", "query", "--help"]):
                cli_main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "query" in captured.out

    def test_no_command_prints_help(self, capsys):
        """Running bhodi without arguments should print help and exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["bhodi"]):
                cli_main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out


class TestCliIndexing:
    """Tests for the indexing CLI command."""

    @patch("bhodi_platform.interfaces.cli.indexing.run_index")
    def test_index_document(self, mock_run_index, capsys):
        """bhodi-index should delegate to run_index."""
        from bhodi_platform.interfaces.cli.indexing import main as index_main

        mock_run_index.return_value = "Indexed 5 chunks from test.pdf"
        index_main(["test.pdf"])
        captured = capsys.readouterr()
        assert "Indexed 5 chunks from test.pdf" in captured.out
        mock_run_index.assert_called_once()

    @patch("bhodi_platform.interfaces.cli.indexing.run_index")
    def test_index_with_options(self, mock_run_index, capsys):
        """bhodi-index should pass chunk-size and overlap."""
        from bhodi_platform.interfaces.cli.indexing import main as index_main

        mock_run_index.return_value = "done"
        index_main(["test.pdf", "--chunk-size", "500", "--overlap", "50"])
        mock_run_index.assert_called_once_with(
            source="test.pdf",
            chunk_size=500,
            overlap=50,
            metadata=None,
        )


class TestCliQuery:
    """Tests for the query CLI command."""

    @patch("bhodi_platform.interfaces.cli.query.run_query")
    def test_query_question(self, mock_run_query, capsys):
        """bhodi query should delegate to run_query."""
        from bhodi_platform.interfaces.cli.query import main as query_main

        mock_run_query.return_value = "Answer: 42"
        query_main(["What is the answer?"])
        captured = capsys.readouterr()
        assert "Answer: 42" in captured.out
        mock_run_query.assert_called_once()

    @patch("bhodi_platform.interfaces.cli.query.run_query")
    def test_query_with_conversation_id(self, mock_run_query, capsys):
        """bhodi query should pass conversation-id."""
        from bhodi_platform.interfaces.cli.query import main as query_main

        mock_run_query.return_value = "Answer: yes"
        query_main(["Is this true?", "--conversation-id", "abc123"])
        mock_run_query.assert_called_once_with(
            question="Is this true?",
            conversation_id="abc123",
            top_k=5,
            temperature=0.7,
        )
