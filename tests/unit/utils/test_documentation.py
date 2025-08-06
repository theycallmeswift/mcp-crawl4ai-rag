"""
Unit tests for documentation processing functions.
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.utils.documentation import (
    discover_documentation_files,
    process_document_files,
    process_repository_docs,
)


class TestDiscoverDocumentationFiles:
    """Tests for discover_documentation_files function."""

    def test_discover_documentation_files_basic(self, temp_repo_dir):
        """Test basic file discovery with valid markdown and text files."""
        doc_files = discover_documentation_files(temp_repo_dir)

        relative_paths = [str(f.relative_to(temp_repo_dir)) for f in doc_files]
        assert "README.md" in relative_paths
        assert "LICENSE.txt" in relative_paths
        assert "docs/api_reference.md" in relative_paths
        assert "docs/getting_started.md" in relative_paths
        assert "docs/changelog.rst" in relative_paths
        assert "large_file.md" not in relative_paths

    def test_discover_documentation_files_excludes_directories(self, temp_repo_dir):
        """Test that excluded directories are properly filtered out."""
        doc_files = discover_documentation_files(temp_repo_dir)

        relative_paths = [str(f.relative_to(temp_repo_dir)) for f in doc_files]
        assert "tests/test_file.md" not in relative_paths
        assert not any("__pycache__" in path for path in relative_paths)
        assert not any("node_modules" in path for path in relative_paths)
        assert not any("build" in path for path in relative_paths)

    def test_discover_documentation_files_includes_all_extensions(self, temp_repo_dir):
        """Test discovery of .md, .rst, .txt, .mdx, .ipynb files."""
        # Setup
        (temp_repo_dir / "test.mdx").write_text("# MDX file")
        notebook_content = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        (temp_repo_dir / "notebook.ipynb").write_text(json.dumps(notebook_content))

        # Exercise
        doc_files = discover_documentation_files(temp_repo_dir)

        # Verify
        relative_paths = [str(f.relative_to(temp_repo_dir)) for f in doc_files]
        assert any(path.endswith(".md") for path in relative_paths)
        assert any(path.endswith(".rst") for path in relative_paths)
        assert any(path.endswith(".txt") for path in relative_paths)
        assert "test.mdx" in relative_paths
        assert "notebook.ipynb" in relative_paths

    def test_discover_documentation_files_excludes_large_files(self, temp_repo_dir):
        """Test that files larger than 500KB are excluded."""
        doc_files = discover_documentation_files(temp_repo_dir)

        relative_paths = [str(f.relative_to(temp_repo_dir)) for f in doc_files]
        assert "large_file.md" not in relative_paths
        assert "README.md" in relative_paths

    def test_discover_documentation_files_handles_missing_path(self):
        """Test graceful handling when repository path doesn't exist."""
        # Setup
        non_existent_path = Path("/non/existent/path")

        # Exercise
        doc_files = discover_documentation_files(non_existent_path)

        # Verify
        assert doc_files == []

    def test_discover_documentation_files_handles_permission_errors(self, mocker):
        """Test handling of files that can't be accessed."""
        # Setup
        mock_walk = mocker.patch("os.walk")
        mock_walk.side_effect = PermissionError("Permission denied")

        # Exercise & Verify
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_files = discover_documentation_files(Path(temp_dir))
            assert doc_files == []


class TestProcessDocumentFiles:
    """Tests for process_document_files function."""

    def test_process_document_files_basic_markdown(self, temp_repo_dir):
        """Test processing regular markdown files."""
        # Setup
        doc_files = [
            temp_repo_dir / "README.md",
            temp_repo_dir / "docs" / "api_reference.md",
        ]

        # Exercise
        result = process_document_files(doc_files, temp_repo_dir)

        # Verify
        assert len(result) == 2
        readme_doc = next(doc for doc in result if doc["url"] == "README.md")
        assert "# Test Repository" in readme_doc["markdown"]
        api_doc = next(doc for doc in result if doc["url"] == "docs/api_reference.md")
        assert "# API Reference" in api_doc["markdown"]

    def test_process_document_files_jupyter_notebook(
        self, temp_repo_dir, sample_notebook_content
    ):
        """Test processing Jupyter notebooks."""
        # Setup
        notebook_path = temp_repo_dir / "test_notebook.ipynb"
        notebook_path.write_text(json.dumps(sample_notebook_content))
        doc_files = [notebook_path]

        # Exercise
        result = process_document_files(doc_files, temp_repo_dir)

        # Verify
        assert len(result) == 1
        assert result[0]["url"] == "test_notebook.ipynb"
        assert "# Sample Notebook" in result[0]["markdown"]
        assert "```python" in result[0]["markdown"]
        assert "import pandas as pd" in result[0]["markdown"]

    def test_process_document_files_mixed_file_types(self, temp_repo_dir):
        """Test processing a mix of .md, .rst, .txt files."""
        # Setup
        doc_files = [
            temp_repo_dir / "README.md",
            temp_repo_dir / "LICENSE.txt",
            temp_repo_dir / "docs" / "changelog.rst",
        ]

        # Exercise
        result = process_document_files(doc_files, temp_repo_dir)

        # Verify
        assert len(result) == 3
        urls = [doc["url"] for doc in result]
        assert "README.md" in urls
        assert "LICENSE.txt" in urls
        assert "docs/changelog.rst" in urls

    def test_process_document_files_handles_encoding_errors(
        self, temp_repo_dir, mocker
    ):
        """Test graceful handling of files with encoding issues."""
        # Setup
        bad_file = temp_repo_dir / "bad_encoding.md"
        bad_file.write_text("Good content")
        doc_files = [bad_file]
        mock_builtin_open = mocker.patch("builtins.open")
        mock_builtin_open.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid start byte"
        )

        # Exercise
        result = process_document_files(doc_files, temp_repo_dir)

        # Verify
        assert result == []

    def test_process_document_files_handles_malformed_notebook(self, temp_repo_dir):
        """Test handling of malformed notebook files."""
        # Setup
        notebook_path = temp_repo_dir / "bad_notebook.ipynb"
        notebook_path.write_text("{ invalid json }")
        doc_files = [notebook_path]

        # Exercise
        result = process_document_files(doc_files, temp_repo_dir)

        # Verify
        assert result == []

    def test_process_document_files_empty_list(self, temp_repo_dir):
        """Test behavior with empty file list."""
        result = process_document_files([], temp_repo_dir)

        assert result == []


class TestProcessRepositoryDocs:
    """Tests for process_repository_docs function."""

    @pytest.mark.asyncio
    async def test_process_repository_docs_success(
        self, temp_repo_dir, mock_supabase_client, mocker
    ):
        """Test successful processing of repository documentation."""
        # Setup
        mocker.patch("src.utils.text_processing.smart_chunk_markdown", return_value=["chunk1", "chunk2"])
        mocker.patch("src.utils.document_storage.add_documents_to_supabase", return_value=None)
        repo_name = "test-repo"
        repo_url = "https://github.com/user/test-repo.git"

        # Exercise
        result = await process_repository_docs(
            mock_supabase_client, temp_repo_dir, repo_name, repo_url
        )

        # Verify
        assert result["files_processed"] > 0
        assert result["chunks_created"] > 0
        assert "code_examples_extracted" in result

    @pytest.mark.asyncio
    async def test_process_repository_docs_no_files_found(self, mock_supabase_client):
        """Test behavior when no documentation files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            empty_repo = Path(temp_dir)

            # Exercise
            result = await process_repository_docs(
                mock_supabase_client,
                empty_repo,
                "empty-repo",
                "https://github.com/user/empty-repo.git",
            )

            # Verify
            assert result["files_processed"] == 0
            assert result["chunks_created"] == 0
            assert result["code_examples_extracted"] == 0
            assert "No documentation files found" in result["message"]

    @pytest.mark.asyncio
    async def test_process_repository_docs_handles_processing_errors(
        self, temp_repo_dir, mock_supabase_client, mocker
    ):
        """Test graceful error handling during processing."""
        # Setup
        mocker.patch(
            "src.utils.documentation.process_document_files",
            side_effect=Exception("Processing error"),
        )

        # Exercise
        result = await process_repository_docs(
            mock_supabase_client,
            temp_repo_dir,
            "test-repo",
            "https://github.com/user/test-repo.git",
        )

        # Verify
        assert result["files_processed"] == 0
        assert result["chunks_created"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_process_repository_docs_returns_correct_statistics(
        self, temp_repo_dir, mock_supabase_client, mocker
    ):
        """Test that returned statistics match actual processing."""
        # Setup
        mocker.patch(
            "src.utils.documentation.smart_chunk_markdown", return_value=["chunk1", "chunk2", "chunk3"]
        )
        mocker.patch("src.utils.document_storage.add_documents_to_supabase", return_value=None)

        # Exercise
        result = await process_repository_docs(
            mock_supabase_client,
            temp_repo_dir,
            "test-repo",
            "https://github.com/user/test-repo.git",
        )

        # Verify
        expected_files = len(discover_documentation_files(temp_repo_dir))
        assert result["files_processed"] == expected_files
        expected_chunks = expected_files * 3  # 3 chunks per file from mock
        assert result["chunks_created"] == expected_chunks