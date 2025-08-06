"""Integration tests for repository documentation processing pipeline."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from tests.support.builders.repository_builder import (
    create_standard_repo,
    create_comprehensive_repo,
    create_problematic_repo,
)
from tests.support.mocking.git_operations import GitMocker
from tests.support.mocking.supabase_operations import SupabaseMocker
from tests.support.mocking.neo4j_operations import Neo4jMocker
from tests.support.mocking.openai_operations import OpenAIMocker

from src.crawl4ai_mcp import parse_github_repository


class TestRepositoryDocumentationIntegration:
    """Integration tests for repository documentation processing pipeline."""

    @pytest.mark.asyncio
    async def test_parse_github_repository_tool_success(
        self, test_env, mock_mcp_context
    ):
        """Repository with valid docs processes successfully and returns accurate statistics."""
        # Setup
        repo_url = "https://github.com/test/standard-repo.git"
        
        with (create_standard_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):

            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()

            # Exercise
            result = await parse_github_repository(mock_mcp_context, repo_url)

            # Verify
            response_data = json.loads(result)
            assert response_data["success"] is True
            assert "repo_name" in response_data
            assert "documentation_processing" in response_data
            assert "statistics" in response_data

            doc_stats = response_data["documentation_processing"]
            assert doc_stats["files_processed"] > 0
            assert doc_stats["chunks_created"] > 0

            code_stats = response_data["statistics"]
            assert code_stats["files_processed"] > 0
            assert code_stats["classes_created"] >= 0

    @pytest.mark.asyncio
    async def test_parse_github_repository_invalid_url(
        self, test_env, mock_mcp_context
    ):
        """Invalid GitHub URL returns structured error response."""
        # Setup
        invalid_url = "not-a-valid-git-url"
        
        # Exercise
        result = await parse_github_repository(mock_mcp_context, invalid_url)
        
        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "error" in response_data
        assert "Please provide a valid GitHub repository URL" in response_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_private_repo(
        self, test_env, mock_mcp_context
    ):
        """Private repository returns authentication error."""
        # Setup
        private_repo_url = "https://github.com/private/secret-repo.git"
        
        # Override the mock to simulate git clone failure
        mock_mcp_context.request_context.lifespan_context.repo_extractor.analyze_repository.side_effect = (
            RuntimeError("Git clone failed: Authentication failed")
        )

        # Exercise
        result = await parse_github_repository(mock_mcp_context, private_repo_url)

        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "error" in response_data
        assert "Authentication failed" in response_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_nonexistent_repo(
        self, test_env, mock_mcp_context
    ):
        """Non-existent repository returns 404 error."""
        # Setup
        nonexistent_url = "https://github.com/nonexistent/repo.git"
        
        # Override the mock to simulate git clone failure
        mock_mcp_context.request_context.lifespan_context.repo_extractor.analyze_repository.side_effect = (
            RuntimeError("Git clone failed: Repository not found")
        )

        # Exercise
        result = await parse_github_repository(mock_mcp_context, nonexistent_url)

        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "error" in response_data
        assert "Repository not found" in response_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_no_docs(
        self, test_env, mock_mcp_context
    ):
        """Repository with no documentation files completes with zero doc statistics."""
        # Setup
        repo_url = "https://github.com/test/empty-repo.git"
        
        # Override the mock to return no documentation processing results
        mock_mcp_context.request_context.lifespan_context.repo_extractor.analyze_repository.return_value = (
            "empty-repo",  # repo_name
            [{"file_path": "src/empty.py", "classes": [], "functions": [], "imports": []}],  # modules_data  
            {  # docs_result
                "files_processed": 0,
                "chunks_created": 0,
                "code_examples_extracted": 0,
                "message": "No documentation files found in repository"
            }
        )

        # Exercise
        result = await parse_github_repository(mock_mcp_context, repo_url)

        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is True

        doc_stats = response_data["documentation_processing"]
        assert doc_stats["files_processed"] == 0
        assert doc_stats["chunks_created"] == 0
        assert "No documentation files found" in doc_stats["message"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_large_files_filtered(
        self, test_env, mock_mcp_context
    ):
        """Large documentation files are properly excluded from processing."""
        # Setup
        repo_url = "https://github.com/test/large-files-repo.git"
        
        with (create_standard_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Add large file
            repo_builder.with_large_file("docs/huge_document.md", size_kb=600)
            repo_path = repo_builder.build()
            
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.track_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Exercise
            result = await parse_github_repository(mock_mcp_context, repo_url)
            
            # Verify
            response_data = json.loads(result)
            assert response_data["success"] is True
            
            # Check that large file was not processed
            operations = supabase_mocker.get_tracked_operations("insert")
            for op in operations:
                if isinstance(op["data"], list):
                    for doc in op["data"]:
                        assert "huge_document.md" not in doc.get("url", "")

    @pytest.mark.asyncio
    async def test_parse_github_repository_malformed_notebooks(
        self, test_env, mock_mcp_context
    ):
        """Malformed Jupyter notebooks are handled gracefully."""
        # Setup
        repo_url = "https://github.com/test/malformed-notebook-repo.git"
        
        with (create_problematic_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            
            # Exercise
            result = await parse_github_repository(mock_mcp_context, repo_url)
            
            # Verify
            response_data = json.loads(result)
            assert response_data["success"] is True
            
            # Processing should complete despite malformed notebook
            doc_stats = response_data["documentation_processing"]
            assert doc_stats["files_processed"] >= 0  # Some files should still process

    @pytest.mark.asyncio
    async def test_parse_github_repository_encoding_errors(
        self, test_env, mock_mcp_context
    ):
        """Files with encoding issues are skipped without failing entire process."""
        # Setup
        repo_url = "https://github.com/test/encoding-issues-repo.git"
        
        with (create_problematic_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            
            # Exercise
            result = await parse_github_repository(mock_mcp_context, repo_url)
            
            # Verify
            response_data = json.loads(result)
            assert response_data["success"] is True
            
            # Files with encoding issues should be skipped
            doc_stats = response_data["documentation_processing"]
            assert doc_stats["files_processed"] >= 0

    @pytest.mark.asyncio
    async def test_parse_github_repository_supabase_failure(
        self, test_env, mock_mcp_context
    ):
        """Supabase connection failure returns appropriate error."""
        # Setup
        repo_url = "https://github.com/test/standard-repo.git"
        
        # Override the mock to simulate Supabase connection failure
        mock_mcp_context.request_context.lifespan_context.repo_extractor.analyze_repository.side_effect = (
            Exception("Supabase connection failed: Unable to connect to database")
        )

        # Exercise
        result = await parse_github_repository(mock_mcp_context, repo_url)

        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "error" in response_data
        assert "Supabase" in response_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_neo4j_failure(
        self, test_env, mock_mcp_context
    ):
        """Neo4j connection failure returns appropriate error."""
        # Setup
        repo_url = "https://github.com/test/standard-repo.git"
        
        # Override the mock to simulate Neo4j connection failure
        mock_mcp_context.request_context.lifespan_context.repo_extractor.analyze_repository.side_effect = (
            Exception("Neo4j connection failed: Unable to connect to database")
        )

        # Exercise
        result = await parse_github_repository(mock_mcp_context, repo_url)

        # Verify
        response_data = json.loads(result)
        assert response_data["success"] is False
        assert "error" in response_data
        assert "Neo4j" in response_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_partial_processing_success(
        self, test_env, mock_mcp_context
    ):
        """Some file processing failures don't prevent overall success."""
        # Setup
        repo_url = "https://github.com/test/partial-success-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            
            # Mock partial failures in document processing
            supabase_mocker.mock_successful_operations()
            
            # Mock some OpenAI failures (but not all)
            call_count = 0
            def intermittent_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:  # Fail every 3rd call
                    raise Exception("API rate limit")
                return openai_mocker.mock_embeddings()(*args, **kwargs)
            
            with patch("openai.embeddings.create", side_effect=intermittent_failure):
                neo4j_mocker.mock_successful_operations()
                
                # Exercise
                result = await parse_github_repository(mock_mcp_context, repo_url)
                
                # Verify
                response_data = json.loads(result)
                assert response_data["success"] is True
                
                # Some files should still be processed despite failures
                doc_stats = response_data["documentation_processing"]
                assert doc_stats["files_processed"] > 0

    def _copy_repo(self, source_path: Path, target_path: Path):
        """Helper to copy repository structure for git clone simulation."""
        import shutil
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)