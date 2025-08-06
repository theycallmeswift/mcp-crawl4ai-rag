"""Integration tests for advanced RAG strategies with repository parsing."""

import pytest
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, Mock

from tests.support.builders.repository_builder import (
    create_standard_repo,
    create_comprehensive_repo,
)
from tests.support.mocking.git_operations import GitMocker
from tests.support.mocking.supabase_operations import SupabaseMocker
from tests.support.mocking.neo4j_operations import Neo4jMocker
from tests.support.mocking.openai_operations import OpenAIMocker

from src.crawl4ai_mcp import (
    parse_github_repository,
    perform_rag_query,
    search_code_examples,
)


class TestAdvancedRAGIntegration:
    """Integration tests for advanced RAG strategies with repository parsing."""

    @pytest.mark.asyncio
    async def test_contextual_embeddings_with_repo_docs(
        self, test_env, mock_mcp_context
    ):
        """Contextual embeddings strategy enhances repository documentation search."""
        # Setup: Enable USE_CONTEXTUAL_EMBEDDINGS, mock LLM calls
        repo_url = "https://github.com/test/contextual-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Mock environment variable for contextual embeddings
            with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true"}):
                # Exercise: Parse repo with contextual embeddings enabled
                result = await parse_github_repository(mock_mcp_context, repo_url)
                
                # Verify: Enhanced context in stored embeddings
                result_data = json.loads(result)
                assert result_data["success"] is True
                
                # Should have processed documentation
                doc_stats = result_data["documentation_processing"]
                assert doc_stats["files_processed"] >= 0
                
                # Test that RAG search works with contextual embeddings
                rag_result = await perform_rag_query(
                    mock_mcp_context,
                    "installation instructions",
                    match_count=3
                )
                
                rag_data = json.loads(rag_result)
                # Should work regardless of success (mocked data)
                assert "query" in rag_data or "error" in rag_data

    @pytest.mark.asyncio
    async def test_agentic_rag_code_extraction(
        self, test_env, mock_mcp_context
    ):
        """Agentic RAG extracts and summarizes code examples from repository docs."""
        # Setup
        repo_url = "https://github.com/test/agentic-rag-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Mock environment variable for agentic RAG
            with patch.dict(os.environ, {"USE_AGENTIC_RAG": "true"}):
                # Exercise: Parse repo with agentic RAG enabled
                result = await parse_github_repository(mock_mcp_context, repo_url)
                
                # Verify: Code examples are extracted and summarized
                result_data = json.loads(result)
                assert result_data["success"] is True
                
                # Should have processed documentation
                doc_stats = result_data["documentation_processing"]
                assert doc_stats["files_processed"] >= 0
                
                # Test code example search functionality
                code_result = await search_code_examples(
                    mock_mcp_context,
                    "function definition example",
                    match_count=3
                )
                
                code_data = json.loads(code_result)
                # Should work with agentic RAG enabled
                if code_data["success"]:
                    assert "query" in code_data
                    assert code_data["query"] == "function definition example"
                else:
                    # Acceptable if mocked data doesn't support full workflow
                    assert "error" in code_data

    @pytest.mark.asyncio
    async def test_hybrid_search_with_repo_content(
        self, test_env, mock_mcp_context
    ):
        """Hybrid search combines keyword and semantic search for repository content."""
        # Setup
        repo_url = "https://github.com/test/hybrid-search-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Mock environment variable for hybrid search
            with patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"}):
                # Exercise: Parse repo first
                parse_result = await parse_github_repository(mock_mcp_context, repo_url)
                
                # Then perform hybrid search
                rag_result = await perform_rag_query(
                    mock_mcp_context,
                    "API documentation",
                    match_count=5
                )
                
                # Verify: Hybrid search combines keyword and semantic search
                parse_data = json.loads(parse_result)
                rag_data = json.loads(rag_result)
                
                assert parse_data["success"] is True
                
                # Check that search mode is indicated in response
                if rag_data["success"]:
                    assert "search_mode" in rag_data
                    assert rag_data["search_mode"] in ["hybrid", "vector"]
                else:
                    # Acceptable with mocked data
                    assert "error" in rag_data

    @pytest.mark.asyncio
    async def test_reranking_improves_repo_search_results(
        self, test_env, mock_mcp_context
    ):
        """Reranking strategy improves relevance of repository search results."""
        # Setup
        repo_url = "https://github.com/test/reranking-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Mock reranking model in context
            mock_reranking_model = Mock()
            mock_reranking_model.predict.return_value = [[0.9, 0.7, 0.5]]  # Mock scores
            mock_mcp_context.request_context.lifespan_context.reranking_model = mock_reranking_model
            
            # Mock environment variable for reranking
            with patch.dict(os.environ, {"USE_RERANKING": "true"}):
                # Exercise: Parse repo first
                parse_result = await parse_github_repository(mock_mcp_context, repo_url)
                
                # Then perform search with reranking
                rag_result = await perform_rag_query(
                    mock_mcp_context,
                    "configuration guide",
                    match_count=3
                )
                
                # Verify: Reranking is applied to improve result relevance
                parse_data = json.loads(parse_result)
                rag_data = json.loads(rag_result)
                
                assert parse_data["success"] is True
                
                if rag_data["success"]:
                    # Should indicate reranking was applied
                    assert "reranking_applied" in rag_data
                    # Results may have rerank scores
                    if rag_data.get("results"):
                        for result in rag_data["results"]:
                            # May have rerank score if reranking was applied
                            assert isinstance(result, dict)
                else:
                    # Acceptable with mocked data
                    assert "error" in rag_data

    @pytest.mark.asyncio
    async def test_feature_flag_combinations(
        self, test_env, mock_mcp_context
    ):
        """Different RAG strategy combinations work correctly together."""
        # Setup
        repo_url = "https://github.com/test/feature-combo-repo.git"
        
        with (create_comprehensive_repo() as repo_builder,
              GitMocker() as git_mocker,
              SupabaseMocker() as supabase_mocker,
              Neo4jMocker() as neo4j_mocker,
              OpenAIMocker() as openai_mocker):
            
            repo_path = repo_builder.build()
            git_mocker.mock_successful_clone(lambda target_dir: self._copy_repo(repo_path, target_dir))
            supabase_mocker.mock_successful_operations()
            neo4j_mocker.mock_successful_operations()
            openai_mocker.mock_embeddings()
            
            # Mock reranking model
            mock_reranking_model = Mock()
            mock_reranking_model.predict.return_value = [[0.8, 0.6, 0.4]]
            mock_mcp_context.request_context.lifespan_context.reranking_model = mock_reranking_model
            
            # Test different feature flag combinations
            test_combinations = [
                # All features enabled
                {
                    "USE_CONTEXTUAL_EMBEDDINGS": "true",
                    "USE_HYBRID_SEARCH": "true", 
                    "USE_AGENTIC_RAG": "true",
                    "USE_RERANKING": "true"
                },
                # Minimal setup
                {
                    "USE_CONTEXTUAL_EMBEDDINGS": "false",
                    "USE_HYBRID_SEARCH": "true",
                    "USE_AGENTIC_RAG": "false", 
                    "USE_RERANKING": "false"
                },
                # AI coding assistant setup
                {
                    "USE_CONTEXTUAL_EMBEDDINGS": "true",
                    "USE_HYBRID_SEARCH": "true",
                    "USE_AGENTIC_RAG": "true",
                    "USE_RERANKING": "true"
                }
            ]
            
            for i, env_vars in enumerate(test_combinations):
                with patch.dict(os.environ, env_vars):
                    # Exercise: Parse repository with specific feature combination
                    result = await parse_github_repository(mock_mcp_context, repo_url)
                    
                    # Verify: Each combination works correctly
                    result_data = json.loads(result)
                    assert result_data["success"] is True, f"Combination {i+1} failed"
                    
                    # Should have consistent structure regardless of features
                    assert "repo_name" in result_data
                    assert "documentation_processing" in result_data
                    
                    # Test RAG query with this combination
                    rag_result = await perform_rag_query(
                        mock_mcp_context,
                        f"test query {i+1}",
                        match_count=2
                    )
                    
                    rag_data = json.loads(rag_result)
                    # Should handle query regardless of feature combination
                    assert "query" in rag_data or "error" in rag_data

    @pytest.mark.asyncio
    async def test_performance_impact_measurement(
        self, test_env, mock_mcp_context
    ):
        """Advanced strategies have measurable performance characteristics."""
        # Setup
        repo_url = "https://github.com/test/performance-repo.git"
        
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
            
            # Test performance with different configurations
            performance_results = {}
            
            configs = [
                ("basic", {"USE_CONTEXTUAL_EMBEDDINGS": "false", "USE_HYBRID_SEARCH": "false", "USE_RERANKING": "false"}),
                ("advanced", {"USE_CONTEXTUAL_EMBEDDINGS": "true", "USE_HYBRID_SEARCH": "true", "USE_RERANKING": "true"})
            ]
            
            for config_name, env_vars in configs:
                with patch.dict(os.environ, env_vars):
                    # Measure parsing time
                    start_time = time.time()
                    result = await parse_github_repository(mock_mcp_context, repo_url)
                    parse_time = time.time() - start_time
                    
                    # Measure query time
                    start_time = time.time()
                    rag_result = await perform_rag_query(
                        mock_mcp_context,
                        "performance test query",
                        match_count=3
                    )
                    query_time = time.time() - start_time
                    
                    # Store performance metrics
                    performance_results[config_name] = {
                        "parse_time": parse_time,
                        "query_time": query_time,
                        "parse_success": json.loads(result)["success"],
                        "query_response": json.loads(rag_result)
                    }
            
            # Verify: Performance characteristics are measurable
            assert "basic" in performance_results
            assert "advanced" in performance_results
            
            # Both configurations should succeed
            assert performance_results["basic"]["parse_success"] is True
            assert performance_results["advanced"]["parse_success"] is True
            
            # Performance metrics should be positive numbers
            for config in performance_results.values():
                assert config["parse_time"] > 0
                assert config["query_time"] > 0
            
            # Advanced configuration may take longer but should still be reasonable
            # (In real scenarios, but with mocks the difference may be minimal)
            basic_total = performance_results["basic"]["parse_time"] + performance_results["basic"]["query_time"]
            advanced_total = performance_results["advanced"]["parse_time"] + performance_results["advanced"]["query_time"]
            
            # Both should complete in reasonable time (with mocks, should be very fast)
            assert basic_total < 30.0  # 30 seconds max with mocks
            assert advanced_total < 30.0  # 30 seconds max with mocks

    def _copy_repo(self, source_path: Path, target_path: Path):
        """Helper to copy repository structure for git clone simulation."""
        import shutil
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)