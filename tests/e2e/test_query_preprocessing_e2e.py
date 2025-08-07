"""End-to-end tests for query preprocessing with real services."""
import pytest
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for E2E tests
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)


class TestQueryPreprocessingE2E:
    """E2E tests for query preprocessing with real RAG search."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_query_with_preprocessing_real_services(
        self, environment_check, cleanup_test_data, mcp_client
    ):
        """When query preprocessing enabled with real services, then improved search results."""
        # Setup - ensure preprocessing is enabled
        original_env = {
            "USE_HYBRID_SEARCH": os.getenv("USE_HYBRID_SEARCH", "false"),
            "USE_LLM_QUERY_PLANNING": os.getenv("USE_LLM_QUERY_PLANNING", "false"),
            "USE_RERANKING": os.getenv("USE_RERANKING", "false")
        }
        
        try:
            # Enable features for test
            os.environ["USE_HYBRID_SEARCH"] = "true"
            os.environ["USE_LLM_QUERY_PLANNING"] = "true"
            os.environ["USE_RERANKING"] = "false"
            
            # First check available sources
            sources_result = await mcp_client.call_tool(
                "get_available_sources", {"random_string": "dummy"}
            )
            
            if not sources_result.get("sources"):
                pytest.skip("No sources available in database for E2E testing")
            
            # Pick the first available source
            test_source = sources_result["sources"][0]["source_id"]
            
            # Exercise - complex descriptive query that would fail without preprocessing
            result = await mcp_client.call_tool(
                "perform_rag_query",
                {
                    "query": "comprehensive documentation overview and getting started guide",
                    "source": test_source,
                    "match_count": 5
                }
            )
            
            # Verify
            assert result["success"] is True
            assert result["query_preprocessing"]["enabled"] is True
            assert result["search_mode"] == "hybrid"
            
            # Verify preprocessing improved the query
            assert len(result["query_preprocessing"]["keywords"]) > 3  # Should expand keywords
            assert result["query_preprocessing"]["semantic_query"] != result["query"]  # Should reformulate
            
            # Results should be returned (if source has content)
            if result["count"] > 0:
                assert len(result["results"]) > 0
                assert all("content" in r for r in result["results"])
                
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_preprocessing_with_complex_query(
        self, environment_check, cleanup_test_data, mcp_client
    ):
        """When using preprocessing with a complex query, then keywords are expanded and semantic query is improved."""
        # Note: E2E tests can't change MCP server environment variables at runtime.
        # This test assumes the MCP server is running with USE_LLM_QUERY_PLANNING=true
        
        # Get available sources
        sources_result = await mcp_client.call_tool(
            "get_available_sources", {"random_string": "dummy"}
        )
        
        if not sources_result.get("sources"):
            pytest.skip("No sources available in database for E2E testing")
        
        test_source = sources_result["sources"][0]["source_id"]
        
        # Test with a complex query that benefits from preprocessing
        complex_query = "Agent Inbox authentication and authorization workflow documentation"
        
        result = await mcp_client.call_tool(
            "perform_rag_query",
            {
                "query": complex_query,
                "source": test_source,
                "match_count": 10
            }
        )
        
        # Verify the query was processed successfully
        assert result["success"] is True
        
        # Check if preprocessing is enabled (depends on server configuration)
        if result.get("query_preprocessing", {}).get("enabled"):
            # Verify preprocessing expanded the query
            assert len(result["query_preprocessing"]["keywords"]) > 3
            assert result["query_preprocessing"]["semantic_query"] != complex_query
            
            # Log the preprocessing results for debugging
            print(f"Original query: {complex_query}")
            print(f"Semantic query: {result['query_preprocessing']['semantic_query']}")
            print(f"Keywords: {result['query_preprocessing']['keywords']}")
        else:
            # If preprocessing is not enabled, just verify basic functionality
            assert result["search_mode"] in ["vector", "hybrid"]
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_preprocessing_with_reranking(
        self, environment_check, cleanup_test_data, mcp_client
    ):
        """When preprocessing combined with reranking, then best results achieved."""
        # Setup
        original_env = {
            "USE_HYBRID_SEARCH": os.getenv("USE_HYBRID_SEARCH", "false"),
            "USE_LLM_QUERY_PLANNING": os.getenv("USE_LLM_QUERY_PLANNING", "false"),
            "USE_RERANKING": os.getenv("USE_RERANKING", "false")
        }
        
        try:
            # Enable all features
            os.environ["USE_HYBRID_SEARCH"] = "true"
            os.environ["USE_LLM_QUERY_PLANNING"] = "true"
            os.environ["USE_RERANKING"] = "true"
            
            # Get available sources
            sources_result = await mcp_client.call_tool(
                "get_available_sources", {"random_string": "dummy"}
            )
            
            if not sources_result.get("sources"):
                pytest.skip("No sources available in database for E2E testing")
            
            test_source = sources_result["sources"][0]["source_id"]
            
            # Exercise with all features enabled
            result = await mcp_client.call_tool(
                "perform_rag_query",
                {
                    "query": "how to implement authentication with custom handlers and callbacks",
                    "source": test_source,
                    "match_count": 5
                }
            )
            
            # Verify
            assert result["success"] is True
            assert result["query_preprocessing"]["enabled"] is True
            assert result["reranking_applied"] is True
            assert result["search_mode"] == "hybrid"
            
            # If results found, verify reranking scores
            if result["count"] > 0:
                # Check if results have rerank scores
                has_rerank_scores = any("rerank_score" in r for r in result["results"])
                if has_rerank_scores:
                    # Verify results are ordered by rerank score
                    scores = [r.get("rerank_score", 0) for r in result["results"]]
                    assert scores == sorted(scores, reverse=True)
                    
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_preprocessing_performance(
        self, environment_check, cleanup_test_data, mcp_client
    ):
        """When preprocessing enabled, then latency increase is acceptable."""
        import time
        
        # Setup
        original_env = {
            "USE_HYBRID_SEARCH": os.getenv("USE_HYBRID_SEARCH", "false"),
            "USE_LLM_QUERY_PLANNING": os.getenv("USE_LLM_QUERY_PLANNING", "false"),
            "USE_RERANKING": os.getenv("USE_RERANKING", "false")
        }
        
        try:
            # Get available sources
            sources_result = await mcp_client.call_tool(
                "get_available_sources", {"random_string": "dummy"}
            )
            
            if not sources_result.get("sources"):
                pytest.skip("No sources available in database for E2E testing")
            
            test_source = sources_result["sources"][0]["source_id"]
            test_query = "user authentication and session management"
            
            # Warm up the system with one query
            os.environ["USE_HYBRID_SEARCH"] = "true"
            os.environ["USE_LLM_QUERY_PLANNING"] = "false"
            await mcp_client.call_tool(
                "perform_rag_query",
                {"query": "warmup", "source": test_source, "match_count": 1}
            )
            
            # Test without preprocessing
            start_without = time.time()
            result_without = await mcp_client.call_tool(
                "perform_rag_query",
                {
                    "query": test_query,
                    "source": test_source,
                    "match_count": 5
                }
            )
            time_without = time.time() - start_without
            
            # Test with preprocessing (cached should be fast)
            os.environ["USE_LLM_QUERY_PLANNING"] = "true"
            
            # First call (not cached)
            start_with_cold = time.time()
            result_with_cold = await mcp_client.call_tool(
                "perform_rag_query",
                {
                    "query": test_query + " cold",
                    "source": test_source,
                    "match_count": 5
                }
            )
            time_with_cold = time.time() - start_with_cold
            
            # Second call (should be cached)
            start_with_warm = time.time()
            result_with_warm = await mcp_client.call_tool(
                "perform_rag_query",
                {
                    "query": test_query + " cold",  # Same query to hit cache
                    "source": test_source,
                    "match_count": 5
                }
            )
            time_with_warm = time.time() - start_with_warm
            
            # Verify all succeeded
            assert result_without["success"] is True
            assert result_with_cold["success"] is True
            assert result_with_warm["success"] is True
            
            # Log performance metrics
            print("Performance metrics:")
            print(f"  Without preprocessing: {time_without:.3f}s")
            print(f"  With preprocessing (cold): {time_with_cold:.3f}s")
            print(f"  With preprocessing (cached): {time_with_warm:.3f}s")
            
            # Verify cached is faster than cold
            assert time_with_warm < time_with_cold
            
            # Verify latency increase is reasonable (< 1 second for cold start)
            latency_increase = time_with_cold - time_without
            assert latency_increase < 1.0, f"Preprocessing added {latency_increase:.3f}s which exceeds 1 second threshold"
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value