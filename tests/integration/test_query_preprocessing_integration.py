"""Integration tests for query preprocessing with RAG search."""
import pytest
import json
import os
from unittest.mock import patch, Mock
from tests.support.mocking.openai_operations import OpenAIMocker
from tests.support.mocking.supabase_operations import SupabaseMocker


class TestQueryPreprocessingIntegration:
    """Integration tests for query preprocessing with perform_rag_query."""
    
    @pytest.mark.asyncio
    async def test_rag_query_with_preprocessing_enabled(
        self, test_env, mock_mcp_context
    ):
        """When query preprocessing enabled, then optimized search strategy applied."""
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Mock successful Supabase operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
            
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
            
            # Mock embeddings for vector search
            openai_mocker.mock_embeddings()
            
            # Mock query preprocessing response
            preprocessing_response = {
                "search_terms": ["Agent Inbox", "documentation", "overview"],
                "keywords": ["agent", "inbox", "docs", "documentation", "overview", "guide", "manual"],
                "semantic_query": "What is Agent Inbox and how does it work? Complete documentation overview",
                "search_breadth": "broad"
            }
            
            with patch('src.utils.query_preprocessing.openai.chat.completions.create') as mock_chat:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps(preprocessing_response)
                mock_chat.return_value = mock_response
                
                # Exercise with preprocessing enabled
                with patch.dict(os.environ, {
                    "USE_HYBRID_SEARCH": "true",
                    "USE_LLM_QUERY_PLANNING": "true",
                    "USE_RERANKING": "false"
                }):
                    from src.crawl4ai_mcp import perform_rag_query
                    
                    result = await perform_rag_query(
                        mock_mcp_context,
                        query="Agent Inbox documentation overview",
                        source="test.com",
                        match_count=5
                    )
                    
                    # Verify
                    data = json.loads(result)
                    assert data["success"] is True
                    assert data["query_preprocessing"]["enabled"] is True
                    assert data["query_preprocessing"]["semantic_query"] == preprocessing_response["semantic_query"]
                    assert data["query_preprocessing"]["keywords"] == preprocessing_response["keywords"]
                    assert data["query_preprocessing"]["search_breadth"] == "broad"
                    assert data["search_mode"] == "hybrid"
                    
                    # Verify LLM was called for preprocessing
                    mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rag_query_preprocessing_disabled(
        self, test_env, mock_mcp_context
    ):
        """When query preprocessing disabled, then standard search performed."""
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Mock successful operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
            
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
            
            openai_mocker.mock_embeddings()
            
            # Exercise with preprocessing disabled
            with patch.dict(os.environ, {
                "USE_HYBRID_SEARCH": "true",
                "USE_LLM_QUERY_PLANNING": "false",
                "USE_RERANKING": "false"
            }):
                from src.crawl4ai_mcp import perform_rag_query
                
                result = await perform_rag_query(
                    mock_mcp_context,
                    query="Agent Inbox documentation overview",
                    source="test.com",
                    match_count=5
                )
                
                # Verify
                data = json.loads(result)
                assert data["success"] is True
                assert data["query_preprocessing"]["enabled"] is False
                assert data["search_mode"] == "hybrid"
    
    @pytest.mark.asyncio
    async def test_rag_query_preprocessing_fallback_on_error(
        self, test_env, mock_mcp_context
    ):
        """When query preprocessing fails, then fallback to original query."""
        # Clear any cached results from previous tests
        from src.utils.query_preprocessing import _cached_plan_search_strategy
        _cached_plan_search_strategy.cache_clear()
        
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
    
            # Mock successful Supabase operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
    
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
    
            openai_mocker.mock_embeddings()
    
            # Mock preprocessing failure
            with patch('src.utils.query_preprocessing.openai.chat.completions.create') as mock_chat:
                mock_chat.side_effect = Exception("API error")
                
                # Exercise with preprocessing enabled but failing
                with patch.dict(os.environ, {
                    "USE_HYBRID_SEARCH": "true",
                    "USE_LLM_QUERY_PLANNING": "true",
                    "USE_RERANKING": "false"
                }):
                    from src.crawl4ai_mcp import perform_rag_query
                    
                    result = await perform_rag_query(
                        mock_mcp_context,
                        query="test query",
                        source="test.com",
                        match_count=5
                    )
                    
                    # Verify fallback behavior
                    data = json.loads(result)
                    assert data["success"] is True
                    # Preprocessing is still enabled, but uses fallback strategy
                    assert data["query_preprocessing"]["enabled"] is True
                    # Verify fallback values were used
                    assert data["query_preprocessing"]["search_terms"] == ["test query"]
                    assert data["query_preprocessing"]["keywords"] == ["test", "query"]
                    assert data["query_preprocessing"]["semantic_query"] == "test query"
                    assert data["query_preprocessing"]["search_breadth"] == "medium"
                    assert data["search_mode"] == "hybrid"
                    
                    # Verify LLM was attempted 3 times (retries)
                    assert mock_chat.call_count == 3
    
    @pytest.mark.asyncio
    async def test_vector_only_search_with_preprocessing(
        self, test_env, mock_mcp_context
    ):
        """When preprocessing enabled with vector-only search, then semantic query used."""
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Mock successful operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
            
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
            
            openai_mocker.mock_embeddings()
            
            # Mock query preprocessing
            preprocessing_response = {
                "search_terms": ["authentication", "handler"],
                "keywords": ["auth", "authenticate", "handler", "callback"],
                "semantic_query": "How does authentication and authorization work with handlers?",
                "search_breadth": "medium"
            }
            
            with patch('src.utils.query_preprocessing.openai.chat.completions.create') as mock_chat:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps(preprocessing_response)
                mock_chat.return_value = mock_response
                
                # Exercise with vector-only search and preprocessing
                with patch.dict(os.environ, {
                    "USE_HYBRID_SEARCH": "false",
                    "USE_LLM_QUERY_PLANNING": "true",
                    "USE_RERANKING": "false"
                }):
                    from src.crawl4ai_mcp import perform_rag_query
                    
                    result = await perform_rag_query(
                        mock_mcp_context,
                        query="authentication handler",
                        match_count=5
                    )
                    
                    # Verify
                    data = json.loads(result)
                    assert data["success"] is True
                    assert data["query_preprocessing"]["enabled"] is True
                    assert data["query_preprocessing"]["semantic_query"] == preprocessing_response["semantic_query"]
                    assert data["search_mode"] == "vector"
    
    @pytest.mark.asyncio  
    async def test_preprocessing_with_special_characters(
        self, test_env, mock_mcp_context
    ):
        """When query contains special characters, then preprocessing handles correctly."""
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Mock successful operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
            
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
            
            openai_mocker.mock_embeddings()
            
            # Mock preprocessing with special chars in keywords
            preprocessing_response = {
                "search_terms": ["test_function", "api/v2"],
                "keywords": ["test_function", "api", "v2", "function", "endpoint"],
                "semantic_query": "How to use test_function with api/v2 endpoint?",
                "search_breadth": "narrow"
            }
            
            with patch('src.utils.query_preprocessing.openai.chat.completions.create') as mock_chat:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps(preprocessing_response)
                mock_chat.return_value = mock_response
                
                # Exercise
                with patch.dict(os.environ, {
                    "USE_HYBRID_SEARCH": "true",
                    "USE_LLM_QUERY_PLANNING": "true",
                    "USE_RERANKING": "false"
                }):
                    from src.crawl4ai_mcp import perform_rag_query
                    
                    result = await perform_rag_query(
                        mock_mcp_context,
                        query="test_function api/v2",
                        match_count=5
                    )
                    
                    # Verify
                    data = json.loads(result)
                    assert data["success"] is True
                    assert data["query_preprocessing"]["enabled"] is True
                    assert "test_function" in data["query_preprocessing"]["keywords"]
    
    @pytest.mark.asyncio
    async def test_preprocessing_caching_behavior(
        self, test_env, mock_mcp_context
    ):
        """When same query executed multiple times, then preprocessing cached."""
        # Setup
        with (SupabaseMocker() as supabase_mocker,
              OpenAIMocker() as openai_mocker):
            
            # Mock successful operations
            mock_supabase_client = supabase_mocker.mock_successful_operations()
            
            # Set the Supabase client in the mock context
            mock_mcp_context.request_context.lifespan_context.supabase_client = mock_supabase_client
            
            openai_mocker.mock_embeddings()
            
            # Mock preprocessing
            preprocessing_response = {
                "search_terms": ["cached", "query"],
                "keywords": ["cached", "query", "test"],
                "semantic_query": "Testing cached query behavior",
                "search_breadth": "narrow"
            }
            
            with patch('src.utils.query_preprocessing.openai.chat.completions.create') as mock_chat:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps(preprocessing_response)
                mock_chat.return_value = mock_response
                
                # Clear any existing cache
                from src.utils.query_preprocessing import _cached_plan_search_strategy
                _cached_plan_search_strategy.cache_clear()
                
                # Exercise - same query twice
                with patch.dict(os.environ, {
                    "USE_HYBRID_SEARCH": "true",
                    "USE_LLM_QUERY_PLANNING": "true",
                    "USE_RERANKING": "false"
                }):
                    from src.crawl4ai_mcp import perform_rag_query
                    
                    # First call
                    result1 = await perform_rag_query(
                        mock_mcp_context,
                        query="cached test query",
                        match_count=5
                    )
                    
                    # Second call with same query
                    result2 = await perform_rag_query(
                        mock_mcp_context,
                        query="cached test query",
                        match_count=5
                    )
                    
                    # Verify both succeeded
                    data1 = json.loads(result1)
                    data2 = json.loads(result2)
                    assert data1["success"] is True
                    assert data2["success"] is True
                    
                    # Verify preprocessing was only called once (cached)
                    assert mock_chat.call_count == 1