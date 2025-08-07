"""Unit tests for query preprocessing functionality."""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.utils.query_preprocessing import (
    SearchPlan, 
    plan_search_strategy, 
    build_keyword_conditions,
    _plan_search_strategy_impl
)


class TestSearchPlan:
    """Tests for SearchPlan dataclass."""
    
    def test_search_plan_creation(self):
        """When SearchPlan created with all fields, then fields accessible."""
        # Setup
        plan = SearchPlan(
            search_terms=["Agent Inbox", "documentation"],
            keywords=["agent", "inbox", "docs"],
            semantic_query="What is Agent Inbox documentation?",
            search_breadth="broad"
        )
        
        # Verify
        assert plan.search_terms == ["Agent Inbox", "documentation"]
        assert plan.keywords == ["agent", "inbox", "docs"]
        assert plan.semantic_query == "What is Agent Inbox documentation?"
        assert plan.search_breadth == "broad"
    
    def test_search_plan_from_dict(self):
        """When SearchPlan created from dictionary, then fields populated correctly."""
        # Setup
        data = {
            "search_terms": ["test", "query"],
            "keywords": ["test", "query", "search"],
            "semantic_query": "How to test query search?",
            "search_breadth": "narrow"
        }
        
        # Exercise
        plan = SearchPlan.from_dict(data)
        
        # Verify
        assert plan.search_terms == ["test", "query"]
        assert plan.keywords == ["test", "query", "search"]
        assert plan.semantic_query == "How to test query search?"
        assert plan.search_breadth == "narrow"
    
    def test_search_plan_from_dict_with_missing_fields(self):
        """When SearchPlan created from incomplete dict, then defaults applied."""
        # Setup
        data = {"semantic_query": "partial data"}
        
        # Exercise
        plan = SearchPlan.from_dict(data)
        
        # Verify
        assert plan.search_terms == []
        assert plan.keywords == []
        assert plan.semantic_query == "partial data"
        assert plan.search_breadth == "medium"


class TestPlanSearchStrategy:
    """Tests for plan_search_strategy function."""
    
    @patch('src.utils.query_preprocessing.openai')
    def test_plan_search_strategy_success(self, mock_openai):
        """When valid query provided, then LLM generates optimized search plan."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "search_terms": ["Agent Inbox", "documentation", "overview"],
            "keywords": ["agent", "inbox", "docs", "documentation", "overview", "guide"],
            "semantic_query": "What is Agent Inbox and how does it work?",
            "search_breadth": "broad"
        })
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Exercise
        result = plan_search_strategy("Agent Inbox documentation overview", use_cache=False)
        
        # Verify
        assert result.search_terms == ["Agent Inbox", "documentation", "overview"]
        assert result.keywords == ["agent", "inbox", "docs", "documentation", "overview", "guide"]
        assert result.semantic_query == "What is Agent Inbox and how does it work?"
        assert result.search_breadth == "broad"
        
        # Verify API call
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-3.5-turbo'
        assert call_args.kwargs['temperature'] == 0
    
    @patch('src.utils.query_preprocessing.openai')
    def test_plan_search_strategy_with_retry(self, mock_openai):
        """When LLM fails initially, then retries and succeeds."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "search_terms": ["test"],
            "keywords": ["test"],
            "semantic_query": "test query",
            "search_breadth": "narrow"
        })
        
        # First call fails, second succeeds
        mock_openai.chat.completions.create.side_effect = [
            Exception("API error"),
            mock_response
        ]
        
        # Exercise
        result = plan_search_strategy("test query", use_cache=False)
        
        # Verify
        assert result.search_terms == ["test"]
        assert result.keywords == ["test"]
        assert result.semantic_query == "test query"
        assert mock_openai.chat.completions.create.call_count == 2
    
    @patch('src.utils.query_preprocessing.openai')
    def test_plan_search_strategy_fallback_after_failures(self, mock_openai):
        """When LLM fails all retries, then returns simple fallback strategy."""
        # Setup
        mock_openai.chat.completions.create.side_effect = Exception("API error")
        
        # Exercise
        result = plan_search_strategy("test query with spaces", use_cache=False)
        
        # Verify fallback behavior
        assert result.search_terms == ["test query with spaces"]
        assert result.keywords == ["test", "query", "with", "spaces"]
        assert result.semantic_query == "test query with spaces"
        assert result.search_breadth == "medium"
        assert mock_openai.chat.completions.create.call_count == 3  # All retries
    
    def test_plan_search_strategy_disabled(self):
        """When LLM query planning disabled, then returns simple fallback."""
        # Set USE_LLM_QUERY_PLANNING=false
        with patch.dict('os.environ', {"USE_LLM_QUERY_PLANNING": "false"}):
            # Also ensure no OpenAI API key (since implementation might still check it)
            with patch.dict('os.environ', {"OPENAI_API_KEY": ""}, clear=False):
                # Exercise
                result = plan_search_strategy("test query", use_cache=False)
                
                # Verify
                assert result.search_terms == ["test query"]
                assert result.keywords == ["test", "query"]
                assert result.semantic_query == "test query"
                assert result.search_breadth == "medium"
    
    @patch('src.utils.query_preprocessing.openai')
    def test_plan_search_strategy_caching(self, mock_openai):
        """When same query called multiple times with caching, then LLM called only once."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "search_terms": ["cached"],
            "keywords": ["cached"],
            "semantic_query": "cached query",
            "search_breadth": "narrow"
        })
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Clear any existing cache
        from src.utils.query_preprocessing import _cached_plan_search_strategy
        _cached_plan_search_strategy.cache_clear()
        
        # Exercise - call twice with caching enabled
        result1 = plan_search_strategy("cached query test", use_cache=True)
        result2 = plan_search_strategy("cached query test", use_cache=True)
        
        # Verify
        assert result1.search_terms == result2.search_terms == ["cached"]
        assert mock_openai.chat.completions.create.call_count == 1  # Called only once
    
    @patch('src.utils.query_preprocessing.openai')
    def test_plan_search_strategy_no_caching(self, mock_openai):
        """When caching disabled, then LLM called every time."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "search_terms": ["no cache"],
            "keywords": ["no", "cache"],
            "semantic_query": "no cache query",
            "search_breadth": "medium"
        })
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Exercise - call twice with caching disabled
        result1 = plan_search_strategy("no cache query", use_cache=False)
        result2 = plan_search_strategy("no cache query", use_cache=False)
        
        # Verify
        assert mock_openai.chat.completions.create.call_count == 2  # Called twice


class TestBuildKeywordConditions:
    """Tests for build_keyword_conditions function."""
    
    def test_build_keyword_conditions_single(self):
        """When single keyword provided, then single ILIKE condition returned."""
        # Exercise
        result = build_keyword_conditions(["agent"])
        
        # Verify
        assert result == "content.ilike.%agent%"
    
    def test_build_keyword_conditions_multiple(self):
        """When multiple keywords provided, then comma-separated conditions returned."""
        # Exercise
        result = build_keyword_conditions(["agent", "inbox", "docs"])
        
        # Verify
        assert result == "content.ilike.%agent%,content.ilike.%inbox%,content.ilike.%docs%"
    
    def test_build_keyword_conditions_empty(self):
        """When empty list provided, then empty string returned."""
        # Exercise
        result = build_keyword_conditions([])
        
        # Verify
        assert result == ""
    
    def test_build_keyword_conditions_special_chars(self):
        """When keywords contain special characters, then properly escaped."""
        # Exercise
        result = build_keyword_conditions(["test%", "query_with_underscore"])
        
        # Verify
        assert result == "content.ilike.%test\\%%,content.ilike.%query\\_with\\_underscore%"
    
    def test_build_keyword_conditions_with_operator(self):
        """When operator specified, then conditions formatted for that operator."""
        # Exercise
        result_or = build_keyword_conditions(["test", "query"], operator="or")
        result_and = build_keyword_conditions(["test", "query"], operator="and")
        
        # Verify - both should return same format for Supabase
        assert result_or == "content.ilike.%test%,content.ilike.%query%"
        assert result_and == "content.ilike.%test%,content.ilike.%query%"