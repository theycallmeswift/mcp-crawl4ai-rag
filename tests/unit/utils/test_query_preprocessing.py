"""Unit tests for query preprocessing functionality."""

import json
import pytest
from unittest.mock import Mock, patch
from src.utils.query_preprocessing import (
    SearchPlan,
    plan_search_strategy,
    build_keyword_conditions,
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
        )

        # Verify
        assert plan.search_terms == ["Agent Inbox", "documentation"]
        assert plan.keywords == ["agent", "inbox", "docs"]
        assert plan.semantic_query == "What is Agent Inbox documentation?"

    def test_search_plan_from_dict(self):
        """When SearchPlan created from dictionary, then fields populated correctly."""
        # Setup
        data = {
            "search_terms": ["test", "query"],
            "keywords": ["test", "query", "search"],
            "semantic_query": "How to test query search?",
        }

        # Exercise
        plan = SearchPlan.from_dict(data)

        # Verify
        assert plan.search_terms == ["test", "query"]
        assert plan.keywords == ["test", "query", "search"]
        assert plan.semantic_query == "How to test query search?"

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


class TestPlanSearchStrategy:
    """Tests for plan_search_strategy function."""

    @patch("src.utils.query_preprocessing.generate_chat_completion")
    def test_plan_search_strategy_success(self, mock_generate):
        """When valid query provided, then LLM generates optimized search plan."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search_terms": ["Agent Inbox", "documentation", "overview"],
                "keywords": [
                    "agent",
                    "inbox",
                    "docs",
                    "documentation",
                    "overview",
                    "guide",
                ],
                "semantic_query": "What is Agent Inbox and how does it work?",
            }
        )
        mock_generate.return_value = mock_response

        # Exercise
        with patch.dict(
            "os.environ",
            {"USE_LLM_QUERY_PLANNING": "true", "OPENAI_API_KEY": "test-key"},
        ):
            result = plan_search_strategy("Agent Inbox documentation overview")

        # Verify
        assert result.search_terms == ["Agent Inbox", "documentation", "overview"]
        assert result.keywords == [
            "agent",
            "inbox",
            "docs",
            "documentation",
            "overview",
            "guide",
        ]
        assert result.semantic_query == "What is Agent Inbox and how does it work?"

        # Verify API call
        mock_generate.assert_called_once()

    @patch("src.utils.query_preprocessing.generate_chat_completion")
    def test_plan_search_strategy_with_exception(self, mock_generate):
        """When LLM fails, then raises RuntimeError."""
        # Setup
        mock_generate.side_effect = Exception("API error")

        # Exercise & Verify
        with patch.dict(
            "os.environ",
            {"USE_LLM_QUERY_PLANNING": "true", "OPENAI_API_KEY": "test-key"},
        ):
            with pytest.raises(RuntimeError) as exc_info:
                plan_search_strategy("test query")
            
            assert "Failed to plan search strategy: API error" in str(exc_info.value)

    @patch("src.utils.query_preprocessing.generate_chat_completion")
    def test_plan_search_strategy_json_decode_error(self, mock_generate):
        """When LLM returns invalid JSON, then raises ValueError."""
        # Setup
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_generate.return_value = mock_response

        # Exercise & Verify
        with patch.dict(
            "os.environ",
            {"USE_LLM_QUERY_PLANNING": "true", "OPENAI_API_KEY": "test-key"},
        ):
            with pytest.raises(ValueError) as exc_info:
                plan_search_strategy("test query with spaces")
            
            assert "Failed to parse LLM response as JSON" in str(exc_info.value)

    def test_plan_search_strategy_disabled(self):
        """When LLM query planning disabled, then returns simple fallback."""
        # Set USE_LLM_QUERY_PLANNING=false
        with patch.dict("os.environ", {"USE_LLM_QUERY_PLANNING": "false"}):
            # Exercise
            result = plan_search_strategy("test query")

            # Verify - should return simple fallback when disabled
            assert result.search_terms == ["test query"]
            assert result.keywords == ["test", "query"]
            assert result.semantic_query == "test query"


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
        assert (
            result == "content.ilike.%agent%,content.ilike.%inbox%,content.ilike.%docs%"
        )

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
        assert (
            result
            == "content.ilike.%test\\%%,content.ilike.%query\\_with\\_underscore%"
        )

    def test_build_keyword_conditions_with_operator(self):
        """When operator specified, then conditions formatted for that operator."""
        # Exercise
        result_or = build_keyword_conditions(["test", "query"], operator="or")
        result_and = build_keyword_conditions(["test", "query"], operator="and")

        # Verify - both should return same format for Supabase
        assert result_or == "content.ilike.%test%,content.ilike.%query%"
        assert result_and == "content.ilike.%test%,content.ilike.%query%"
