"""
Query preprocessing utilities for improving RAG search results.

This module provides LLM-powered query planning to transform complex,
descriptive queries into optimized search strategies that work well
with hybrid search systems.
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from src.utils.openai_client import generate_chat_completion
from src.utils.feature_flags import is_llm_query_planning_enabled


@dataclass
class SearchPlan:
    """Represents an optimized search plan for a query."""

    search_terms: List[str]  # Exact phrases to search for
    keywords: List[str]  # Individual key terms for keyword search
    semantic_query: str  # Reformulated query for embedding search

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchPlan":
        """Create SearchPlan from dictionary."""
        return cls(
            search_terms=data.get("search_terms", []),
            keywords=data.get("keywords", []),
            semantic_query=data.get("semantic_query", ""),
        )


def plan_search_strategy(query: str) -> SearchPlan:
    """
    Use an LLM to intelligently plan how to search for information.

    This function analyzes the user's query and generates an optimized
    search strategy that includes:
    - Exact phrases to search for
    - Individual keywords for keyword matching
    - A reformulated query optimized for semantic search

    Args:
        query: The user's search query

    Returns:
        SearchPlan with optimized search strategies
    """
    # Check if LLM query planning is enabled
    if not is_llm_query_planning_enabled():
        # LLM query planning disabled, return simple fallback
        return SearchPlan(
            search_terms=[query], keywords=query.split(), semantic_query=query
        )

    prompt = f"""Given this search query: "{query}"
    
Generate a search strategy that will find relevant content in a documentation system.

Return a JSON object with:
{{
    "search_terms": ["exact phrases to search for"],
    "keywords": ["individual key terms"],
    "semantic_query": "reformulated query optimized for embedding search"
}}

Guidelines:
- search_terms: Extract key phrases that are likely to appear verbatim in documents
- keywords: Break down concepts into individual terms, include synonyms and related terms
- semantic_query: Rephrase as a natural question that captures the intent

Example:
Query: "Agent Inbox documentation overview"
{{
    "search_terms": ["Agent Inbox", "documentation", "overview"],
    "keywords": ["agent", "inbox", "docs", "documentation", "overview", "guide", "introduction"],
    "semantic_query": "What is Agent Inbox and how does it work? Overview and documentation guide"
}}"""

    try:
        # Use the centralized OpenAI client with retry logic
        response = generate_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a search query optimizer. Always return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return SearchPlan.from_dict(result)

    except json.JSONDecodeError as e:
        # Re-raise JSON decode errors with context
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
    except Exception as e:
        # Re-raise other exceptions with context
        raise RuntimeError(f"Failed to plan search strategy: {e}") from e


def build_keyword_conditions(keywords: List[str], operator: str = "or") -> str:
    """
    Build Supabase query conditions for keyword matching.

    Args:
        keywords: List of keywords to match
        operator: Logical operator ('or' or 'and')

    Returns:
        Comma-separated string of ILIKE conditions for Supabase .or_() or .and_()
    """
    if not keywords:
        return ""

    # Escape special characters and build conditions
    conditions = []
    for keyword in keywords:
        # Basic escaping for ILIKE patterns
        escaped = keyword.replace("%", "\\%").replace("_", "\\_")
        conditions.append(f"content.ilike.%{escaped}%")

    # Join with commas for Supabase .or_() or .and_() methods
    return ",".join(conditions)
