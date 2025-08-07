"""
Query preprocessing utilities for improving RAG search results.

This module provides LLM-powered query planning to transform complex, 
descriptive queries into optimized search strategies that work well 
with hybrid search systems.
"""
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from functools import lru_cache
import time
import hashlib

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class SearchPlan:
    """Represents an optimized search plan for a query."""
    search_terms: List[str]  # Exact phrases to search for
    keywords: List[str]      # Individual key terms for keyword search
    semantic_query: str      # Reformulated query for embedding search
    search_breadth: str      # narrow, medium, or broad
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchPlan':
        """Create SearchPlan from dictionary."""
        return cls(
            search_terms=data.get('search_terms', []),
            keywords=data.get('keywords', []),
            semantic_query=data.get('semantic_query', ''),
            search_breadth=data.get('search_breadth', 'medium')
        )


def _get_cache_key(query: str, use_cache: bool) -> Optional[str]:
    """Generate cache key for query if caching is enabled."""
    if not use_cache:
        return None

    # Use a stable hash of the query and cache version for the cache key
    cache_version = "v1"  # Increment if cache logic changes
    key_material = f"{cache_version}:{query}"

    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1000)
def _cached_plan_search_strategy(cache_key: str, query: str) -> SearchPlan:
    """Cached version of plan_search_strategy."""
    return _plan_search_strategy_impl(query)


def _plan_search_strategy_impl(query: str) -> SearchPlan:
    """
    Implementation of search strategy planning using LLM.
    
    Args:
        query: The user's search query
        
    Returns:
        SearchPlan with optimized search strategies
    """
    prompt = f"""Given this search query: "{query}"
    
Generate a search strategy that will find relevant content in a documentation system.

Return a JSON object with:
{{
    "search_terms": ["exact phrases to search for"],
    "keywords": ["individual key terms"],
    "semantic_query": "reformulated query optimized for embedding search",
    "search_breadth": "narrow|medium|broad"
}}

Guidelines:
- search_terms: Extract key phrases that are likely to appear verbatim in documents
- keywords: Break down concepts into individual terms, include synonyms and related terms
- semantic_query: Rephrase as a natural question that captures the intent
- search_breadth: 
  - narrow: Very specific technical query
  - medium: General technical concept
  - broad: High-level overview or multiple concepts

Example:
Query: "Agent Inbox documentation overview"
{{
    "search_terms": ["Agent Inbox", "documentation", "overview"],
    "keywords": ["agent", "inbox", "docs", "documentation", "overview", "guide", "introduction"],
    "semantic_query": "What is Agent Inbox and how does it work? Overview and documentation guide",
    "search_breadth": "broad"
}}"""
    
    max_retries = 3
    retry_delay = 1.0
    
    for retry in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Fast, cheap model for preprocessing
                messages=[{
                    "role": "system",
                    "content": "You are a search query optimizer. Always return valid JSON."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return SearchPlan.from_dict(result)
            
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error planning search strategy (attempt {retry + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to plan search strategy after {max_retries} attempts: {e}")
                # Fallback to simple strategy
                return SearchPlan(
                    search_terms=[query],
                    keywords=query.split(),
                    semantic_query=query,
                    search_breadth="medium"
                )


def plan_search_strategy(query: str, use_cache: bool = True) -> SearchPlan:
    """
    Use an LLM to intelligently plan how to search for information.
    
    This function analyzes the user's query and generates an optimized
    search strategy that includes:
    - Exact phrases to search for
    - Individual keywords for keyword matching
    - A reformulated query optimized for semantic search
    - Search breadth recommendation
    
    Args:
        query: The user's search query
        use_cache: Whether to cache results for repeated queries
        
    Returns:
        SearchPlan with optimized search strategies
    """
    # Check if LLM query planning is enabled
    if not os.getenv("OPENAI_API_KEY"):
        # No API key, return simple fallback
        return SearchPlan(
            search_terms=[query],
            keywords=query.split(),
            semantic_query=query,
            search_breadth="medium"
        )
    
    # Use caching if enabled
    cache_key = _get_cache_key(query, use_cache)
    if cache_key:
        return _cached_plan_search_strategy(cache_key, query)
    else:
        return _plan_search_strategy_impl(query)


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
        escaped = keyword.replace('%', '\\%').replace('_', '\\_')
        conditions.append(f"content.ilike.%{escaped}%")
    
    # Join with commas for Supabase .or_() or .and_() methods
    return ','.join(conditions)