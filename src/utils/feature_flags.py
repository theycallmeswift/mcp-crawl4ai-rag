"""
Feature flag utilities for managing application features.

This module provides a centralized way to check feature flags
throughout the application, making it easier to enable/disable
features and mock them in tests.
"""

import os


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature flag is enabled via environment variable.

    Feature flags are expected to be environment variables with
    values of "true" (case-insensitive) to be considered enabled.
    Any unset or empty value is considered False.

    Args:
        feature_name: Name of the feature flag environment variable

    Returns:
        True if the feature is enabled, False otherwise
    """
    env_value = os.getenv(feature_name, "").lower()
    return env_value == "true"


def is_llm_query_planning_enabled() -> bool:
    """Check if LLM query planning feature is enabled."""
    return is_feature_enabled("USE_LLM_QUERY_PLANNING")


def is_hybrid_search_enabled() -> bool:
    """Check if hybrid search feature is enabled."""
    return is_feature_enabled("USE_HYBRID_SEARCH")


def is_reranking_enabled() -> bool:
    """Check if reranking feature is enabled."""
    return is_feature_enabled("USE_RERANKING")


def is_agentic_rag_enabled() -> bool:
    """Check if agentic RAG (code example extraction) is enabled."""
    return is_feature_enabled("USE_AGENTIC_RAG")
