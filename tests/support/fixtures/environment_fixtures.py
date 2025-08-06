"""Environment-related test fixtures."""

import os
import pytest
from tests.support.database_helpers import load_test_environment as load_env


# Environment variable requirements
REQUIRED_ENV_VARS = [
    "USE_KNOWLEDGE_GRAPH",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "OPENAI_API_KEY",
]


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables after each test.
    
    This ensures test isolation by restoring the original environment
    after each test completes.
    """
    original_env = os.environ.copy()

    yield

    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_env(monkeypatch):
    """Set up complete test environment.
    
    This is the one fixture to rule them all - sets up everything needed
    for any test that requires environment variables.
    """
    # Core services
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    
    # Knowledge graph (both variations some code uses)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("USE_KNOWLEDGE_GRAPH", "true")
    
    # Features
    monkeypatch.setenv("USE_AGENTIC_RAG", "true")


@pytest.fixture(scope="function")
def environment_check():
    """Verify required environment variables are present.
    
    Used by E2E tests to ensure the test environment is properly configured.
    Skips test if environment is not ready.
    """
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

    if missing_vars:
        pytest.skip(f"Missing required environment variables: {missing_vars}")

    if os.getenv("USE_KNOWLEDGE_GRAPH") != "true":
        pytest.skip("Knowledge graph not enabled")


def load_test_environment():
    """Load environment variables for tests.
    
    This function is called during test session setup.
    """
    load_env()