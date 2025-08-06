"""
Consolidated conftest.py for all test configurations.

This module provides shared fixtures and configurations for unit, integration, and E2E tests.
All fixtures are organized into logical groups and imported from fixture factories.
"""

import asyncio
import pytest
import pytest_asyncio

# Import fixture factories
from tests.support.fixtures.repository_fixtures import (
    temp_repo_dir,
    temp_git_repo,
    test_repository_config,
    minimal_repository_config,
)

from tests.support.fixtures.mock_fixtures import (
    mock_supabase_client,
    mock_supabase_client_with_patch,
    mock_git_clone,
    mock_neo4j_session,
    mock_openai_embeddings,
    mock_mcp_context,
    mock_all_external_services,
)

from tests.support.fixtures.environment_fixtures import (
    reset_env_vars,
    test_env,
    environment_check,
    load_test_environment,
)

from tests.support.fixtures.data_fixtures import (
    sample_notebook_content,
    repo_info,
    doc_file_info,
    sample_markdown_content,
    sample_code_examples,
    sample_supabase_documents,
    sample_neo4j_nodes,
    sample_processing_statistics,
)

# Import E2E specific fixtures
from tests.support.database_helpers import (
    get_neo4j_client,
    get_supabase_client,
    cleanup_neo4j_test_data,
    cleanup_supabase_test_data,
)
from tests.support.mcp_client import MCPTestClient


# Load environment for all tests
load_test_environment()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()

    yield loop

    loop.close()


@pytest_asyncio.fixture(scope="function")
async def mcp_client():
    """MCP client connected to the running dev server.
    
    Used for E2E tests that need to interact with the actual MCP server.
    """
    client = MCPTestClient()

    try:
        await client._connect()
        yield client
    except Exception as e:
        # Only skip if we can't connect initially
        pytest.skip(f"MCP server not available: {e}")
    finally:
        # Manual cleanup without triggering skip
        try:
            if hasattr(client, "session") and client.session:
                await client.session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if hasattr(client, "_client_context") and client._client_context:
                await client._client_context.__aexit__(None, None, None)
        except Exception:
            pass


@pytest_asyncio.fixture(scope="function")
async def cleanup_test_data(test_repository_config):
    """Clean up test data before and after tests.
    
    Used for E2E tests to ensure clean database state.
    """
    repo_name = test_repository_config["name"]
    supabase_pattern = test_repository_config["supabase_source_pattern"]

    async def cleanup():
        neo4j_client = await get_neo4j_client()
        try:
            await cleanup_neo4j_test_data(neo4j_client, repo_name)
        finally:
            await neo4j_client.close()

        supabase_client = get_supabase_client()
        cleanup_supabase_test_data(supabase_client, supabase_pattern)

    # Clean up before test
    await cleanup()

    # Yield control to test
    yield cleanup

    # Clean up after test
    await cleanup()


# Test markers for different test types
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (isolated, fast)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (mocked external deps)"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (real external services)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Get the relative path from the tests directory
        relative_path = item.fspath.relto(item.session.fspath.join("tests"))
        
        if relative_path:
            if relative_path.startswith("unit/"):
                item.add_marker(pytest.mark.unit)
            elif relative_path.startswith("integration/"):
                item.add_marker(pytest.mark.integration)
            elif relative_path.startswith("e2e/"):
                item.add_marker(pytest.mark.e2e)


# Make all fixtures available at module level for easy importing
__all__ = [
    # Repository fixtures
    "temp_repo_dir",
    "temp_git_repo", 
    "test_repository_config",
    "minimal_repository_config",
    
    # Mock fixtures
    "mock_supabase_client",
    "mock_supabase_client_with_patch",
    "mock_git_clone",
    "mock_neo4j_session", 
    "mock_openai_embeddings",
    "mock_mcp_context",
    "mock_all_external_services",
    
    # Environment fixtures
    "reset_env_vars",
    "test_env",
    "environment_check",
    
    # Data fixtures
    "sample_notebook_content",
    "repo_info",
    "doc_file_info",
    "sample_markdown_content",
    "sample_code_examples",
    "sample_supabase_documents",
    "sample_neo4j_nodes",
    "sample_processing_statistics",
    
    # E2E fixtures
    "mcp_client",
    "cleanup_test_data",
    "event_loop",
]