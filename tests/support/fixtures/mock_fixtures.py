"""Mock-related test fixtures for external services."""

import pytest
from unittest.mock import Mock, patch
from supabase import Client


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing.
    
    Returns a basic mock suitable for unit tests.
    """
    mock_client = Mock(spec=Client)
    mock_table = Mock()

    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])

    return mock_client


@pytest.fixture
def mock_supabase_client_with_patch():
    """Mock Supabase client with automatic patching.
    
    Used for integration tests where the client needs to be patched globally.
    """
    mock_client = Mock()
    mock_client.table.return_value = Mock()
    mock_client.table.return_value.insert.return_value = Mock()
    mock_client.table.return_value.select.return_value = Mock()
    mock_client.table.return_value.execute.return_value = Mock()
    
    with patch("src.utils.supabase_client.get_supabase_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_git_clone():
    """Mock git clone operation for repository testing."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def mock_neo4j_session():
    """Mock Neo4j session for knowledge graph operations."""
    mock_session = Mock()
    mock_session.run.return_value = []
    mock_session.close.return_value = None
    
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value.session.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings API."""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    
    with patch("openai.embeddings.create", return_value=mock_response):
        yield mock_response


@pytest.fixture
def mock_mcp_context():
    """Mock MCP server context for tool testing."""
    from unittest.mock import AsyncMock, Mock
    
    mock_context = Mock()
    mock_context.session = Mock()
    
    # Mock the repo_extractor that parse_github_repository expects
    mock_repo_extractor = Mock()
    
    # Mock the Neo4j driver with proper async context manager support
    mock_driver = Mock()
    mock_session = AsyncMock()
    
    # Mock the query result with proper data structures
    mock_result = AsyncMock()
    mock_record = {
        'repo_name': 'test-repo',
        'files_count': 3,
        'classes_count': 2,
        'methods_count': 8,
        'functions_count': 4,
        'attributes_count': 6,
        'sample_modules': ['src.main', 'src.utils', 'src.models']
    }
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result
    
    # Create a proper async context manager that doesn't leave unawaited coroutines
    class AsyncContextManager:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    mock_driver.session.return_value = AsyncContextManager()
    
    mock_repo_extractor.driver = mock_driver
    mock_repo_extractor.supabase_client = Mock()
    
    # Configure analyze_repository with default successful results
    # Individual tests can override this behavior by setting side_effect
    mock_repo_extractor.analyze_repository = AsyncMock(return_value=(
        "test-repo",  # repo_name
        [{"file_path": "src/main.py", "classes": [], "functions": [], "imports": []}],  # modules_data
        {  # docs_result
            "files_processed": 3,
            "chunks_created": 15,
            "code_examples_extracted": 2,
            "message": "Successfully processed documentation"
        }
    ))
    
    # Set up the nested context structure that parse_github_repository expects
    mock_context.request_context = Mock()
    mock_context.request_context.lifespan_context = Mock()
    mock_context.request_context.lifespan_context.repo_extractor = mock_repo_extractor
    
    # Add Supabase client mock for perform_rag_query and other tools
    mock_supabase_client = Mock()
    mock_context.request_context.lifespan_context.supabase_client = mock_supabase_client
    
    # Add reranking model mock (set to None by default - tests can override)
    mock_context.request_context.lifespan_context.reranking_model = None
    
    return mock_context


@pytest.fixture
def mock_all_external_services():
    """Convenience fixture that mocks all external services at once.
    
    Useful for integration tests that need to isolate from all external dependencies.
    """
    mocks = {}
    
    # Mock Supabase
    mock_supabase = Mock()
    mock_supabase.table.return_value = Mock()
    mock_supabase.table.return_value.insert.return_value = Mock()
    mock_supabase.table.return_value.select.return_value = Mock()
    mock_supabase.table.return_value.execute.return_value = Mock(data=[])
    mocks['supabase'] = mock_supabase
    
    # Mock Neo4j
    mock_neo4j = Mock()
    mock_neo4j.run.return_value = []
    mock_neo4j.close.return_value = None
    mocks['neo4j'] = mock_neo4j
    
    # Mock OpenAI
    mock_openai = Mock()
    mock_openai.data = [Mock(embedding=[0.1] * 1536)]
    mocks['openai'] = mock_openai
    
    # Mock Git
    mock_git = Mock()
    mock_git.returncode = 0
    mock_git.stdout = ""
    mock_git.stderr = ""
    mocks['git'] = mock_git
    
    with (patch("src.utils.supabase_client.get_supabase_client", return_value=mock_supabase),
          patch("neo4j.GraphDatabase.driver") as mock_driver,
          patch("openai.embeddings.create", return_value=mock_openai),
          patch("subprocess.run", return_value=mock_git)):
        
        mock_driver.return_value.session.return_value = mock_neo4j
        yield mocks