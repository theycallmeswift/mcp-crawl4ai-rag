# Test Fixtures Guide

This directory contains organized test fixtures for the MCP Crawl4AI RAG system. All fixtures are automatically available in tests through the consolidated `tests/conftest.py`.

## Fixture Categories

### 📁 Repository Fixtures (`repository_fixtures.py`)

**Repository Structure Fixtures:**
- `temp_repo_dir` - Standard unit test repository with docs, large files, excluded dirs
- `temp_git_repo` - Integration test repository with git metadata, notebooks, source code
- `test_repository_config` - E2E test repository configuration (session-scoped)
- `minimal_repository_config` - Minimal E2E test repository configuration (session-scoped)

**Helper Functions:**
- `create_temp_repo_with_structure(structure: Dict)` - Create custom repository layouts

**Example Usage:**
```python
def test_with_standard_repo(temp_repo_dir):
    # temp_repo_dir provides: README.md, docs/, LICENSE.txt, large_file.md
    assert (temp_repo_dir / "README.md").exists()

def test_with_git_repo(temp_git_repo):
    # temp_git_repo provides: .git/, docs/, src/, notebooks
    assert (temp_git_repo / ".git").exists()
    assert (temp_git_repo / "docs" / "tutorial.ipynb").exists()
```

### 🎭 Mock Fixtures (`mock_fixtures.py`)

**Individual Service Mocks:**
- `mock_supabase_client` - Basic mock for unit tests
- `mock_supabase_client_with_patch` - Auto-patched mock for integration tests
- `mock_git_clone` - Mock git clone operations
- `mock_neo4j_session` - Mock Neo4j database operations
- `mock_openai_embeddings` - Mock OpenAI API calls
- `mock_mcp_context` - Mock MCP server context

**Convenience Mocks:**
- `mock_all_external_services` - Mocks all external services at once

**Example Usage:**
```python
def test_database_operation(mock_supabase_client):
    # Use mock for unit test
    result = some_function_that_uses_supabase()
    mock_supabase_client.table.assert_called()

def test_integration_scenario(mock_all_external_services):
    # All external services are mocked
    result = complex_integration_function()
    assert result["success"] is True
```

### 🌍 Environment Fixtures (`environment_fixtures.py`)

**Environment Setup:**
- `reset_env_vars` - Auto-reset environment after each test (autouse)
- `test_env` - Complete environment setup for all test types

**Environment Validation:**
- `environment_check` - Validate required env vars for E2E tests (skips if missing)

**Example Usage:**
```python
def test_with_environment(test_env):
    # Sets all required environment variables:
    # SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY
    # NEO4J_URI, NEO4J_USER, NEO4J_USERNAME, NEO4J_PASSWORD
    # USE_KNOWLEDGE_GRAPH, USE_AGENTIC_RAG
    client = get_supabase_client()
    assert client is not None

@pytest.mark.e2e
def test_e2e_scenario(environment_check):
    # Validates all required env vars or skips test
    # Test only runs if environment is properly configured
```

### 📊 Data Fixtures (`data_fixtures.py`)

**Sample Content:**
- `sample_notebook_content` - Jupyter notebook structure
- `repo_info` - Repository metadata
- `doc_file_info` - Documentation file information  
- `sample_markdown_content` - Rich markdown with code examples
- `sample_code_examples` - Array of code examples in different languages

**Sample Database Data:**
- `sample_supabase_documents` - Documents as they appear in Supabase
- `sample_neo4j_nodes` - Knowledge graph node data
- `sample_processing_statistics` - Processing result statistics

**Helper Functions:**
- `create_sample_notebook(title, code_cells)` - Create custom notebook content

**Example Usage:**
```python
def test_notebook_processing(sample_notebook_content):
    # Process the sample notebook
    result = convert_notebook_to_markdown(sample_notebook_content)
    assert "# Sample Notebook" in result

def test_with_custom_data():
    notebook = create_sample_notebook("Custom Title", ["import numpy", "print('test')"])
    # Use custom notebook content
```

## Test Type Organization

### 🧪 Unit Tests (`tests/unit/`)
- **Focus**: Individual functions and classes
- **Fixtures**: `temp_repo_dir`, `mock_supabase_client`, `test_env_minimal`, data fixtures
- **Isolation**: Fast, no external dependencies

### 🔗 Integration Tests (`tests/integration/`)
- **Focus**: Component interactions with mocked externals
- **Fixtures**: `temp_git_repo`, `mock_all_external_services`, `integration_test_env`
- **Scope**: Test workflows with realistic data

### 🌐 E2E Tests (`tests/e2e/`)
- **Focus**: Full system with real external services
- **Fixtures**: `mcp_client`, `cleanup_test_data`, `environment_check`
- **Requirements**: Live services (Supabase, Neo4j, OpenAI)

## Best Practices

### ✅ DO
- Import fixtures implicitly (they're auto-available via conftest.py)
- Use appropriate fixture scope for your test type
- Combine fixtures for complex scenarios
- Use helper functions for custom data generation

### ❌ DON'T
- Don't import fixtures explicitly from fixture modules
- Don't create new conftest.py files (use the consolidated one)
- Don't duplicate fixture logic (use existing fixtures or helpers)
- Don't use E2E fixtures in unit tests (wrong scope)

## Adding New Fixtures

1. **Choose the right category** based on fixture purpose
2. **Add to the appropriate fixture file** (`repository_fixtures.py`, etc.)
3. **Export in the main conftest.py** by adding to imports
4. **Document the fixture** with clear docstring and example
5. **Update this README** if introducing new patterns

## Migration from Old conftest.py Files

The old pattern of multiple conftest.py files has been replaced with this organized fixture factory system:

**Before:**
```python
# tests/unit/conftest.py
@pytest.fixture
def mock_supabase_client():
    # ... implementation

# tests/integration/conftest.py  
@pytest.fixture
def mock_supabase_client():
    # ... duplicate implementation
```

**After:**
```python
# tests/support/fixtures/mock_fixtures.py
@pytest.fixture
def mock_supabase_client():
    # ... single implementation

# All tests automatically have access via consolidated conftest.py
```

This eliminates duplication and makes fixtures discoverable across the entire test suite.