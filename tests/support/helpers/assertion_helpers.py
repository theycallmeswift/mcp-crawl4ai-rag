"""Helper functions for common test assertions."""

from typing import Dict, Any, List, Optional
import json
import shutil
from pathlib import Path


def assert_response_structure(response_data: Dict[str, Any], success: bool = True):
    """Assert that a response has the expected structure.
    
    Args:
        response_data: Response dictionary to validate
        success: Expected success status
    """
    assert isinstance(response_data, dict), "Response must be a dictionary"
    assert "success" in response_data, "Response must have 'success' field"
    assert response_data["success"] is success, f"Expected success={success}"
    
    if success:
        assert "error" not in response_data or response_data["error"] is None
    else:
        assert "error" in response_data, "Failed response must have 'error' field"
        assert response_data["error"], "Error message must not be empty"


def assert_processing_statistics(stats: Dict[str, Any], min_files: int = 0, min_chunks: int = 0):
    """Assert that processing statistics have expected structure and values.
    
    Args:
        stats: Processing statistics dictionary
        min_files: Minimum expected files processed
        min_chunks: Minimum expected chunks created
    """
    required_fields = ["files_processed", "chunks_created"]
    for field in required_fields:
        assert field in stats, f"Statistics must have '{field}' field"
        assert isinstance(stats[field], int), f"'{field}' must be an integer"
        
    assert stats["files_processed"] >= min_files, f"Expected at least {min_files} files processed"
    assert stats["chunks_created"] >= min_chunks, f"Expected at least {min_chunks} chunks created"


def assert_supabase_documents(documents: List[Dict[str, Any]], min_count: int = 1):
    """Assert that Supabase documents have expected structure.
    
    Args:
        documents: List of document dictionaries
        min_count: Minimum expected document count
    """
    assert isinstance(documents, list), "Documents must be a list"
    assert len(documents) >= min_count, f"Expected at least {min_count} documents"
    
    required_fields = ["id", "source_id", "url", "content"]
    for doc in documents:
        for field in required_fields:
            assert field in doc, f"Document must have '{field}' field"
        assert isinstance(doc["content"], str), "Document content must be a string"
        assert len(doc["content"]) > 0, "Document content must not be empty"


def assert_neo4j_nodes(nodes: List[Dict[str, Any]], node_type: str, min_count: int = 1):
    """Assert that Neo4j nodes have expected structure.
    
    Args:
        nodes: List of node dictionaries
        node_type: Expected node type (e.g., 'Class', 'Method', 'Function')
        min_count: Minimum expected node count
    """
    assert isinstance(nodes, list), "Nodes must be a list"
    assert len(nodes) >= min_count, f"Expected at least {min_count} {node_type} nodes"
    
    for node in nodes:
        assert isinstance(node, dict), "Node must be a dictionary"
        assert "name" in node, f"{node_type} node must have 'name' field"
        assert isinstance(node["name"], str), f"{node_type} name must be a string"
        assert len(node["name"]) > 0, f"{node_type} name must not be empty"


def assert_code_examples(examples: List[Dict[str, Any]], min_count: int = 1):
    """Assert that code examples have expected structure.
    
    Args:
        examples: List of code example dictionaries
        min_count: Minimum expected example count
    """
    assert isinstance(examples, list), "Code examples must be a list"
    assert len(examples) >= min_count, f"Expected at least {min_count} code examples"
    
    required_fields = ["language", "code"]
    for example in examples:
        for field in required_fields:
            assert field in example, f"Code example must have '{field}' field"
        assert isinstance(example["code"], str), "Code must be a string"
        assert len(example["code"].strip()) > 0, "Code must not be empty"


def assert_repository_structure(repo_path, required_files: List[str], required_dirs: List[str] = None):
    """Assert that a repository has expected structure.
    
    Args:
        repo_path: Path to repository
        required_files: List of required file paths (relative to repo)
        required_dirs: List of required directory paths (relative to repo)
    """
    from pathlib import Path
    
    repo_path = Path(repo_path)
    assert repo_path.exists(), f"Repository path {repo_path} must exist"
    assert repo_path.is_dir(), f"Repository path {repo_path} must be a directory"
    
    for file_path in required_files:
        full_path = repo_path / file_path
        assert full_path.exists(), f"Required file {file_path} not found"
        assert full_path.is_file(), f"Required path {file_path} is not a file"
    
    if required_dirs:
        for dir_path in required_dirs:
            full_path = repo_path / dir_path
            assert full_path.exists(), f"Required directory {dir_path} not found"
            assert full_path.is_dir(), f"Required path {dir_path} is not a directory"


def assert_json_serializable(data: Any, description: str = "data"):
    """Assert that data can be JSON serialized.
    
    Args:
        data: Data to test
        description: Description for error messages
    """
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"{description} must be JSON serializable: {e}")


def assert_markdown_content(content: str, required_elements: List[str] = None):
    """Assert that markdown content has expected structure.
    
    Args:
        content: Markdown content string
        required_elements: List of required content (headers, code blocks, etc.)
    """
    assert isinstance(content, str), "Markdown content must be a string"
    assert len(content.strip()) > 0, "Markdown content must not be empty"
    
    if required_elements:
        for element in required_elements:
            assert element in content, f"Markdown must contain '{element}'"


def assert_notebook_structure(notebook: Dict[str, Any]):
    """Assert that notebook data has valid Jupyter structure.
    
    Args:
        notebook: Notebook dictionary
    """
    required_fields = ["cells", "metadata", "nbformat", "nbformat_minor"]
    for field in required_fields:
        assert field in notebook, f"Notebook must have '{field}' field"
    
    assert isinstance(notebook["cells"], list), "Notebook cells must be a list"
    assert len(notebook["cells"]) > 0, "Notebook must have at least one cell"
    
    for cell in notebook["cells"]:
        assert "cell_type" in cell, "Cell must have 'cell_type' field"
        assert cell["cell_type"] in ["markdown", "code", "raw"], "Invalid cell type"


def assert_environment_variables(required_vars: List[str]):
    """Assert that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
    """
    import os
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    assert not missing_vars, f"Missing required environment variables: {missing_vars}"


def assert_mock_called_with_pattern(mock_obj, pattern: str, call_index: int = 0):
    """Assert that a mock was called with arguments matching a pattern.
    
    Args:
        mock_obj: Mock object to check
        pattern: String pattern to search for in call arguments
        call_index: Which call to check (default: first call)
    """
    assert mock_obj.called, "Mock was not called"
    assert len(mock_obj.call_args_list) > call_index, f"Mock was not called {call_index + 1} times"
    
    call_args = mock_obj.call_args_list[call_index]
    call_str = str(call_args)
    assert pattern in call_str, f"Pattern '{pattern}' not found in call: {call_str}"


def copy_repo(source_path: Path, target_path: Path):
    """
    Helper to copy repository structure for git clone simulation.
    
    This function safely copies a source repository directory to a target location,
    removing the target directory if it already exists to ensure a clean copy.
    
    Args:
        source_path: Path to the source repository directory
        target_path: Path to the target directory where the repo should be copied
    """
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)


# Convenient aliases for common assertions
assert_valid_response = assert_response_structure
assert_valid_docs = assert_supabase_documents
assert_valid_nodes = assert_neo4j_nodes
assert_valid_repo = assert_repository_structure