"""Data-related test fixtures for sample content and metadata."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_notebook_content():
    """Sample Jupyter notebook content for testing."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Sample Notebook\n", "\n", "This is a test notebook."],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["import pandas as pd\n", "print('Hello world')"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


@pytest.fixture
def repo_info():
    """Sample repository information."""
    return {"name": "test-repo", "url": "https://github.com/user/test-repo.git"}


@pytest.fixture
def doc_file_info():
    """Sample documentation file information."""
    return {
        "url": "docs/api.md",
        "markdown": "# API Documentation\n\nThis is API documentation.\n\n```python\nprint('hello')\n```",
    }


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content with various elements."""
    return """# Sample Document

This is a test document with various markdown elements.

## Code Examples

Here's a Python example:

```python
def hello_world():
    print("Hello, World!")
    return "success"
```

## API Reference

### function_name(param1: str, param2: int) -> dict

This function does something useful.

**Parameters:**
- `param1`: Description of param1
- `param2`: Description of param2

**Returns:**
Dictionary with results

## Lists

- Item 1
- Item 2
- Item 3

1. Numbered item 1
2. Numbered item 2
3. Numbered item 3
"""


@pytest.fixture
def sample_code_examples():
    """Sample code examples for testing code extraction."""
    return [
        {
            "language": "python",
            "code": "print('Hello, World!')",
            "context": "Basic greeting example"
        },
        {
            "language": "python", 
            "code": """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
""",
            "context": "Data processing function"
        },
        {
            "language": "javascript",
            "code": "console.log('Hello from JS');",
            "context": "JavaScript greeting"
        }
    ]


@pytest.fixture
def sample_supabase_documents():
    """Sample documents as they would appear in Supabase."""
    return [
        {
            "id": 1,
            "source_id": "github.com/test/repo",
            "url": "README.md",
            "content": "# Test Repository\n\nThis is a test repository.",
            "metadata": {
                "file_type": "md",
                "documentation_category": "readme"
            },
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": 2,
            "source_id": "github.com/test/repo", 
            "url": "docs/api.md",
            "content": "# API Documentation\n\n## Functions\n\n### process()",
            "metadata": {
                "file_type": "md",
                "documentation_category": "api"
            },
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]


@pytest.fixture
def sample_neo4j_nodes():
    """Sample Neo4j node data for testing knowledge graph."""
    return {
        "repository": {
            "name": "test-repo",
            "full_name": "test-user/test-repo",
            "created_at": "2024-01-01T00:00:00Z"
        },
        "classes": [
            {
                "name": "TestClass",
                "full_name": "src.main.TestClass",
                "file_path": "src/main.py",
                "docstring": "Test class for demonstration"
            }
        ],
        "methods": [
            {
                "name": "process",
                "params_list": ["self", "data"],
                "return_type": "dict",
                "class_name": "TestClass"
            }
        ],
        "functions": [
            {
                "name": "utility_function",
                "file_path": "src/utils.py",
                "params_list": ["x", "y"],
                "return_type": "int"
            }
        ]
    }


@pytest.fixture 
def sample_processing_statistics():
    """Sample processing statistics for testing."""
    return {
        "files_processed": 5,
        "chunks_created": 15,
        "code_examples_extracted": 3,
        "processing_time": 2.5,
        "errors": []
    }


def create_sample_notebook(title: str = "Sample Notebook", code_cells: list = None):
    """Helper function to create custom notebook content.
    
    Args:
        title: Notebook title
        code_cells: List of code cell contents
        
    Returns:
        Dictionary representing notebook content
    """
    if code_cells is None:
        code_cells = ["import pandas as pd", "print('Hello world')"]
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", "\n", "This is a test notebook."]
        }
    ]
    
    for i, code in enumerate(code_cells):
        cells.append({
            "cell_type": "code",
            "execution_count": i + 1,
            "metadata": {},
            "outputs": [],
            "source": [code if isinstance(code, list) else [code]]
        })
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python", 
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }