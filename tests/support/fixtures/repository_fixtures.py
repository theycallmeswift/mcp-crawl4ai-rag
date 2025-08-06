"""Repository-related test fixtures."""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def temp_repo_dir():
    """Create a temporary directory structure mimicking a repository.
    
    This is the standard repository structure used in unit tests.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Create various documentation files
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()

        # README file
        (repo_path / "README.md").write_text(
            "# Test Repository\n\nThis is a test repository."
        )

        # API documentation
        (docs_dir / "api_reference.md").write_text(
            "# API Reference\n\nAPI documentation here."
        )

        # Tutorial
        (docs_dir / "getting_started.md").write_text(
            "# Getting Started\n\nTutorial content."
        )

        # Text file
        (repo_path / "LICENSE.txt").write_text("MIT License\n\nCopyright notice.")

        # RST file
        (docs_dir / "changelog.rst").write_text("Changelog\n=========\n\nVersion 1.0.0")

        # Large file (over 500KB)
        large_content = "x" * (600 * 1024)  # 600KB
        (repo_path / "large_file.md").write_text(large_content)

        # Directories to exclude
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_file.md").write_text("Test documentation")

        (repo_path / "__pycache__").mkdir()
        (repo_path / "node_modules").mkdir()
        (repo_path / "build").mkdir()

        yield repo_path


@pytest.fixture
def temp_git_repo():
    """Create a temporary directory with git repository structure.
    
    This is a more comprehensive repository structure used in integration tests.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create a realistic repository structure
        (repo_path / ".git").mkdir()
        (repo_path / "README.md").write_text("# Test Repository\n\nThis is a test repository.")
        (repo_path / "LICENSE").write_text("MIT License")
        
        # Create documentation structure
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "getting_started.md").write_text("# Getting Started\n\nHow to get started.")
        (docs_dir / "api_reference.md").write_text("# API Reference\n\n## Functions\n\n### function_name()\n\nDescription.")
        
        # Create code examples
        (docs_dir / "examples.md").write_text("""# Examples

## Basic Usage

```python
import example
result = example.process()
print(result)
```

## Advanced Usage

```python
from example import AdvancedProcessor

processor = AdvancedProcessor()
data = processor.transform(input_data)
```
""")
        
        # Create a Jupyter notebook
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Example Notebook\n\nThis is an example notebook."]
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})"]
                }
            ],
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
        (docs_dir / "tutorial.ipynb").write_text(json.dumps(notebook_content))
        
        # Create source code files for knowledge graph
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text("""
class ExampleClass:
    def __init__(self, name: str):
        self.name = name
    
    def process(self, data: list) -> dict:
        return {"processed": len(data)}

def helper_function(x: int, y: int) -> int:
    return x + y
""")
        
        yield repo_path


# Configuration fixtures for E2E tests
@pytest.fixture(scope="session")
def test_repository_config():
    """Test repository configuration."""
    from tests.support.database_helpers import TEST_REPOSITORIES
    return TEST_REPOSITORIES["mcp_crawl4ai_rag"]


@pytest.fixture(scope="session")
def minimal_repository_config():
    """Minimal test repository configuration."""
    from tests.support.database_helpers import TEST_REPOSITORIES
    return TEST_REPOSITORIES["hello_world"]


def create_temp_repo_with_structure(structure: Dict[str, Any]) -> Path:
    """Helper function to create custom repository structures.
    
    Args:
        structure: Dictionary defining the repository structure
        
    Returns:
        Path to the created repository
        
    Example:
        structure = {
            "README.md": "# Custom Repo",
            "docs/": {
                "api.md": "# API Documentation"
            }
        }
    """
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    
    def create_from_dict(base_path: Path, items: Dict[str, Any]):
        for name, content in items.items():
            if name.endswith("/"):
                # Directory
                dir_path = base_path / name.rstrip("/")
                dir_path.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    create_from_dict(dir_path, content)
            else:
                # File
                file_path = base_path / name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, str):
                    file_path.write_text(content)
                elif isinstance(content, dict):
                    file_path.write_text(json.dumps(content, indent=2))
    
    create_from_dict(repo_path, structure)
    return repo_path