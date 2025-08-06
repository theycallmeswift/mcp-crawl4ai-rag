"""Mock utilities for git operations."""

from unittest.mock import patch, Mock
from pathlib import Path


class GitMocker:
    """Utility for mocking git clone operations."""
    
    def __init__(self):
        self.clone_patcher = None
        self.mock_run = None
    
    def mock_successful_clone(self, target_repo_structure=None):
        """Mock a successful git clone operation.
        
        Args:
            target_repo_structure: Optional function that sets up the cloned repo structure
        """
        def mock_clone_side_effect(*args, **kwargs):
            # Extract target directory from git clone command
            if len(args) > 0 and isinstance(args[0], list):
                cmd = args[0]
                if "clone" in cmd and len(cmd) >= 3:
                    target_dir = Path(cmd[-1])
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    if target_repo_structure:
                        target_repo_structure(target_dir)
                    else:
                        # Default minimal repo structure
                        (target_dir / ".git").mkdir(exist_ok=True)
                        (target_dir / "README.md").write_text("# Default Test Repo")
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result
        
        # Patch subprocess.run in the knowledge graph module specifically
        self.clone_patcher = patch("knowledge_graphs.parse_repo_into_neo4j.subprocess.run")
        self.mock_run = self.clone_patcher.start()
        self.mock_run.side_effect = mock_clone_side_effect
        return self.mock_run
    
    def mock_failed_clone(self, error_message="Repository not found"):
        """Mock a failed git clone operation."""
        def mock_clone_side_effect(*args, **kwargs):
            from subprocess import CalledProcessError
            # Raise exception like real git clone failure
            raise CalledProcessError(128, ['git', 'clone'], stderr=f"fatal: {error_message}")
        
        # Patch subprocess.run in the knowledge graph module specifically  
        self.clone_patcher = patch("knowledge_graphs.parse_repo_into_neo4j.subprocess.run")
        self.mock_run = self.clone_patcher.start()
        self.mock_run.side_effect = mock_clone_side_effect
        return self.mock_run
    
    def mock_permission_error(self):
        """Mock git clone with permission error."""
        self.clone_patcher = patch("subprocess.run")
        self.mock_run = self.clone_patcher.start()
        self.mock_run.side_effect = PermissionError("Permission denied")
        return self.mock_run
    
    def stop(self):
        """Stop all git mocking."""
        if self.clone_patcher:
            self.clone_patcher.stop()
            self.clone_patcher = None
            self.mock_run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_test_repo_structure(repo_path: Path):
    """Create a realistic test repository structure.
    
    Args:
        repo_path: Path where the repository should be created
    """
    # Git metadata
    (repo_path / ".git").mkdir(exist_ok=True)
    
    # Root files
    (repo_path / "README.md").write_text("""# Test Repository

This is a comprehensive test repository with various documentation types.

## Features
- Multiple documentation formats
- Code examples
- API documentation
""")
    
    (repo_path / "LICENSE").write_text("MIT License\n\nCopyright (c) 2024 Test")
    (repo_path / "CHANGELOG.md").write_text("# Changelog\n\n## v1.0.0\n- Initial release")
    (repo_path / "CONTRIBUTING.md").write_text("# Contributing\n\nHow to contribute to this project.")
    
    # Documentation directory
    docs_dir = repo_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    (docs_dir / "getting_started.md").write_text("""# Getting Started

## Installation

```bash
pip install test-package
```

## Quick Start

```python
import test_package
result = test_package.run()
```
""")
    
    (docs_dir / "api_reference.md").write_text("""# API Reference

## Classes

### TestClass

Main class for testing.

#### Methods

##### process(data)

Process the input data.

```python
tc = TestClass()
result = tc.process([1, 2, 3])
```
""")
    
    # Create a large file that should be excluded
    (docs_dir / "large_file.md").write_text("# Large File\n" + "x" * 600000)  # > 500KB
    
    # Source code for knowledge graph
    src_dir = repo_path / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py").write_text("""
class TestClass:
    \"\"\"Main test class.\"\"\"
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, data: list) -> dict:
        \"\"\"Process input data.\"\"\"
        return {"count": len(data), "name": self.name}

def utility_function(x: int, y: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return x + y
""")