"""Builder for creating test repositories with specific characteristics."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


class RepositoryBuilder:
    """Builder for creating test repositories with configurable characteristics."""
    
    def __init__(self, repo_name: str = "test-repo"):
        self.repo_name = repo_name
        self.repo_path: Optional[Path] = None
        self._temp_dir = None
        self._files: Dict[str, str] = {}
        self._directories: List[str] = []
        self._git_enabled = True
        self._large_files: List[str] = []
        self._malformed_files: List[str] = []
        self._encoding_issues: List[str] = []
    
    def with_readme(self, content: Optional[str] = None) -> "RepositoryBuilder":
        """Add a README file to the repository.
        
        Args:
            content: Custom README content, or default if None
        """
        default_content = f"""# {self.repo_name}

This is a test repository for integration testing.

## Features
- Documentation processing
- Code extraction
- Knowledge graph integration

## Installation

```bash
pip install {self.repo_name}
```

## Quick Start

```python
import {self.repo_name.replace('-', '_')}
result = {self.repo_name.replace('-', '_')}.process()
print(result)
```
"""
        self._files["README.md"] = content or default_content
        return self
    
    def with_license(self, license_type: str = "MIT") -> "RepositoryBuilder":
        """Add a license file.
        
        Args:
            license_type: Type of license (MIT, Apache, GPL, etc.)
        """
        if license_type == "MIT":
            content = f"""MIT License

Copyright (c) 2024 {self.repo_name}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
        else:
            content = f"{license_type} License\n\nCopyright (c) 2024 {self.repo_name}"
        
        self._files["LICENSE"] = content
        return self
    
    def with_changelog(self) -> "RepositoryBuilder":
        """Add a changelog file."""
        content = f"""# Changelog

All notable changes to {self.repo_name} will be documented in this file.

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic functionality
- Documentation

### Changed
- Improved performance

### Fixed
- Bug fixes
"""
        self._files["CHANGELOG.md"] = content
        return self
    
    def with_contributing_guide(self) -> "RepositoryBuilder":
        """Add a contributing guide."""
        content = f"""# Contributing to {self.repo_name}

We welcome contributions! Please follow these guidelines.

## Development Setup

1. Clone the repository
2. Install dependencies
3. Run tests

## Code Style

- Follow PEP 8
- Add type hints
- Write tests

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request
"""
        self._files["CONTRIBUTING.md"] = content
        return self
    
    def with_docs_directory(self, advanced: bool = False) -> "RepositoryBuilder":
        """Add a docs directory with documentation files.
        
        Args:
            advanced: Whether to include advanced documentation structure
        """
        self._directories.append("docs")
        
        # Basic documentation
        self._files["docs/getting_started.md"] = """# Getting Started

## Installation

Install the package using pip:

```bash
pip install test-package
```

## Basic Usage

Here's how to use the package:

```python
from test_package import TestClass

# Create an instance
tc = TestClass("example")

# Process some data
result = tc.process([1, 2, 3, 4, 5])
print(result)
```

## Configuration

You can configure the package with environment variables:

```bash
export TEST_CONFIG="production"
export TEST_DEBUG="false"
```
"""
        
        self._files["docs/api_reference.md"] = """# API Reference

## Classes

### TestClass

Main class for processing data.

#### Constructor

```python
TestClass(name: str)
```

**Parameters:**
- `name`: The name for this instance

#### Methods

##### process(data: List[int]) -> Dict[str, Any]

Process a list of integers.

```python
tc = TestClass("processor")
result = tc.process([1, 2, 3])
# Returns: {"count": 3, "sum": 6, "name": "processor"}
```

**Parameters:**
- `data`: List of integers to process

**Returns:**
- Dictionary with processing results

## Functions

### utility_function(x: int, y: int) -> int

Add two numbers together.

```python
result = utility_function(5, 3)
# Returns: 8
```
"""
        
        if advanced:
            self._directories.extend(["docs/tutorials", "docs/examples", "docs/api"])
            
            self._files["docs/tutorials/advanced_usage.md"] = """# Advanced Usage

## Custom Processors

You can create custom processors by extending the base class:

```python
from test_package import BaseProcessor

class CustomProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.custom_setting = config.get('custom', False)
    
    def process(self, data):
        # Custom processing logic
        result = super().process(data)
        if self.custom_setting:
            result['custom'] = True
        return result
```

## Batch Processing

For large datasets, use batch processing:

```python
from test_package import BatchProcessor

processor = BatchProcessor(batch_size=1000)
for batch in processor.process_batches(large_dataset):
    print(f"Processed batch: {batch}")
```
"""
            
            self._files["docs/examples/real_world.md"] = """# Real World Examples

## Data Analysis Pipeline

```python
import pandas as pd
from test_package import DataProcessor

# Load data
df = pd.read_csv('data.csv')

# Process with our package
processor = DataProcessor()
results = processor.analyze(df)

# Save results
results.to_csv('processed_data.csv')
```

## API Integration

```python
import requests
from test_package import APIProcessor

# Fetch data from API
response = requests.get('https://api.example.com/data')
data = response.json()

# Process API data
processor = APIProcessor()
processed = processor.handle_api_response(data)
```
"""
        
        return self
    
    def with_jupyter_notebook(self, name: str = "tutorial.ipynb") -> "RepositoryBuilder":
        """Add a Jupyter notebook.
        
        Args:
            name: Name of the notebook file
        """
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {self.repo_name} Tutorial\n",
                        "\n",
                        "This notebook demonstrates how to use the package.\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [
                        {
                            "name": "stdout",
                            "output_type": "stream",
                            "text": ["Imported successfully\n"]
                        }
                    ],
                    "source": [
                        f"import {self.repo_name.replace('-', '_')}\n",
                        "print('Imported successfully')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Basic Usage\n",
                        "\n",
                        "Let's start with a simple example:\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": 2,
                    "metadata": {},
                    "outputs": [
                        {
                            "data": {
                                "text/plain": ["{'result': 'success', 'count': 5}"]
                            },
                            "execution_count": 2,
                            "metadata": {},
                            "output_type": "execute_result"
                        }
                    ],
                    "source": [
                        f"processor = {self.repo_name.replace('-', '_')}.TestClass('demo')\n",
                        "result = processor.process([1, 2, 3, 4, 5])\n",
                        "result"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        file_path = name if "/" in name else f"docs/{name}"
        self._files[file_path] = json.dumps(notebook_content, indent=2)
        return self
    
    def with_source_code(self, language: str = "python") -> "RepositoryBuilder":
        """Add source code files for knowledge graph testing.
        
        Args:
            language: Programming language for source files
        """
        if language == "python":
            self._directories.extend(["src", "tests"])
            
            self._files["src/__init__.py"] = f'"""Main package for {self.repo_name}."""\n\n__version__ = "1.0.0"'
            
            self._files["src/main.py"] = f'''"""Main module for {self.repo_name}."""

from typing import List, Dict, Any, Optional


class TestClass:
    """Main test class for processing data."""
    
    def __init__(self, name: str):
        """Initialize with a name.
        
        Args:
            name: Name for this instance
        """
        self.name = name
        self._data_cache = {{}}
    
    def process(self, data: List[int]) -> Dict[str, Any]:
        """Process a list of integers.
        
        Args:
            data: List of integers to process
            
        Returns:
            Dictionary with processing results
        """
        result = {{
            "count": len(data),
            "sum": sum(data),
            "name": self.name
        }}
        
        if data:
            result["avg"] = sum(data) / len(data)
            result["max"] = max(data)
            result["min"] = min(data)
        
        return result
    
    def cache_data(self, key: str, value: Any) -> None:
        """Cache data for later retrieval.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._data_cache[key] = value
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        return self._data_cache.get(key)


def utility_function(x: int, y: int) -> int:
    """Add two numbers together.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Sum of x and y
    """
    return x + y


def process_batch(items: List[str], batch_size: int = 10) -> List[List[str]]:
    """Process items in batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    return batches
'''
            
            self._files["src/utils.py"] = '''"""Utility functions."""

from typing import Any, Dict, List
import json


class ConfigManager:
    """Manage configuration settings."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize with config file.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self._config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)


def format_results(results: Dict[str, Any]) -> str:
    """Format results for display.
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in results.items():
        lines.append(f"{key}: {value}")
    return "\\n".join(lines)
'''
        
        return self
    
    def with_large_file(self, filename: str, size_kb: int = 600) -> "RepositoryBuilder":
        """Add a large file that should be excluded from processing.
        
        Args:
            filename: Name of the large file
            size_kb: Size in kilobytes (default 600KB, over the 500KB limit)
        """
        content = f"# Large File: {filename}\n\n" + "x" * (size_kb * 1024 - 50)
        self._files[filename] = content
        self._large_files.append(filename)
        return self
    
    def with_malformed_notebook(self, filename: str = "malformed.ipynb") -> "RepositoryBuilder":
        """Add a malformed Jupyter notebook.
        
        Args:
            filename: Name of the malformed notebook
        """
        self._files[filename] = "{ invalid json content }"
        self._malformed_files.append(filename)
        return self
    
    def with_encoding_issues(self, filename: str = "encoding_issue.md") -> "RepositoryBuilder":
        """Add a file with encoding issues.
        
        Args:
            filename: Name of the file with encoding issues
        """
        # This will be handled specially during build
        self._encoding_issues.append(filename)
        return self
    
    def without_git(self) -> "RepositoryBuilder":
        """Build repository without git metadata."""
        self._git_enabled = False
        return self
    
    def build(self) -> Path:
        """Build the repository and return its path.
        
        Returns:
            Path to the built repository
        """
        # Create temporary directory
        self._temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self._temp_dir) / self.repo_name
        self.repo_path.mkdir(parents=True)
        
        # Create git metadata if enabled
        if self._git_enabled:
            (self.repo_path / ".git").mkdir()
        
        # Create directories
        for directory in self._directories:
            (self.repo_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create files
        for file_path, content in self._files.items():
            full_path = self.repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle encoding issues
            if file_path in self._encoding_issues:
                # Write with bad encoding
                with open(full_path, 'wb') as f:
                    f.write(content.encode('utf-8') + b'\\xff\\xfe')  # Add invalid bytes
            else:
                full_path.write_text(content)
        
        return self.repo_path
    
    def cleanup(self):
        """Clean up the temporary repository."""
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self.repo_path = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_minimal_repo() -> RepositoryBuilder:
    """Create a minimal repository with just README."""
    return RepositoryBuilder("minimal-repo").with_readme()


def create_standard_repo() -> RepositoryBuilder:
    """Create a standard repository with common files."""
    return (RepositoryBuilder("standard-repo")
            .with_readme()
            .with_license()
            .with_changelog()
            .with_contributing_guide()
            .with_docs_directory()
            .with_source_code())


def create_comprehensive_repo() -> RepositoryBuilder:
    """Create a comprehensive repository with all features."""
    return (RepositoryBuilder("comprehensive-repo")
            .with_readme()
            .with_license()
            .with_changelog()
            .with_contributing_guide()
            .with_docs_directory(advanced=True)
            .with_jupyter_notebook()
            .with_jupyter_notebook("advanced_tutorial.ipynb")
            .with_source_code())


def create_problematic_repo() -> RepositoryBuilder:
    """Create a repository with various issues for error testing."""
    return (RepositoryBuilder("problematic-repo")
            .with_readme()
            .with_large_file("docs/large_document.md")
            .with_malformed_notebook()
            .with_encoding_issues())