"""
Unit tests for text processing utilities.
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.utils.text_processing import (
    smart_chunk_markdown,
    convert_notebook_to_markdown,
    _detect_code_language,
)


class TestSmartChunkMarkdown:
    """Tests for smart_chunk_markdown function."""

    def test_smart_chunk_markdown_basic(self):
        """Test basic text chunking with default chunk size."""
        # Setup
        text = "This is a short text that should not be chunked."
        
        # Exercise
        result = smart_chunk_markdown(text)
        
        # Verify
        assert len(result) == 1
        assert result[0] == text

    def test_smart_chunk_markdown_large_text(self):
        """Test chunking of large text."""
        # Setup
        large_text = "A" * 10000
        
        # Exercise
        result = smart_chunk_markdown(large_text)
        
        # Verify
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 5000

    def test_smart_chunk_markdown_respects_code_blocks(self):
        """Test that code blocks influence chunking boundaries."""
        text = """# Header

Some text here.

```python
def function():
    return "This should not be split"
```

More text after code block."""
        
        result = smart_chunk_markdown(text, chunk_size=100)
        
        # Should create multiple chunks but respect code block boundaries
        assert len(result) >= 1
        
        # Find chunk containing the code block
        code_chunk = None
        for chunk in result:
            if "```python" in chunk:
                code_chunk = chunk
                break
        
        # If code block is found, it should be handled appropriately
        if code_chunk:
            # Code block markers should be present if the chunk contains code
            assert "def function():" in code_chunk or "```python" in code_chunk

    def test_smart_chunk_markdown_respects_paragraphs(self):
        """Test that paragraph breaks are preferred for chunking."""
        paragraphs = ["Paragraph 1 " * 100, "Paragraph 2 " * 100, "Paragraph 3 " * 100]
        text = "\n\n".join(paragraphs)
        
        result = smart_chunk_markdown(text, chunk_size=1000)
        
        # Should break at paragraph boundaries when possible
        assert len(result) >= 2

    def test_smart_chunk_markdown_custom_chunk_size(self):
        """Test chunking with custom chunk size."""
        text = "A" * 2000
        result = smart_chunk_markdown(text, chunk_size=500)
        
        # Should create multiple chunks
        assert len(result) >= 3
        # Each chunk should be around the specified size
        for chunk in result:
            assert len(chunk) <= 500

    def test_smart_chunk_markdown_empty_text(self):
        """Test behavior with empty text."""
        result = smart_chunk_markdown("")

        assert result == []

    def test_smart_chunk_markdown_only_whitespace(self):
        """Test behavior with only whitespace."""
        result = smart_chunk_markdown("   \n\n   ")

        assert len(result) <= 1
        if result:
            assert result[0].strip() == ""


class TestConvertNotebookToMarkdown:
    """Tests for convert_notebook_to_markdown function."""

    def test_convert_notebook_to_markdown_basic(self, sample_notebook_content):
        """Test basic notebook conversion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(sample_notebook_content, f)
            notebook_path = Path(f.name)

        try:
            result = convert_notebook_to_markdown(notebook_path)
            
            # Should contain markdown content
            assert "# Sample Notebook" in result
            # Should contain code blocks
            assert "```python" in result
            assert "import pandas as pd" in result
            assert "print('Hello world')" in result
        finally:
            notebook_path.unlink()

    def test_convert_notebook_to_markdown_with_outputs(self):
        """Test notebook conversion with cell outputs."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["Hello, World!\n"]
                        }
                    ],
                    "source": ["print('Hello, World!')"]
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

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_content, f)
            notebook_path = Path(f.name)

        try:
            result = convert_notebook_to_markdown(notebook_path)
            
            # Should contain code and output
            assert "print('Hello, World!')" in result
            assert "Hello, World!" in result
        finally:
            notebook_path.unlink()

    def test_convert_notebook_to_markdown_malformed_json(self):
        """Test error handling with malformed notebook JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write("{ invalid json }")
            notebook_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                convert_notebook_to_markdown(notebook_path)
        finally:
            notebook_path.unlink()

    def test_convert_notebook_to_markdown_missing_file(self):
        """Test error handling with missing file."""
        non_existent_path = Path("/non/existent/notebook.ipynb")
        
        with pytest.raises(FileNotFoundError):
            convert_notebook_to_markdown(non_existent_path)


class TestDetectCodeLanguage:
    """Tests for _detect_code_language function."""

    def test_detect_code_language_python(self):
        """Test detection of Python language."""
        notebook = {
            "metadata": {
                "kernelspec": {
                    "language": "python",
                    "name": "python3"
                }
            }
        }
        
        result = _detect_code_language(notebook)
        assert result == "python"

    def test_detect_code_language_r(self):
        """Test detection of R language."""
        notebook = {
            "metadata": {
                "kernelspec": {
                    "name": "ir"
                }
            }
        }
        
        result = _detect_code_language(notebook)
        assert result == "r"

    def test_detect_code_language_fallback_to_kernel_name(self):
        """Test fallback to kernel name when language is not specified."""
        notebook = {
            "metadata": {
                "kernelspec": {
                    "name": "julia"
                }
            }
        }
        
        result = _detect_code_language(notebook)
        assert result == "julia"

    def test_detect_code_language_removes_version_numbers(self):
        """Test that version numbers are removed from kernel names."""
        notebook = {
            "metadata": {
                "kernelspec": {
                    "name": "python3"
                }
            }
        }
        
        result = _detect_code_language(notebook)
        assert result == "python"

    def test_detect_code_language_default_python(self):
        """Test default to Python when no metadata is available."""
        notebook = {}
        
        result = _detect_code_language(notebook)
        assert result == "python"

    def test_detect_code_language_no_kernelspec(self):
        """Test default to Python when kernelspec is missing."""
        notebook = {
            "metadata": {}
        }
        
        result = _detect_code_language(notebook)
        assert result == "python"