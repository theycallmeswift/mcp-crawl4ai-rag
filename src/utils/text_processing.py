"""
Text chunking, markdown processing utilities.
"""
import json
import re
from typing import List
from pathlib import Path


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks


def convert_notebook_to_markdown(notebook_path: Path) -> str:
    """
    Convert Jupyter notebook to markdown using only standard Python libraries.
    
    Args:
        notebook_path: Path to the .ipynb file
        
    Returns:
        Markdown content as string
        
    Raises:
        Exception: If notebook cannot be parsed or converted
    """
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    markdown_parts = []
    
    # Add notebook title if available
    if 'metadata' in notebook and 'kernelspec' in notebook['metadata']:
        kernel_name = notebook['metadata']['kernelspec'].get('display_name', 'Notebook')
        markdown_parts.append(f"# {kernel_name}\n")
    
    # Process each cell
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        # Join source lines into content
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = str(source)
        
        if cell_type == 'markdown':
            # Add markdown cell content directly
            markdown_parts.append(content)
            
        elif cell_type == 'code':
            # Add code cell as markdown code block
            if content.strip():
                language = _detect_code_language(notebook)
                markdown_parts.append(f"```{language}\n{content}\n```")
                
                # Optionally include outputs
                outputs = cell.get('outputs', [])
                if outputs:
                    for output in outputs:
                        if output.get('output_type') == 'stream':
                            # Add stream output
                            stream_text = ''.join(output.get('text', []))
                            if stream_text.strip():
                                markdown_parts.append(f"```\n{stream_text}\n```")
                        elif output.get('output_type') in ['execute_result', 'display_data']:
                            # Add text representation of output
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                text_output = ''.join(data['text/plain'])
                                if text_output.strip():
                                    markdown_parts.append(f"```\n{text_output}\n```")
        
        # Add spacing between cells
        markdown_parts.append("\n")
    
    return '\n'.join(markdown_parts)


def _detect_code_language(notebook: dict) -> str:
    """
    Detect the programming language from notebook metadata.
    
    Args:
        notebook: Parsed notebook JSON
        
    Returns:
        Language identifier for code blocks
    """
    if 'metadata' in notebook and 'kernelspec' in notebook['metadata']:
        # First try the explicit language field
        language = notebook['metadata']['kernelspec'].get('language', '')
        if language:
            return language.lower()
        
        # Fall back to kernel name with simple processing
        kernel_name = notebook['metadata']['kernelspec'].get('name', '').lower()
        if kernel_name:
            # Handle special case for R
            if kernel_name == 'ir':
                return 'r'
            
            # Remove version numbers (python3 -> python, python2 -> python)
            clean_name = re.sub(r'\d+$', '', kernel_name)
            return clean_name if clean_name else kernel_name
    
    return 'python'  # Default to Python