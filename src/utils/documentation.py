"""
Repository documentation discovery and processing.
"""
import os
from typing import List, Dict, Any
from pathlib import Path
from supabase import Client

from .text_processing import smart_chunk_markdown, convert_notebook_to_markdown
from .repository_metadata import (
    create_repository_source_id, 
    create_documentation_url, 
    construct_doc_url,
    create_documentation_metadata
)
from .document_storage import add_documents_to_supabase, update_source_info, extract_source_summary
from .code_extraction import extract_code_blocks, generate_code_example_summary, add_code_examples_to_supabase


def discover_documentation_files(repo_path: Path) -> List[Path]:
    """
    Discover documentation files with practical filtering.
    
    Includes extensions: .md, .rst, .txt, .mdx, .ipynb
    Excludes directories: tests, __pycache__, .git, venv, node_modules, build, dist, examples
    Size limit: Skip files larger than 500KB
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        List of Path objects for documentation files
    """
    doc_extensions = {'.md', '.mdx', '.rst', '.ipynb', '.txt'}
    exclude_dirs = {
        'tests', 'test', '__pycache__', '.git', 'venv', 'env',
        'node_modules', 'build', 'dist', '.pytest_cache',
        'examples', 'example', 'demo', 'benchmark', '.tox',
        'htmlcov', 'coverage', '.coverage', '.mypy_cache'
    }
    
    doc_files = []
    max_file_size = 500 * 1024  # 500KB limit
    
    try:
        for root, dirs, files in os.walk(repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check extension
                if file_path.suffix.lower() in doc_extensions:
                    try:
                        # Check file size
                        if file_path.stat().st_size <= max_file_size:
                            doc_files.append(file_path)
                        else:
                            print(
                                f"Skipping large file {file_path.relative_to(repo_path)} "
                                f"({file_path.stat().st_size} bytes > {max_file_size} bytes limit)"
                            )
                    except Exception as e:
                        print(f"Error checking file {file_path}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error discovering documentation files: {e}")
        return []
    
    return doc_files


def process_document_files(doc_files: List[Path], repo_path: Path) -> List[Dict[str, str]]:
    """
    Process documentation files with notebook support.
    
    - Jupyter notebooks: Convert to markdown using built-in JSON parsing
    - Regular files: Read as UTF-8 text
    - Return list of {"url": relative_path, "markdown": content}
    
    Args:
        doc_files: List of documentation file paths
        repo_path: Repository root path
        
    Returns:
        List of dictionaries with url and markdown content
    """
    docs_content = []
    
    for doc_file in doc_files:
        try:
            relative_path = str(doc_file.relative_to(repo_path))
            
            if doc_file.suffix == '.ipynb':
                # Process Jupyter notebook
                try:
                    markdown_content = convert_notebook_to_markdown(doc_file)
                except Exception as e:
                    print(f"Error converting notebook {relative_path}: {e}")
                    continue
            else:
                # Process regular text file
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(doc_file, 'r', encoding='latin-1') as f:
                            markdown_content = f.read()
                    except Exception as e:
                        print(f"Error reading file {relative_path} with latin-1 encoding: {e}")
                        continue
                except Exception as e:
                    print(f"Error reading file {relative_path}: {e}")
                    continue
            
            # Only add if we got meaningful content
            if markdown_content.strip():
                docs_content.append({
                    "url": relative_path,
                    "markdown": markdown_content
                })
            
        except Exception as e:
            print(f"Skipping file {doc_file} due to processing error: {e}")
            continue
    
    return docs_content


async def process_repository_docs(
    supabase_client: Client,
    repo_path: Path, 
    repo_name: str,
    repo_url: str = ""
) -> Dict[str, Any]:
    """
    Main entry point for documentation processing.
    
    Args:
        supabase_client: Supabase client for storage
        repo_path: Path to cloned repository
        repo_name: Repository name
        repo_url: Repository URL (optional)
        
    Returns:
        Dictionary with processing results:
        {"files_processed": int, "chunks_created": int, "code_examples_extracted": int}
    """
    print(f"Starting documentation processing for repository: {repo_name}")
    
    try:
        # Step 1: Discover documentation files
        print("Discovering documentation files...")
        doc_files = discover_documentation_files(repo_path)
        print(f"Found {len(doc_files)} documentation files")
        
        if not doc_files:
            return {
                "files_processed": 0,
                "chunks_created": 0,
                "code_examples_extracted": 0,
                "message": "No documentation files found"
            }
        
        # Step 2: Process documentation files
        print("Processing documentation content...")
        docs_content = process_document_files(doc_files, repo_path)
        print(f"Successfully processed {len(docs_content)} files")
        
        if not docs_content:
            return {
                "files_processed": 0,
                "chunks_created": 0,
                "code_examples_extracted": 0,
                "message": "No valid documentation content found"
            }
        
        # Step 3: Prepare data for Supabase
        print("Preparing content for Supabase storage...")
        all_urls = []
        all_chunk_numbers = []
        all_contents = []
        all_metadatas = []
        url_to_full_document = {}
        total_code_examples = 0
        
        # Create single repository source ID
        repo_source_id = create_repository_source_id(repo_url)
        
        repo_info = {
            "name": repo_name,
            "url": repo_url
        }
        
        # Collect all content for repository-level summary
        all_repo_content = []
        
        for doc_info in docs_content:
            doc_path = doc_info["url"]
            content = doc_info["markdown"]
            
            # Create individual documentation URL for chunk identification
            doc_url = create_documentation_url(repo_url, doc_path)
            
            # Create metadata
            metadata = create_documentation_metadata(doc_info, repo_info)
            total_code_examples += metadata["code_example_count"]
            
            # Chunk the content
            chunks = smart_chunk_markdown(content)
            
            # Store full document for contextual embeddings (using doc URL)
            url_to_full_document[doc_url] = content
            
            # Collect content for repository summary
            all_repo_content.append(content)
            
            # Add chunked data - ALL chunks use the same repository source ID
            for i, chunk in enumerate(chunks):
                all_urls.append(doc_url)  # Individual doc URL for chunk identification
                all_chunk_numbers.append(i)
                all_contents.append(chunk)
                # Add source_id to metadata for foreign key reference
                metadata_with_source = metadata.copy()
                metadata_with_source["source_id"] = repo_source_id
                all_metadatas.append(metadata_with_source)
        
        # Step 4: Create single repository source record
        print(f"Creating repository source record: {repo_source_id}")
        
        # Combine all documentation content for repository-level summary
        combined_content = "\n\n".join(all_repo_content)
        
        # Extract repository-level summary and word count
        repo_summary = extract_source_summary(repo_source_id, combined_content[:5000])  # Use first 5000 chars
        repo_word_count = len(combined_content.split())
        
        # Create/update single source record for the entire repository
        update_source_info(supabase_client, repo_source_id, repo_summary, repo_word_count)
        
        # Step 5: Store document chunks in Supabase
        print(f"Storing {len(all_contents)} chunks in Supabase...")
        add_documents_to_supabase(
            supabase_client, 
            all_urls, 
            all_chunk_numbers,
            all_contents, 
            all_metadatas,
            url_to_full_document
        )
        
        # Step 6: Process code examples if enabled
        if os.getenv("USE_AGENTIC_RAG", "false") == "true" and total_code_examples > 0:
            print("Processing code examples...")
            try:
                # Extract and store code examples - all using the same repository source ID
                for doc_info in docs_content:
                    content = doc_info["markdown"]
                    
                    code_blocks = extract_code_blocks(content)
                    if code_blocks:
                        summaries = []
                        for block in code_blocks:
                            summary = generate_code_example_summary(
                                block['code'], 
                                block['context_before'], 
                                block['context_after']
                            )
                            summaries.append(summary)
                        
                        # Use the repository source ID for all code examples
                        codes_only = [block['code'] for block in code_blocks]
                        urls = [construct_doc_url(repo_source_id, doc_info['url']) for _ in code_blocks]
                        chunk_numbers = list(range(len(code_blocks)))
                        metadatas = [
                            {
                                "file_path": doc_info['url'],
                                "file_type": doc_info['url'].split('.')[-1] if '.' in doc_info['url'] else 'unknown',
                                "language": block.get('language', ''),
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split()),
                                "chunk_index": i,
                                "repository_url": repo_url,
                                "repository_name": repo_name,
                                "documentation_category": "code_example"
                            }
                            for i, block in enumerate(code_blocks)
                        ]
                        add_code_examples_to_supabase(supabase_client, urls, chunk_numbers, codes_only, summaries, metadatas, repo_source_id)
                        
            except Exception as e:
                print(f"Error processing code examples: {e}")
                # Continue without failing the entire process
        
        print(f"Documentation processing completed for {repo_name}")
        
        return {
            "files_processed": len(docs_content),
            "chunks_created": len(all_contents),
            "code_examples_extracted": total_code_examples,
            "message": f"Successfully processed {len(docs_content)} documentation files"
        }
        
    except Exception as e:
        print(f"Error processing repository documentation: {e}")
        return {
            "files_processed": 0,
            "chunks_created": 0,
            "code_examples_extracted": 0,
            "error": str(e)
        }