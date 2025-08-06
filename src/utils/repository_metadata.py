"""
Repository URL handling, source ID creation.
"""
import os
import time
from typing import Dict, Any
from urllib.parse import urlparse, urljoin
from pathlib import Path




def create_repository_source_id(repo_url: str) -> str:
    """
    Create repository-level source ID for documentation.
    
    Normalizes both SSH and HTTPS URLs to a consistent format.
    
    Examples:
    - https://github.com/user/repo.git -> github.com/user/repo
    - git@github.com:user/repo.git -> github.com/user/repo
    
    Args:
        repo_url: Repository URL (SSH or HTTPS format)
        
    Returns:
        Repository source ID string in format: domain/user/repo
    """
    try:
        # Handle SSH URLs (git@github.com:user/repo.git)
        if repo_url.startswith('git@'):
            # Extract the part after 'git@' and before ':'
            ssh_parts = repo_url.split('@', 1)[1]  # Remove 'git@'
            if ':' in ssh_parts:
                domain, path = ssh_parts.split(':', 1)
                # Remove .git suffix if present
                path = path.rstrip('.git')
                return f"{domain}/{path}"
        
        # Handle HTTPS/HTTP URLs
        parsed_url = urlparse(repo_url)
        if parsed_url.netloc and parsed_url.path:
            # Remove .git suffix and leading slash if present
            path = parsed_url.path.lstrip('/').rstrip('.git')
            return f"{parsed_url.netloc}/{path}"
            
    except Exception as e:
        print(f"Error creating source ID: {e}")
    
    # Fallback to simple string manipulation
    fallback = repo_url.replace('.git', '').replace('https://', '').replace('http://', '')
    if fallback.startswith('git@'):
        fallback = fallback.replace('git@', '').replace(':', '/')
    return fallback


def create_documentation_url(repo_url: str, doc_path: str) -> str:
    """
    Create URL for individual documentation files (for chunk identification).
    
    Examples:
    - github.com/user/repo/docs/api.md
    - github.com/user/repo/README.md
    
    Args:
        repo_url: Repository URL
        doc_path: Relative path to documentation file
        
    Returns:
        Documentation file URL string
    """
    try:
        parsed_url = urlparse(repo_url)
        # Remove .git suffix if present
        path = parsed_url.path.rstrip('.git')
        return f"{parsed_url.netloc}{path}/{doc_path}"
    except Exception as e:
        print(f"Error creating documentation URL: {e}")
        # Fallback to simple concatenation
        return f"{repo_url.replace('.git', '')}/{doc_path}"


def construct_doc_url(repo_source_id: str, doc_path: str) -> str:
    """
    Construct URL for documentation files using repository source ID.
    
    Args:
        repo_source_id: Repository source ID (e.g., "github.com/user/repo")
        doc_path: Relative path to documentation file
        
    Returns:
        Constructed URL string
    """
    return urljoin(repo_source_id.rstrip('/') + '/', doc_path.lstrip('/'))


def create_documentation_metadata(
    doc_file_info: Dict[str, str], 
    repo_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create enhanced metadata for documentation.
    
    Args:
        doc_file_info: Dictionary with 'url' and 'markdown' keys
        repo_info: Repository information (name, url, etc.)
        
    Returns:
        Enhanced metadata dictionary
    """
    doc_path = doc_file_info["url"]
    content = doc_file_info["markdown"]
    
    # Determine documentation category
    doc_category = "documentation"
    filename = Path(doc_path).name.lower()
    
    if filename in ["readme.md", "readme.rst", "readme.txt"]:
        doc_category = "readme"
    elif "api" in filename or "reference" in filename:
        doc_category = "api"
    elif "tutorial" in filename or "guide" in filename or "getting" in filename:
        doc_category = "tutorial"
    elif "changelog" in filename or "history" in filename or "news" in filename:
        doc_category = "changelog"
    elif "license" in filename:
        doc_category = "license"
    elif "contrib" in filename or "develop" in filename:
        doc_category = "contributing"
    
    # Count code blocks if agentic RAG is enabled
    code_example_count = 0
    if os.getenv("USE_AGENTIC_RAG", "false") == "true":
        # Lazy import to avoid circular dependency
        from .code_extraction import extract_code_blocks

        code_example_count = len(extract_code_blocks(content, min_length=200))
    
    return {
        "repository_name": repo_info.get("name", "unknown"),
        "repository_url": repo_info.get("url", ""),
        "file_type": Path(doc_path).suffix[1:] if Path(doc_path).suffix else "txt",
        "file_path": doc_path,
        "documentation_category": doc_category,
        "content_length": len(content),
        "code_example_count": code_example_count,
        "processed_at": time.time()
    }