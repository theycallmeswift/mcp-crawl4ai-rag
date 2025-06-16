# MCP Knowledge Graph Tools Implementation Plan

## Overview
Add two new MCP tools to the `crawl4ai_mcp.py` server to integrate AI hallucination detection and repository parsing functionality from the knowledge_graphs folder.

## Goals
1. **Hallucination Detection Tool**: Check AI-generated scripts for hallucinations using the knowledge graph
2. **Repository Parser Tool**: Parse GitHub repositories into Neo4j knowledge graph for validation

## Current MCP Server Analysis

### Server Structure
- **Framework**: FastMCP with lifespan context management
- **Tool Pattern**: `@mcp.tool()` decorator with `ctx: Context` parameter
- **Response Format**: JSON strings with success/error structure
- **Context Access**: Resources via `ctx.request_context.lifespan_context`

### Existing Tools
1. `crawl_single_page` - Crawl individual web pages
2. `smart_crawl_url` - Intelligent URL crawling with auto-detection
3. `get_available_sources` - List available crawled sources
4. `perform_rag_query` - Vector/hybrid search on documents
5. `search_code_examples` - Search code examples with summaries

## Implementation Tasks

### Task 1: Update Environment Configuration
**File**: `.env.example`
**Changes**: Add Neo4j connection variables

```env
# Neo4j Configuration for Knowledge Graph Tools
# Neo4j connection URI (bolt://localhost:7687 for local, neo4j:// for cloud)
NEO4J_URI=bolt://localhost:7687

# Neo4j username (usually 'neo4j')
NEO4J_USER=neo4j

# Neo4j password for your database
NEO4J_PASSWORD=your_password_here
```

### Task 2: Update Lifespan Context
**File**: `src/crawl4ai_mcp.py`
**Changes**: 
1. Add Neo4j driver to context dataclass
2. Initialize Neo4j connection in lifespan manager
3. Import necessary knowledge graph modules

```python
# Add to imports
import sys
from pathlib import Path

# Add knowledge_graphs folder to path
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor

# Update context dataclass
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    knowledge_validator: Optional[KnowledgeGraphValidator] = None
    repo_extractor: Optional[DirectNeo4jExtractor] = None
    reranking_model: Optional[CrossEncoder] = None
```

### Task 3: Enhance Lifespan Manager
**Function**: `crawl4ai_lifespan`
**Changes**: Initialize Neo4j components

```python
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    # ... existing code ...
    
    # Initialize Neo4j components if credentials are available
    knowledge_validator = None
    repo_extractor = None
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            # Initialize knowledge graph validator
            knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
            await knowledge_validator.initialize()
            
            # Initialize repository extractor
            repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
            await repo_extractor.initialize()
            
        except Exception as e:
            print(f"Failed to initialize Neo4j components: {e}")
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
            reranking_model=reranking_model
        )
    finally:
        # Clean up
        await crawler.__aexit__(None, None, None)
        if knowledge_validator:
            await knowledge_validator.close()
        if repo_extractor:
            await repo_extractor.close()
```

### Task 4: Implement Hallucination Detection Tool
**Tool**: `check_ai_script_hallucinations`

```python
@mcp.tool()
async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    """
    Check an AI-generated Python script for hallucinations using the knowledge graph.
    
    This tool analyzes a Python script for potential AI hallucinations by validating
    imports, method calls, class instantiations, and function calls against a Neo4j
    knowledge graph containing real repository data.
    
    Args:
        ctx: The MCP server provided context
        script_path: Absolute path to the Python script to analyze
    
    Returns:
        JSON string with hallucination detection results and recommendations
    """
    try:
        # Get the knowledge validator from context
        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator
        
        if not knowledge_validator:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph validator not available. Check Neo4j configuration."
            }, indent=2)
        
        # Validate file exists and is Python
        if not os.path.exists(script_path):
            return json.dumps({
                "success": False,
                "error": f"Script not found: {script_path}"
            }, indent=2)
        
        if not script_path.endswith('.py'):
            return json.dumps({
                "success": False,
                "error": "Only Python (.py) files are supported"
            }, indent=2)
        
        # Import and use AI script analyzer
        from ai_script_analyzer import AIScriptAnalyzer
        from hallucination_reporter import HallucinationReporter
        
        # Step 1: Analyze script structure
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)
        
        # Step 2: Validate against knowledge graph
        validation_result = await knowledge_validator.validate_script(analysis_result)
        
        # Step 3: Generate report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)
        
        # Format response
        return json.dumps({
            "success": True,
            "script_path": script_path,
            "overall_confidence": validation_result.overall_confidence,
            "validation_summary": report["validation_summary"],
            "hallucinations_detected": report["hallucinations_detected"],
            "recommendations": report["recommendations"],
            "analysis_metadata": report["analysis_metadata"]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "script_path": script_path,
            "error": str(e)
        }, indent=2)
```

### Task 5: Implement Repository Parser Tool
**Tool**: `parse_github_repository`

```python
@mcp.tool()
async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    """
    Parse a GitHub repository into the Neo4j knowledge graph.
    
    This tool clones a GitHub repository, analyzes its Python files, and stores
    the code structure (classes, methods, functions, imports) in Neo4j for use
    in hallucination detection.
    
    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
    
    Returns:
        JSON string with parsing results and statistics
    """
    try:
        # Get the repository extractor from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        
        if not repo_extractor:
            return json.dumps({
                "success": False,
                "error": "Repository extractor not available. Check Neo4j configuration."
            }, indent=2)
        
        # Validate URL format
        if not repo_url or not isinstance(repo_url, str):
            return json.dumps({
                "success": False,
                "error": "Repository URL is required"
            }, indent=2)
        
        # Check if it looks like a GitHub URL
        if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
            return json.dumps({
                "success": False,
                "error": "Please provide a valid GitHub repository URL"
            }, indent=2)
        
        # Extract repository name for tracking
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Parse the repository
        await repo_extractor.analyze_repository(repo_url)
        
        # Query for statistics (get repository info from Neo4j)
        from neo4j import AsyncGraphDatabase
        
        async with repo_extractor.driver.session() as session:
            # Get repository statistics
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            RETURN 
                r.name as repo_name,
                count(DISTINCT f) as files_count,
                count(DISTINCT c) as classes_count,
                count(DISTINCT m) as methods_count,
                count(DISTINCT func) as functions_count,
                count(DISTINCT a) as attributes_count
            """
            
            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()
            
            if record:
                stats = {
                    "repository": record['repo_name'],
                    "files_processed": record['files_count'],
                    "classes_created": record['classes_count'],
                    "methods_created": record['methods_count'],
                    "functions_created": record['functions_count'],
                    "attributes_created": record['attributes_count']
                }
            else:
                stats = {"error": "Repository not found in database after parsing"}
        
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
            "statistics": stats
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": str(e)
        }, indent=2)
```

### Task 6: Add Helper Functions
**Functions**: Error handling and validation helpers

```python
def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    ])

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    else:
        return f"Neo4j error: {str(error)}"
```

## Implementation Order

1. **Environment Setup** (5 minutes)
   - Update `.env.example` with Neo4j variables
   - Test Neo4j connection locally

2. **Import Integration** (10 minutes)
   - Add knowledge_graphs path to sys.path
   - Import required modules
   - Handle import errors gracefully

3. **Context Enhancement** (15 minutes)
   - Update `Crawl4AIContext` dataclass
   - Modify lifespan manager
   - Add cleanup procedures

4. **Hallucination Tool** (20 minutes)
   - Implement `check_ai_script_hallucinations` tool
   - Add comprehensive error handling
   - Test with sample script

5. **Repository Parser Tool** (20 minutes)
   - Implement `parse_github_repository` tool
   - Add URL validation
   - Include statistics gathering

6. **Testing & Validation** (15 minutes)
   - Test both tools with valid inputs
   - Test error conditions
   - Verify JSON response formats

## Expected Usage

### Hallucination Detection
```python
# MCP Tool Call
{
    "tool": "check_ai_script_hallucinations",
    "arguments": {
        "script_path": "/path/to/my_ai_script.py"
    }
}

# Response
{
    "success": true,
    "script_path": "/path/to/my_ai_script.py",
    "overall_confidence": 0.844,
    "validation_summary": {
        "total_validations": 8,
        "valid_count": 8,
        "invalid_count": 0,
        "not_found_count": 0,
        "hallucination_rate": 0.0
    },
    "hallucinations_detected": [],
    "recommendations": ["No hallucinations detected..."]
}
```

### Repository Parsing
```python
# MCP Tool Call
{
    "tool": "parse_github_repository",
    "arguments": {
        "repo_url": "https://github.com/getzep/graphiti.git"
    }
}

# Response
{
    "success": true,
    "repo_url": "https://github.com/getzep/graphiti.git",
    "repo_name": "graphiti",
    "statistics": {
        "files_processed": 45,
        "classes_created": 23,
        "methods_created": 156,
        "functions_created": 78,
        "attributes_created": 89
    }
}
```

## Error Handling

### Missing Neo4j Configuration
```json
{
    "success": false,
    "error": "Knowledge graph validator not available. Check Neo4j configuration."
}
```

### Invalid File Path
```json
{
    "success": false,
    "error": "Script not found: /invalid/path.py"
}
```

### Repository Parsing Failure
```json
{
    "success": false,
    "error": "Failed to clone repository: authentication required"
}
```

## Benefits

1. **Seamless Integration**: Use existing MCP infrastructure
2. **No Subprocess Overhead**: Direct function imports
3. **Comprehensive Error Handling**: User-friendly error messages
4. **Consistent API**: Follows existing tool patterns
5. **Flexible Configuration**: Optional Neo4j setup
6. **Rich Response Data**: Detailed statistics and recommendations

## Testing Strategy

1. **Unit Tests**: Test individual functions with mock data
2. **Integration Tests**: Test with real Neo4j database
3. **Error Tests**: Verify graceful handling of edge cases
4. **Performance Tests**: Ensure reasonable response times
5. **End-to-End Tests**: Full workflow from MCP client to response

This implementation provides a robust, well-integrated solution for adding knowledge graph functionality to the existing MCP server architecture.