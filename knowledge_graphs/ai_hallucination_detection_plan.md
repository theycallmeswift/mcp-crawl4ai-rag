# AI Coding Assistant Hallucination Detection System

## Overview

This system validates AI-generated code by cross-referencing imports, methods, attributes, and function parameters against a Neo4j knowledge graph of open-source repositories. It detects hallucinations where AI assistants invent non-existent methods, incorrect parameters, or invalid attributes.

## Knowledge Graph Schema Analysis

Based on `parse_repo_into_neo4j.py`, our Neo4j graph contains these node types:

### Node Types
1. **Repository** - Root container for all code
   - Properties: `name`, `created_at`

2. **File** - Individual Python files
   - Properties: `name`, `path`, `module_name`, `line_count`, `created_at`
   - Unique constraint: `path`

3. **Class** - Class definitions
   - Properties: `name`, `full_name`, `created_at`
   - Unique constraint: `full_name`

4. **Method** - Class methods
   - Properties: `name`, `full_name`, `method_id`, `args`, `params_list`, `return_type`, `created_at`
   - Unique identifier: `method_id` (format: `ClassName::method_name`)

5. **Attribute** - Class attributes
   - Properties: `name`, `full_name`, `attr_id`, `type`, `created_at`
   - Unique identifier: `attr_id` (format: `ClassName::attr_name`)

6. **Function** - Top-level functions
   - Properties: `name`, `full_name`, `func_id`, `args`, `params_list`, `return_type`, `created_at`
   - Unique identifier: `func_id` (format: `filepath::function_name`)

### Relationship Types
- `Repository -[:CONTAINS]-> File`
- `File -[:DEFINES]-> Class`
- `File -[:DEFINES]-> Function`
- `Class -[:HAS_METHOD]-> Method`
- `Class -[:HAS_ATTRIBUTE]-> Attribute`
- `File -[:IMPORTS]-> File`

## Hallucination Detection Strategy

### Phase 1: Code Analysis & Import Extraction
1. **Parse AI-generated script** using Python AST
2. **Extract all imports** (both `import` and `from ... import`)
3. **Identify library usage patterns** throughout the code
4. **Map imported names to actual usage** in the script

### Phase 2: Knowledge Graph Validation
1. **Validate imports exist** in our knowledge graph
2. **Check class definitions** are real
3. **Verify method signatures** including parameters and types
4. **Validate attribute access** on classes
5. **Confirm function calls** with correct parameters

### Phase 3: Report Generation
Generate detailed report showing:
- Valid vs invalid imports
- Correct vs incorrect method signatures
- Proper vs improper attribute usage
- Parameter validation results

## Critical Cypher Queries

### 1. Find Repository by Import Module
```cypher
MATCH (f:File)
WHERE f.module_name = $import_module 
   OR f.module_name STARTS WITH $import_module
RETURN f.path, f.module_name
```

### 2. Validate Class Exists
```cypher
MATCH (c:Class)
WHERE c.name = $class_name 
   OR c.full_name = $full_class_name
RETURN c.name, c.full_name
```

### 3. Validate Method with Parameters
```cypher
MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
WHERE c.name = $class_name 
  AND m.name = $method_name
RETURN m.name, m.params_list, m.return_type, m.args
```

### 4. Validate Class Attribute
```cypher
MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
WHERE c.name = $class_name 
  AND a.name = $attr_name
RETURN a.name, a.type
```

### 5. Validate Top-level Function
```cypher
MATCH (f:Function)
WHERE f.name = $function_name
RETURN f.name, f.params_list, f.return_type, f.full_name
```

### 6. Get All Methods for a Class
```cypher
MATCH (c:Class {name: $class_name})-[:HAS_METHOD]->(m:Method)
RETURN m.name, m.params_list, m.return_type
ORDER BY m.name
```

## Implementation Tasks

### Task 1: AST-Based Code Parser
**File**: `ai_script_analyzer.py`
- Parse Python script using AST
- Extract imports with aliases
- Identify class instantiations and method calls
- Map variable assignments to class types
- Track attribute access patterns
- Collect function calls with parameters

### Task 2: Knowledge Graph Validator
**File**: `knowledge_graph_validator.py`  
- Connect to Neo4j database
- Execute validation queries
- Match imported modules to graph repositories
- Validate method signatures and parameters
- Check attribute existence and types
- Cross-reference function calls

### Task 3: Hallucination Reporter
**File**: `hallucination_reporter.py`
- Generate structured validation results
- Categorize findings (valid, invalid, uncertain)
- Provide detailed explanations for failures
- Suggest corrections for hallucinated code
- Export results in multiple formats (JSON, markdown)

### Task 4: Main Detection Engine
**File**: `ai_hallucination_detector.py`
- Orchestrate the entire detection process
- Handle multiple library validation
- Manage error cases gracefully
- Provide CLI interface
- Support batch processing

## Validation Logic

### Import Validation
1. Extract base module name from import
2. Query graph for files matching module pattern
3. If found, mark import as VALID
4. If not found, mark as POTENTIALLY_INVALID (could be external library)

### Method Validation
1. For each method call on imported classes:
   - Find class in knowledge graph
   - Query for method with exact name
   - Compare parameter count and types
   - Validate return type if used

### Attribute Validation  
1. For each attribute access:
   - Find class in knowledge graph
   - Query for attribute with exact name
   - Compare types if type-annotated

### Parameter Validation
1. For each function/method call:
   - Extract passed parameters (positional and keyword)
   - Compare against knowledge graph signature
   - Flag mismatched parameter names/types
   - Check required vs optional parameters

## Error Handling & Edge Cases

### Ambiguous Imports
- Handle `from module import *`
- Resolve aliased imports (`import pandas as pd`)
- Track context across method chains

### Dynamic Code Patterns
- Method calls via `getattr()`
- Dynamic attribute access
- Conditional imports

### External Libraries
- Distinguish between tracked and external libraries
- Provide confidence scores for unknown imports
- Suggest adding libraries to knowledge graph

## Output Format

### JSON Report Structure
```json
{
  "script_path": "path/to/analyzed/script.py", 
  "analysis_timestamp": "2024-01-01T12:00:00Z",
  "libraries_analyzed": [
    {
      "library_name": "pydantic",
      "import_status": "VALID",
      "classes_used": [
        {
          "class_name": "BaseModel",
          "status": "VALID",
          "methods_called": [
            {
              "method_name": "model_validate", 
              "status": "VALID",
              "parameters_provided": ["data"],
              "parameters_expected": ["obj", "strict", "from_attributes"],
              "parameter_validation": "VALID"
            }
          ],
          "attributes_accessed": [
            {
              "attribute_name": "model_fields",
              "status": "VALID", 
              "type_expected": "dict"
            }
          ]
        }
      ],
      "functions_called": [],
      "confidence_score": 0.95
    }
  ],
  "summary": {
    "total_validations": 15,
    "valid_count": 13, 
    "invalid_count": 2,
    "uncertain_count": 0,
    "overall_confidence": 0.87
  },
  "hallucinations_detected": [
    {
      "type": "METHOD_NOT_FOUND",
      "location": "line 45",
      "description": "Method 'nonexistent_method' not found on class 'BaseModel'",
      "suggestion": "Did you mean 'model_validate'?"
    }
  ]
}
```

## Testing Strategy

### Test Cases Required
1. **Perfect Script** - All imports/methods valid
2. **Complete Hallucination** - Non-existent library
3. **Method Hallucination** - Real class, fake method  
4. **Parameter Hallucination** - Real method, wrong parameters
5. **Attribute Hallucination** - Real class, fake attribute
6. **Mixed Scenario** - Some valid, some invalid

### Validation Metrics
- **Precision**: Correctly identified hallucinations / Total identified hallucinations
- **Recall**: Correctly identified hallucinations / Total actual hallucinations  
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence Calibration**: Alignment between confidence scores and actual accuracy

## Performance Considerations

### Query Optimization
- Use indexes on frequently queried properties
- Batch similar queries together
- Cache validation results for repeated patterns
- Limit result sets appropriately

### Scalability
- Support streaming analysis for large scripts
- Parallel validation of independent components
- Configurable timeout limits
- Memory-efficient AST processing