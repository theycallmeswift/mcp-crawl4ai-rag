# AI Hallucination Detection Report

**Script:** `test_script.py`
**Analysis Date:** 2025-06-16T16:11:56.171200+00:00
**Overall Confidence:** 67.14%

## Summary

- **Total Validations:** 7
- **Valid:** 5 (71.4%)
- **Invalid:** 0 (0.0%)
- **Not Found:** 2 (28.6%)
- **Uncertain:** 0 (0.0%)
- **Hallucination Rate:** 28.6%

## 📚 Libraries Analyzed

### __future__
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%

### typing
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%

### dataclasses
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%

### pydantic
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `Field()` (80.0%)
  - ❌ `Field()` (80.0%)
  - ❌ `Field()` (80.0%)
  - ❌ `Field()` (80.0%)
  - ❌ `Field()` (80.0%)

### dotenv
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `load_dotenv()` (80.0%)

### rich.markdown
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `Markdown()` (80.0%)

### rich.console
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Classes Used:**
  - ❌ `Console` (80.0%)

### rich.live
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `Live()` (80.0%)

### asyncio
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%

### os
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Classes Used:**
  - ❌ `os.getenv` (80.0%)
  - ❌ `os.getenv` (80.0%)

### pydantic_ai.providers.openai
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `OpenAIProvider()` (20.0%)

### pydantic_ai.models.openai
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Functions Called:**
  - ❌ `OpenAIModel()` (20.0%)

### pydantic_ai
**Import Status:** VALID
**Import Confidence:** 90.00%
**Classes Used:**
  - ✅ `Agent` (80.0%)
**Methods Called:**
  - ✅ `Agent.run_stream()` (90.0%)
**Attributes Accessed:**
  - ✅ `Agent.tool` (80.0%)

### graphiti_core
**Import Status:** UNCERTAIN
**Import Confidence:** 80.00%
**Classes Used:**
  - ❌ `Graphiti` (80.0%)
**Methods Called:**
  - ❌ `Graphiti.build_indices_and_constraints()` (80.0%)
  - ❌ `Graphiti.close()` (80.0%)

## 💡 Recommendations

- No hallucinations detected in knowledge graph libraries. External library usage appears to be working as expected.
- Overall confidence is moderate. Most validations were for external libraries not in the knowledge graph.

## 📋 Detailed Validation Results

### 🔍 Not Found Items

- **FUNCTION_CALL** `OpenAIModel` (Line 31) - Function 'pydantic_ai.models.openai.OpenAIModel' not found in knowledge graph
- **FUNCTION_CALL** `OpenAIProvider` (Line 31) - Function 'pydantic_ai.providers.openai.OpenAIProvider' not found in knowledge graph

### ✅ Valid Items (Sample)

- **IMPORT** `pydantic_ai` (Line 14) - Module 'pydantic_ai' found in knowledge graph
- **IMPORT** `pydantic_ai` (Line 14) - Module 'pydantic_ai' found in knowledge graph
- **CLASS_INSTANTIATION** `Agent` (Line 34) - Class 'pydantic_ai.Agent' found in knowledge graph
- **METHOD_CALL** `run_stream` (Line 135) - Method 'run_stream' found on class 'pydantic_ai.Agent'
- **ATTRIBUTE_ACCESS** `tool` (Line 52) - 'tool' found as method on class 'pydantic_ai.Agent' (likely used as decorator)
