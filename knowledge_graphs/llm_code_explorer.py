"""
LLM-Powered Code Repository Explorer

Uses OpenAI to intelligently navigate and explore code repositories.
Starts with repo overview, selects relevant files, and explores deeply.

Flow:
1. User asks question about repository
2. LLM sees all files and picks most relevant ones (2-3)
3. LLM explores classes/methods in selected files
4. LLM provides detailed answer based on exploration
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class LLMCodeExplorer:
    """AI-powered code exploration using repository graph"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    async def close(self):
        await self.neo4j_driver.close()
    
    async def explore_repository(self, user_question: str, repo_name: str = None) -> Dict[str, Any]:
        """
        Main exploration function: uses LLM to navigate repository and answer questions
        """
        print(f"🤖 Exploring repository to answer: '{user_question}'")
        print("=" * 80)
        
        # Step 1: Get repository overview
        repo_overview = await self._get_repository_overview(repo_name)
        print(f"📁 Found repository: {repo_overview['repo_name']}")
        print(f"📄 Total files: {len(repo_overview['files'])}")
        print()
        
        # Step 2: LLM selects relevant files
        selected_files = await self._select_relevant_files(user_question, repo_overview)
        print(f"🎯 LLM selected {len(selected_files)} relevant files:")
        for file_info in selected_files:
            print(f"   • {file_info['path']} - {file_info['reasoning']}")
        print()
        
        # Step 3: Deep exploration of selected files
        file_explorations = []
        for file_info in selected_files:
            exploration = await self._explore_file_deeply(file_info['path'], user_question)
            file_explorations.append(exploration)
            print(f"🔍 Explored {file_info['path']}: {len(exploration['classes'])} classes, {len(exploration['functions'])} functions")
        print()
        
        # Step 4: LLM synthesizes final answer
        final_answer = await self._synthesize_answer(user_question, repo_overview, selected_files, file_explorations)
        
        print("🎉 Final Answer:")
        print("=" * 80)
        print(final_answer['answer'])
        print()
        
        if final_answer.get('code_examples'):
            print("💡 Code Examples:")
            print(final_answer['code_examples'])
            print()
        
        if final_answer.get('related_files'):
            print("📚 Related Files to Explore:")
            for file_path in final_answer['related_files']:
                print(f"   • {file_path}")
        
        return {
            'question': user_question,
            'repo_overview': repo_overview,
            'selected_files': selected_files,
            'explorations': file_explorations,
            'final_answer': final_answer
        }
    
    async def _get_repository_overview(self, repo_name: str = None) -> Dict[str, Any]:
        """Get high-level repository structure"""
        async with self.neo4j_driver.session() as session:
            # Get repository info
            if repo_name:
                repo_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
                repo_result = await session.run(repo_query, repo_name=repo_name)
                repo_record = await repo_result.single()
                actual_repo_name = repo_record['name'] if repo_record else 'Unknown'
            else:
                # Get any repository (assuming single repo)
                repo_query = "MATCH (r:Repository) RETURN r.name as name LIMIT 1"
                repo_result = await session.run(repo_query)
                repo_record = await repo_result.single()
                actual_repo_name = repo_record['name'] if repo_record else 'Unknown'
            
            # Get all files with summary info
            files_query = """
            MATCH (f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            RETURN f.path as path, f.module_name as module_name, f.line_count as line_count,
                   count(DISTINCT c) as class_count, count(DISTINCT func) as function_count
            ORDER BY f.path
            """
            
            files_result = await session.run(files_query)
            files = []
            async for record in files_result:
                files.append({
                    'path': record['path'],
                    'module_name': record['module_name'],
                    'line_count': record['line_count'],
                    'class_count': record['class_count'],
                    'function_count': record['function_count']
                })
            
            return {
                'repo_name': actual_repo_name,
                'files': files,
                'total_files': len(files),
                'total_classes': sum(f['class_count'] for f in files),
                'total_functions': sum(f['function_count'] for f in files)
            }
    
    async def _select_relevant_files(self, user_question: str, repo_overview: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to select 2-3 most relevant files for the question"""
        
        files_summary = "\n".join([
            f"• {f['path']} ({f['class_count']} classes, {f['function_count']} functions, {f['line_count']} lines)"
            for f in repo_overview['files']
        ])
        
        prompt = f"""You are analyzing the repository "{repo_overview['repo_name']}" to answer this question:
"{user_question}"

Here are all the files in the repository:
{files_summary}

Your task: Select the 2-3 most relevant files that would likely contain information to answer the user's question.

Consider:
- File names and paths that relate to the question
- Files that might contain relevant classes or functionality
- Core/main files vs utility files
- The question's focus (e.g., if asking about "agents", look for agent-related files)

Respond with a JSON array of objects, each containing:
- "path": the file path
- "reasoning": why you selected this file (1-2 sentences)

Example format:
[
  {{"path": "pydantic_ai/agent.py", "reasoning": "Main agent implementation file, likely contains core agent functionality."}},
  {{"path": "pydantic_ai/models/openai.py", "reasoning": "OpenAI model integration, relevant for model-related questions."}}
]

Select 2-3 files maximum. Focus on quality over quantity."""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            selected_files = json.loads(response.choices[0].message.content)
            return selected_files[:3]  # Ensure max 3 files
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM file selection, using fallback")
            # Fallback: select first 2 files
            return [
                {"path": repo_overview['files'][0]['path'], "reasoning": "Fallback selection"},
                {"path": repo_overview['files'][1]['path'], "reasoning": "Fallback selection"}
            ] if len(repo_overview['files']) >= 2 else []
    
    async def _explore_file_deeply(self, file_path: str, user_question: str) -> Dict[str, Any]:
        """Get detailed information about a specific file"""
        async with self.neo4j_driver.session() as session:
            
            # Debug: Check if file exists and what paths are available
            debug_query = "MATCH (f:File) WHERE f.path CONTAINS $partial_path RETURN f.path as path LIMIT 10"
            partial_path = file_path.split('/')[-1].replace('.py', '')  # Get just the filename
            debug_result = await session.run(debug_query, partial_path=partial_path)
            available_paths = []
            async for record in debug_result:
                available_paths.append(record['path'])
            
            # Try to find exact match or closest match
            exact_match = None
            for path in available_paths:
                if path == file_path:
                    exact_match = file_path
                    break
                elif file_path.endswith(path) or path.endswith(file_path.split('/')[-1]):
                    exact_match = path
                    break
            
            if not exact_match and available_paths:
                print(f"⚠️  File '{file_path}' not found exactly. Available similar files: {available_paths[:3]}")
                exact_match = available_paths[0]  # Use first similar file
            elif not available_paths:
                print(f"❌ No files found matching '{file_path}'")
                return {'file_path': file_path, 'classes': [], 'functions': [], 'imports': []}
            
            # Use the matched file path
            actual_file_path = exact_match
            print(f"🔍 Using file: {actual_file_path}")
            # Get classes in this file
            classes_query = """
            MATCH (f:File {path: $file_path})-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            RETURN c.name as class_name, c.full_name as class_full_name,
                   collect(DISTINCT {name: m.name, args: m.args, params_list: m.params_list, return_type: m.return_type, full_name: m.full_name}) as methods,
                   collect(DISTINCT {name: a.name, type: a.type, full_name: a.full_name}) as attributes
            ORDER BY c.name
            """
            
            classes_result = await session.run(classes_query, file_path=actual_file_path)
            classes = []
            async for record in classes_result:
                classes.append({
                    'name': record['class_name'],
                    'full_name': record['class_full_name'],
                    'methods': [m for m in record['methods'] if m['name']],  # Filter out null methods
                    'attributes': [a for a in record['attributes'] if a['name']]  # Filter out null attributes
                })
            
            # Get top-level functions in this file
            functions_query = """
            MATCH (f:File {path: $file_path})-[:DEFINES]->(func:Function)
            RETURN func.name as function_name, func.full_name as function_full_name, 
                   func.args as args, func.params_list as params_list, func.return_type as return_type
            ORDER BY func.name
            """
            
            functions_result = await session.run(functions_query, file_path=actual_file_path)
            functions = []
            async for record in functions_result:
                functions.append({
                    'name': record['function_name'],
                    'full_name': record['function_full_name'],
                    'args': record['args'],
                    'params_list': record['params_list'] or [],
                    'return_type': record['return_type'] or 'Any'
                })
            
            # Get imports for this file
            imports_query = """
            MATCH (f:File {path: $file_path})-[:IMPORTS]->(imported:File)
            RETURN imported.path as imported_path, imported.module_name as imported_module
            ORDER BY imported.path
            """
            
            imports_result = await session.run(imports_query, file_path=actual_file_path)
            imports = []
            async for record in imports_result:
                imports.append({
                    'path': record['imported_path'],
                    'module': record['imported_module']
                })
            
            return {
                'file_path': actual_file_path,
                'classes': classes,
                'functions': functions,
                'imports': imports
            }
    
    async def _synthesize_answer(self, user_question: str, repo_overview: Dict[str, Any], 
                                selected_files: List[Dict[str, Any]], file_explorations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to synthesize final answer based on exploration"""
        
        # Prepare exploration summary
        exploration_summary = []
        for i, exploration in enumerate(file_explorations):
            file_info = selected_files[i]
            summary = f"\n=== {exploration['file_path']} ===\n"
            summary += f"Selection reasoning: {file_info['reasoning']}\n"
            
            if exploration['classes']:
                summary += f"\nClasses ({len(exploration['classes'])}):\n"
                for cl in exploration['classes']:
                    summary += f"• {cl['name']}\n"
                    if cl['methods']:
                        method_sigs = []
                        for m in cl['methods'][:5]:
                            if m.get('params_list'):
                                # Use detailed parameter info if available
                                ret_type = m.get('return_type', 'Any')
                                method_sig = f"{m['name']}({', '.join(m['params_list'])}) -> {ret_type}"
                            else:
                                # Fallback to simple args
                                method_sig = f"{m['name']}({', '.join(m.get('args', []))})"
                            method_sigs.append(method_sig)
                        summary += f"  Methods: {'; '.join(method_sigs)}\n"
                    if cl['attributes']:
                        methods = ', '.join([f"{a['name']}: {a['type']}" for a in cl['attributes'][:3]])
                        summary += f"  Attributes: {methods}\n"
            
            if exploration['functions']:
                summary += f"\nTop-level Functions ({len(exploration['functions'])}):\n"
                for func in exploration['functions']:
                    if func.get('params_list'):
                        # Use detailed parameter info if available
                        ret_type = func.get('return_type', 'Any')
                        func_sig = f"{func['name']}({', '.join(func['params_list'])}) -> {ret_type}"
                    else:
                        # Fallback to simple args
                        func_sig = f"{func['name']}({', '.join(func.get('args', []))})"
                    summary += f"• {func_sig}\n"
            
            if exploration['imports']:
                summary += f"\nImports: {', '.join([imp['module'] for imp in exploration['imports'][:5]])}\n"
            
            exploration_summary.append(summary)
        
        exploration_text = "\n".join(exploration_summary)
        
        prompt = f"""You are a code expert analyzing the repository "{repo_overview['repo_name']}" to answer this question:

"{user_question}"

I explored the repository and examined these files in detail:
{exploration_text}

Based on this exploration, provide a comprehensive answer that:

1. **Directly answers the user's question** with specific guidance
2. **References specific classes, methods, or functions** found in the exploration
3. **Provides code examples or usage patterns** if applicable
4. **Suggests related files** the user might want to explore further

Format your response as JSON:
{{
  "answer": "Detailed answer to the user's question...",
  "code_examples": "Code snippets or usage examples if applicable...",
  "key_classes_methods": ["Class.method", "function_name", ...],
  "related_files": ["other/file/to/explore.py", ...],
  "confidence": "high|medium|low"
}}

Be specific and actionable. If the exploration didn't find relevant information, say so and suggest what to look for."""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {
                "answer": response.choices[0].message.content,
                "code_examples": "",
                "key_classes_methods": [],
                "related_files": [],
                "confidence": "medium"
            }


async def ask_codebase(question: str, repo_name: str = None) -> None:
    """
    Easy function to ask questions about your codebase
    
    Usage:
    await ask_codebase("How do I get messages from an agent execution?")
    """
    load_dotenv()
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    explorer = LLMCodeExplorer(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
    
    try:
        result = await explorer.explore_repository(question, repo_name)
        return result
    finally:
        await explorer.close()


async def main():
    """Example usage with various questions"""
    
    questions = [
        "How do I get messages from an agent execution?",
        "How do I create a new OpenAI model instance?",
        "What are the attributes available for the HTTP MCP instances?",
        "How do I handle responses from the agent?",
        "What providers are available for Pydantic AI?"
    ]
    
    for question in questions:
        print(f"\\n{'='*100}")
        print(f"🔍 EXPLORING: {question}")
        print('='*100)
        
        await ask_codebase(question)
        
        print("\\n" + "="*100 + "\\n")
        await asyncio.sleep(1)  # Brief pause between questions


if __name__ == "__main__":
    asyncio.run(main())