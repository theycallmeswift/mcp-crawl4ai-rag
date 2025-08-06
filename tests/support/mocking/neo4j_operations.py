"""Mock utilities for Neo4j operations."""

from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Optional


class Neo4jMocker:
    """Utility for mocking Neo4j database operations."""
    
    def __init__(self):
        self.driver_patcher = None
        self.session_patcher = None
        self.mock_driver = None
        self.mock_session = None
        self._mock_data = {}
    
    def mock_successful_operations(self):
        """Mock successful Neo4j operations."""
        self.mock_session = AsyncMock()
        self.mock_driver = Mock()
        
        # Mock async session operations  
        self.mock_session.run.return_value = AsyncMock()
        self.mock_session.close.return_value = None
        
        # Mock driver operations with async session context manager
        async_session_context = AsyncMock()
        async_session_context.__aenter__.return_value = self.mock_session
        async_session_context.__aexit__.return_value = None
        self.mock_driver.session.return_value = async_session_context
        self.mock_driver.close.return_value = None
        
        # Patch the AsyncGraphDatabase driver creation
        self.driver_patcher = patch("neo4j.AsyncGraphDatabase.driver")
        self.driver_patcher.start().return_value = self.mock_driver
        
        return self.mock_session
    
    def mock_connection_failure(self):
        """Mock Neo4j connection failure."""
        self.driver_patcher = patch("neo4j.AsyncGraphDatabase.driver")
        self.driver_patcher.start().side_effect = ConnectionError("Unable to connect to Neo4j")
        return self.driver_patcher
    
    def mock_query_results(self, results: List[Dict[str, Any]]):
        """Mock query results.
        
        Args:
            results: List of result records to return
        """
        if not self.mock_session:
            self.mock_successful_operations()
        
        # Create mock records
        mock_records = []
        for result in results:
            mock_record = Mock()
            mock_record.data.return_value = result
            mock_record.values.return_value = list(result.values())
            for key, value in result.items():
                setattr(mock_record, key, value)
            mock_records.append(mock_record)
        
        self.mock_session.run.return_value = mock_records
        return self.mock_session
    
    def track_queries(self):
        """Enable tracking of Neo4j queries for verification."""
        if not self.mock_session:
            self.mock_successful_operations()
        
        original_run = self.mock_session.run
        def tracked_run(query, parameters=None):
            self._track_query(query, parameters)
            return original_run(query, parameters)
        
        self.mock_session.run = tracked_run
        return self.mock_session
    
    def get_tracked_queries(self) -> List[Dict[str, Any]]:
        """Get tracked queries.
        
        Returns:
            List of tracked queries with parameters
        """
        return self._mock_data.get("queries", [])
    
    def _track_query(self, query: str, parameters: Optional[Dict[str, Any]]):
        """Track a query for verification."""
        if "queries" not in self._mock_data:
            self._mock_data["queries"] = []
        
        self._mock_data["queries"].append({
            "query": query,
            "parameters": parameters or {}
        })
    
    def stop(self):
        """Stop all Neo4j mocking."""
        if self.driver_patcher:
            self.driver_patcher.stop()
            self.driver_patcher = None
        if self.session_patcher:
            self.session_patcher.stop()
            self.session_patcher = None
        self.mock_driver = None
        self.mock_session = None
        self._mock_data.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_mock_repository_node(
    name: str = "test-repo",
    full_name: str = "test-user/test-repo"
) -> Dict[str, Any]:
    """Create a mock repository node.
    
    Args:
        name: Repository name
        full_name: Full repository name with owner
    
    Returns:
        Mock repository node data
    """
    return {
        "r": {
            "name": name,
            "full_name": full_name,
            "created_at": "2024-01-01T00:00:00Z"
        }
    }


def create_mock_class_node(
    name: str = "TestClass",
    file_path: str = "src/main.py",
    docstring: str = "Test class for demonstration"
) -> Dict[str, Any]:
    """Create a mock class node.
    
    Args:
        name: Class name
        file_path: File path where class is defined
        docstring: Class docstring
    
    Returns:
        Mock class node data
    """
    return {
        "c": {
            "name": name,
            "full_name": f"{file_path.replace('/', '.')}.{name}",
            "file_path": file_path,
            "docstring": docstring
        }
    }


def create_mock_method_node(
    name: str = "process",
    class_name: str = "TestClass",
    params: List[str] = None,
    return_type: str = "dict"
) -> Dict[str, Any]:
    """Create a mock method node.
    
    Args:
        name: Method name
        class_name: Parent class name
        params: Method parameters
        return_type: Return type annotation
    
    Returns:
        Mock method node data
    """
    return {
        "m": {
            "name": name,
            "params_list": params or ["self", "data"],
            "return_type": return_type,
            "class_name": class_name
        }
    }


def create_mock_function_node(
    name: str = "utility_function",
    file_path: str = "src/utils.py",
    params: List[str] = None,
    return_type: str = "int"
) -> Dict[str, Any]:
    """Create a mock function node.
    
    Args:
        name: Function name
        file_path: File path where function is defined
        params: Function parameters
        return_type: Return type annotation
    
    Returns:
        Mock function node data
    """
    return {
        "f": {
            "name": name,
            "file_path": file_path,
            "params_list": params or ["x", "y"],
            "return_type": return_type
        }
    }