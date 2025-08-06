"""Mock utilities for Supabase operations."""

from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional


class SupabaseMocker:
    """Utility for mocking Supabase database operations."""
    
    def __init__(self):
        self.client_patcher = None
        self.mock_client = None
        self._mock_data = {}
    
    def mock_successful_operations(self):
        """Mock successful Supabase operations."""
        self.mock_client = Mock()
        
        # Mock table operations
        mock_table = Mock()
        mock_table.insert.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = self._create_success_response()
        
        self.mock_client.table.return_value = mock_table
        
        # Mock RPC operations
        self.mock_client.rpc.return_value = mock_table
        
        self.client_patcher = patch("src.utils.supabase_client.get_supabase_client")
        self.client_patcher.start().return_value = self.mock_client
        
        return self.mock_client
    
    def mock_connection_failure(self):
        """Mock Supabase connection failure."""
        self.client_patcher = patch("src.utils.supabase_client.get_supabase_client")
        self.client_patcher.start().side_effect = ConnectionError("Unable to connect to Supabase")
        return self.client_patcher
    
    def mock_insert_failure(self):
        """Mock Supabase insert operation failure."""
        self.mock_client = Mock()
        mock_table = Mock()
        mock_table.insert.return_value = mock_table
        mock_table.execute.side_effect = Exception("Insert failed")
        
        self.mock_client.table.return_value = mock_table
        
        self.client_patcher = patch("src.utils.supabase_client.get_supabase_client")
        self.client_patcher.start().return_value = self.mock_client
        
        return self.mock_client
    
    def mock_search_results(self, results: List[Dict[str, Any]]):
        """Mock search operation with specific results.
        
        Args:
            results: List of documents to return from search
        """
        self.mock_client = Mock()
        mock_table = Mock()
        
        # Mock the RPC call for vector search
        mock_rpc = Mock()
        mock_rpc.execute.return_value = self._create_success_response(results)
        self.mock_client.rpc.return_value = mock_rpc
        
        # Mock regular table operations
        mock_table.select.return_value = mock_table
        mock_table.execute.return_value = self._create_success_response(results)
        self.mock_client.table.return_value = mock_table
        
        self.client_patcher = patch("src.utils.supabase_client.get_supabase_client")
        self.client_patcher.start().return_value = self.mock_client
        
        return self.mock_client
    
    def track_operations(self):
        """Enable tracking of Supabase operations for verification."""
        if not self.mock_client:
            self.mock_successful_operations()
        
        # Create a new mock table that tracks operations
        mock_table = Mock()
        
        # Mock and track insert operations
        def tracked_insert(data):
            self._track_operation("insert", data)
            return mock_table
        mock_table.insert = tracked_insert
        
        # Mock and track delete operations
        def tracked_delete():
            self._track_operation("delete", {})
            return mock_table
        mock_table.delete = tracked_delete
        
        # Chain methods return the same table for fluent interface
        mock_table.eq.return_value = mock_table
        setattr(mock_table, 'in_', Mock(return_value=mock_table))  # Handle 'in' keyword
        mock_table.select.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.execute.return_value = self._create_success_response()
        
        # Replace the table method to return our tracking table
        self.mock_client.table.return_value = mock_table
        
        return self.mock_client
    
    def get_tracked_operations(self, operation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tracked operations.
        
        Args:
            operation_type: Filter by operation type ('insert', 'select', etc.)
        
        Returns:
            List of tracked operations
        """
        operations = self._mock_data.get("operations", [])
        if operation_type:
            return [op for op in operations if op["type"] == operation_type]
        return operations
    
    def _track_operation(self, operation_type: str, data: Any):
        """Track an operation for verification."""
        if "operations" not in self._mock_data:
            self._mock_data["operations"] = []
        
        self._mock_data["operations"].append({
            "type": operation_type,
            "data": data
        })
    
    def _create_success_response(self, data: Optional[List[Dict[str, Any]]] = None):
        """Create a successful Supabase response."""
        mock_response = Mock()
        mock_response.data = data or []
        mock_response.error = None
        return mock_response
    
    def stop(self):
        """Stop all Supabase mocking."""
        if self.client_patcher:
            self.client_patcher.stop()
            self.client_patcher = None
            self.mock_client = None
            self._mock_data.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_mock_document_response(
    source_id: str = "github.com/test/repo",
    url: str = "README.md",
    content: str = "Test document content",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a mock document response from Supabase.
    
    Args:
        source_id: Repository source identifier
        url: Document URL
        content: Document content
        metadata: Additional metadata
    
    Returns:
        Mock document response
    """
    return {
        "id": 1,
        "source_id": source_id,
        "url": url,
        "content": content,
        "metadata": metadata or {},
        "created_at": "2024-01-01T00:00:00Z",
        "embedding": [0.1] * 1536
    }


def create_mock_code_example_response(
    source_id: str = "github.com/test/repo",
    code: str = "print('hello')",
    language: str = "python",
    summary: str = "Hello world example"
) -> Dict[str, Any]:
    """Create a mock code example response from Supabase.
    
    Args:
        source_id: Repository source identifier
        code: Code content
        language: Programming language
        summary: Code summary
    
    Returns:
        Mock code example response
    """
    return {
        "id": 1,
        "source_id": source_id,
        "code": code,
        "language": language,
        "summary": summary,
        "created_at": "2024-01-01T00:00:00Z",
        "embedding": [0.1] * 1536
    }