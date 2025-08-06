"""Mock utilities for OpenAI API operations."""

from unittest.mock import Mock, patch
from typing import List, Optional


class OpenAIMocker:
    """Utility for mocking OpenAI API operations."""
    
    def __init__(self):
        self.embeddings_patcher = None
        self.chat_patcher = None
        self._call_count = 0
    
    def mock_embeddings(self, embedding_dim: int = 1536):
        """Mock OpenAI embeddings API.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        def create_embedding_response(*args, **kwargs):
            # Extract input from arguments
            input_text = kwargs.get('input', args[0] if args else "")
            
            # Handle both single strings and lists
            if isinstance(input_text, str):
                input_list = [input_text]
            else:
                input_list = input_text
            
            mock_response = Mock()
            mock_response.data = []
            
            for i, text in enumerate(input_list):
                mock_embedding = Mock()
                # Create deterministic embeddings based on text hash
                embedding_value = hash(text) % 1000 / 1000.0
                mock_embedding.embedding = [embedding_value] * embedding_dim
                mock_embedding.index = i
                mock_response.data.append(mock_embedding)
            
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = sum(len(text.split()) for text in input_list)
            
            self._call_count += 1
            return mock_response
        
        self.embeddings_patcher = patch("openai.embeddings.create")
        mock_create = self.embeddings_patcher.start()
        mock_create.side_effect = create_embedding_response
        
        return mock_create
    
    def mock_chat_completion(self, responses: Optional[List[str]] = None):
        """Mock OpenAI chat completion API.
        
        Args:
            responses: List of responses to cycle through
        """
        default_responses = [
            "This is a test response from the mocked OpenAI API.",
            "Here's another response for testing purposes.",
            "Final test response for comprehensive testing."
        ]
        
        response_list = responses or default_responses
        response_index = 0
        
        def create_chat_response(*args, **kwargs):
            nonlocal response_index
            
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            
            # Get the current response and cycle through
            current_response = response_list[response_index % len(response_list)]
            response_index += 1
            
            mock_message.content = current_response
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = len(current_response.split()) * 2  # Rough estimate
            
            self._call_count += 1
            return mock_response
        
        self.chat_patcher = patch("openai.chat.completions.create")
        mock_create = self.chat_patcher.start()
        mock_create.side_effect = create_chat_response
        
        return mock_create
    
    def mock_api_error(self, error_type: str = "rate_limit"):
        """Mock OpenAI API errors.
        
        Args:
            error_type: Type of error to mock ('rate_limit', 'invalid_key', 'timeout')
        """
        if error_type == "rate_limit":
            error = Exception("Rate limit exceeded")
        elif error_type == "invalid_key":
            error = Exception("Invalid API key")
        elif error_type == "timeout":
            error = Exception("Request timeout")
        else:
            error = Exception(f"Unknown error: {error_type}")
        
        self.embeddings_patcher = patch("openai.embeddings.create")
        self.embeddings_patcher.start().side_effect = error
        
        if self.chat_patcher:
            self.chat_patcher.stop()
        self.chat_patcher = patch("openai.chat.completions.create")
        self.chat_patcher.start().side_effect = error
        
        return error
    
    def get_call_count(self) -> int:
        """Get the number of API calls made."""
        return self._call_count
    
    def reset_call_count(self):
        """Reset the API call counter."""
        self._call_count = 0
    
    def stop(self):
        """Stop all OpenAI mocking."""
        if self.embeddings_patcher:
            self.embeddings_patcher.stop()
            self.embeddings_patcher = None
        if self.chat_patcher:
            self.chat_patcher.stop()
            self.chat_patcher = None
        self._call_count = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """Create a deterministic mock embedding for text.
    
    Args:
        text: Input text
        dim: Embedding dimension
    
    Returns:
        Mock embedding vector
    """
    # Create deterministic embedding based on text hash
    base_value = hash(text) % 1000 / 1000.0
    return [base_value + (i * 0.001) % 1.0 for i in range(dim)]


def create_mock_contextual_summary(content: str, context: str = "") -> str:
    """Create a mock contextual summary.
    
    Args:
        content: Content to summarize
        context: Additional context
    
    Returns:
        Mock summary
    """
    word_count = len(content.split())
    has_code = "```" in content
    
    summary = f"This is a mock summary of content with {word_count} words."
    
    if has_code:
        summary += " The content includes code examples."
    
    if context:
        summary += f" Context: {context[:50]}..."
    
    return summary


def make_intermittent_failure(success_function, fail_every: int = 3, exception: Exception = None):
    """Create a function that fails intermittently for testing resilience.
    
    This utility creates a wrapper function that calls the success_function normally
    but raises an exception on every nth call. Useful for testing retry logic
    and error handling in integration tests.
    
    Args:
        success_function: The function to call on successful attempts
        fail_every: Fail on every nth call (default: 3)
        exception: Exception to raise on failure (default: Exception("API rate limit"))
    
    Returns:
        A function that intermittently fails
    
    Example:
        >>> mock_embeddings = openai_mocker.mock_embeddings()
        >>> intermittent_failure = make_intermittent_failure(
        ...     mock_embeddings,
        ...     fail_every=3,
        ...     exception=Exception("API rate limit")
        ... )
        >>> # Use intermittent_failure as a side_effect in your mock
    """
    if exception is None:
        exception = Exception("API rate limit")
    
    call_count = 0
    
    def intermittent_function(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % fail_every == 0:
            raise exception
        return success_function(*args, **kwargs)
    
    return intermittent_function