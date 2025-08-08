"""
OpenAI client utilities with best practices for error handling and retries.

This module provides a centralized way to interact with OpenAI API
with proper error handling, retries, and configuration management.
"""

import os
import time
from typing import Any, Dict, Optional, Callable, TypeVar
from functools import wraps
from openai import OpenAI

T = TypeVar("T")


def get_openai_client():
    """
    Get configured OpenAI client with proper error handling.

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Use modern OpenAI v1.0+ client instantiation
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    """
    Get the configured model name from environment.

    Returns:
        str: Model name to use for OpenAI calls
    """
    return os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")


def with_retries(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator that adds retry logic to functions.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"All {max_retries} attempts failed")

            # Re-raise the last exception if all retries failed
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


@with_retries(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def generate_chat_completion(
    messages: list[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0,
    response_format: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Any:
    """
    Generate chat completion using OpenAI API with automatic retry logic.

    Args:
        messages: List of message dictionaries for the chat
        model: Model to use (defaults to MODEL_CHOICE env var)
        temperature: Temperature setting for the model
        response_format: Optional response format specification
        **kwargs: Additional arguments to pass to the API

    Returns:
        API response object

    Raises:
        Exception: If all retry attempts fail
    """
    client = get_openai_client()

    if model is None:
        model = get_model_name()

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        **kwargs,
    )
