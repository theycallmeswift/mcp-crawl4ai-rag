"""Helper functions for managing environment variables in tests."""

import os
from typing import Dict, Optional
from contextlib import contextmanager


class EnvironmentManager:
    """Manager for temporarily overriding environment variables in tests."""

    def __init__(self):
        self._original_values: Dict[str, Optional[str]] = {}

    def override(self, **env_vars) -> None:
        """
        Override environment variables, storing original values.

        Args:
            **env_vars: Key-value pairs of environment variables to set
        """
        for key, value in env_vars.items():
            # Store original value (might be None if not set)
            if key not in self._original_values:
                self._original_values[key] = os.environ.get(key)

            # Set new value
            os.environ[key] = str(value)

    def restore(self) -> None:
        """Restore all environment variables to their original values."""
        for key, original_value in self._original_values.items():
            if original_value is None:
                # Variable wasn't set originally, remove it
                os.environ.pop(key, None)
            else:
                # Restore original value
                os.environ[key] = original_value

        # Clear the tracking
        self._original_values.clear()

    def get_current_values(self, *keys: str) -> Dict[str, Optional[str]]:
        """
        Get current values of specified environment variables.

        Args:
            *keys: Environment variable names to get

        Returns:
            Dictionary of current values
        """
        return {key: os.environ.get(key) for key in keys}


@contextmanager
def temporary_env(**env_vars):
    """
    Context manager for temporarily setting environment variables.

    Usage:
        with temporary_env(MY_VAR="value", ANOTHER_VAR="another"):
            # Environment variables are set here
            do_something()
        # Environment variables are restored here

    Args:
        **env_vars: Key-value pairs of environment variables to set
    """
    manager = EnvironmentManager()
    try:
        manager.override(**env_vars)
        yield manager
    finally:
        manager.restore()
