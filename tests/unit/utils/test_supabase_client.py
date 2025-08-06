"""
Unit tests for supabase client utilities.
"""

import pytest
from unittest.mock import patch

from src.utils.supabase_client import get_supabase_client


class TestGetSupabaseClient:
    """Tests for get_supabase_client function."""

    def test_get_supabase_client_success(self, monkeypatch):
        """Test successful client creation with valid environment variables."""
        # Setup
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")

        with patch("src.utils.supabase_client.create_client") as mock_create:
            mock_client = "mock_client"
            mock_create.return_value = mock_client

            # Exercise
            result = get_supabase_client()

            # Verify
            assert result == mock_client
            mock_create.assert_called_once_with("https://test.supabase.co", "test-service-key")

    def test_get_supabase_client_missing_url(self, monkeypatch):
        """Test error when SUPABASE_URL is missing."""
        # Setup
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")
        monkeypatch.delenv("SUPABASE_URL", raising=False)

        # Exercise & Verify
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            get_supabase_client()

    def test_get_supabase_client_missing_service_key(self, monkeypatch):
        """Test error when SUPABASE_SERVICE_KEY is missing."""
        # Setup
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        # Exercise & Verify
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            get_supabase_client()

    def test_get_supabase_client_both_missing(self, monkeypatch):
        """Test error when both environment variables are missing."""
        # Setup
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        # Exercise & Verify
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            get_supabase_client()