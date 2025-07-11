"""
Test API key validation for providers.
"""

import os
import pytest
from unittest.mock import patch
from batchata.providers.anthropic import AnthropicBatchProvider


class TestAPIKeyValidation:
    """Test API key validation across providers."""

    def test_anthropic_provider_requires_api_key(self):
        """Test that AnthropicBatchProvider raises error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY from environment
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
                AnthropicBatchProvider()

    def test_anthropic_provider_works_with_api_key(self):
        """Test that AnthropicBatchProvider works when API key is set."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            # Should not raise an error
            provider = AnthropicBatchProvider()
            assert provider is not None

    def test_anthropic_provider_empty_api_key_fails(self):
        """Test that empty API key string fails validation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
                AnthropicBatchProvider()

    def test_anthropic_provider_whitespace_api_key_fails(self):
        """Test that whitespace-only API key fails validation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "   "}):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
                AnthropicBatchProvider()