"""Tests for provider registry.

Testing:
1. Provider lookup by model name
2. Registration and discovery
3. Error handling for unknown models
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from batchata.providers import get_provider
from batchata.providers.provider_registry import providers
from batchata.exceptions import ProviderNotFoundError
from tests.mocks.mock_provider import MockProvider


class TestProviderRegistry:
    """Test provider registry functionality."""
    
    @pytest.fixture(autouse=True)
    def mock_api_keys(self):
        """Provide mock API keys for provider initialization."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'OPENAI_API_KEY': 'test-openai-key'
        }):
            yield
    
    def test_provider_lookup_by_model(self):
        """Test looking up providers by model name."""
        # Create a mock provider and register it
        mock_provider = MockProvider()
        
        # Temporarily modify the registry
        original_providers = providers.copy()
        try:
            # Clear and add our mock
            providers.clear()
            providers["mock-model-basic"] = mock_provider
            providers["mock-model-advanced"] = mock_provider
            
            # Test successful lookup
            provider = get_provider("mock-model-basic")
            assert provider is mock_provider
            
            provider2 = get_provider("mock-model-advanced")
            assert provider2 is mock_provider
            
            # Same provider instance for both models
            assert provider is provider2
            
        finally:
            # Restore original registry
            providers.clear()
            providers.update(original_providers)
    
    def test_provider_not_found_error(self):
        """Test error handling for unknown models."""
        # Save original state
        original_providers = providers.copy()
        
        try:
            # Set up limited registry
            providers.clear()
            providers["known-model"] = MockProvider()
            
            # Test unknown model
            with pytest.raises(ProviderNotFoundError) as exc_info:
                get_provider("unknown-model")
            
            error_msg = str(exc_info.value)
            assert "No provider for model: unknown-model" in error_msg
            assert "Available: known-model" in error_msg
            
        finally:
            # Restore
            providers.clear()
            providers.update(original_providers)
    
    def test_registry_lazy_initialization(self):
        """Test that providers are initialized when first accessed."""
        # Clear registry to test lazy initialization
        providers.clear()
        
        # Registry should be empty initially
        assert len(providers) == 0
        
        # Check for some expected Anthropic models
        expected_models = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229"
        ]
        
        # First call should trigger initialization
        provider = get_provider(expected_models[0])
        assert provider.__class__.__name__ == "AnthropicProvider"
        
        # Now registry should be populated
        for model in expected_models:
            assert model in providers
            assert providers[model].__class__.__name__ == "AnthropicProvider"
        
        # Verify all registered models have the same provider instance
        anthropic_providers = [p for p in providers.values() 
                             if p.__class__.__name__ == "AnthropicProvider"]
        if anthropic_providers:
            # All should be the same instance
            first = anthropic_providers[0]
            assert all(p is first for p in anthropic_providers)