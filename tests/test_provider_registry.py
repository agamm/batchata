"""Tests for ProviderRegistry."""

import pytest

from batchata.providers import ProviderRegistry, ModelConfig
from batchata.exceptions import ProviderNotFoundError
from tests.mocks import MockProvider


class TestProviderRegistry:
    """Tests for ProviderRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()
    
    def test_singleton_pattern(self):
        """Test that ProviderRegistry is a singleton."""
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()
        assert registry1 is registry2
    
    def test_register_provider(self):
        """Test provider registration."""
        mock_provider = MockProvider(auto_register=False)
        ProviderRegistry.register(mock_provider)
        
        # Check that models are registered
        models = ProviderRegistry.list_models()
        assert "mock-model-basic" in models
        assert "mock-model-advanced" in models
        assert "mock-model-simple" in models
    
    def test_get_provider(self):
        """Test getting provider by model name."""
        mock_provider = MockProvider(auto_register=False)
        ProviderRegistry.register(mock_provider)
        
        # Get provider for a model
        provider = ProviderRegistry.get_provider("mock-model-basic")
        assert isinstance(provider, MockProvider)
        assert provider is mock_provider
    
    def test_get_provider_not_found(self):
        """Test error when provider not found."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            ProviderRegistry.get_provider("unknown-model")
        
        assert "No provider found for model: unknown-model" in str(exc_info.value)
    
    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        models = ProviderRegistry.list_models()
        assert models == []
    
    def test_multiple_providers(self):
        """Test registering multiple providers."""
        # In real usage, different provider classes would be registered
        # Here we'll test that the last registered provider wins for the same class
        provider1 = MockProvider(auto_register=False)
        provider1.models = {"model-a": ModelConfig("model-a", 1000, 100, 0.5)}
        
        ProviderRegistry.register(provider1)
        
        # Check model is available
        models = ProviderRegistry.list_models()
        assert "model-a" in models
        
        # Register another provider (simulating a different provider class)
        # In practice, this would be OpenAIProvider, AnthropicProvider, etc.
        provider2 = MockProvider(auto_register=False)
        provider2.models = {"model-b": ModelConfig("model-b", 2000, 200, 0.6)}
        
        # Clear and re-register to simulate different provider classes
        ProviderRegistry.clear()
        ProviderRegistry.register(provider1)
        
        # Create a custom provider class for testing
        class CustomMockProvider(MockProvider):
            pass
        
        provider3 = CustomMockProvider(auto_register=False)
        provider3.models = {"model-b": ModelConfig("model-b", 2000, 200, 0.6)}
        ProviderRegistry.register(provider3)
        
        # Now both should be available from different providers
        models = ProviderRegistry.list_models()
        assert "model-a" in models
        assert "model-b" in models
        
        # Check correct provider is returned
        assert isinstance(ProviderRegistry.get_provider("model-a"), MockProvider)
        assert isinstance(ProviderRegistry.get_provider("model-b"), CustomMockProvider)