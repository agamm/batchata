"""
AI Provider Modules

Contains provider-specific implementations for different AI services.
"""

from .base import BaseBatchProvider
from .anthropic import AnthropicBatchProvider

# Registry of all available providers
PROVIDERS = [
    AnthropicBatchProvider,
]

def get_provider_for_model(model: str) -> BaseBatchProvider:
    """
    Get the appropriate provider instance for a given model.
    
    Args:
        model: Model name (e.g., "claude-3-haiku-20240307")
        
    Returns:
        Provider instance that supports the model
        
    Raises:
        ValueError: If no provider supports the model
    """
    for provider_class in PROVIDERS:
        if model in provider_class.get_supported_models():
            return provider_class()
    
    # Build helpful error message with all supported models
    all_models = set()
    for provider_class in PROVIDERS:
        all_models.update(provider_class.get_supported_models())
    
    raise ValueError(
        f"No provider supports model '{model}'. "
        f"Supported models: {sorted(all_models)}"
    )

__all__ = [
    "BaseBatchProvider",
    "AnthropicBatchProvider",
    "get_provider_for_model",
]