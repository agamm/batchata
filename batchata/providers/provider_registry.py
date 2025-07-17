"""Provider registry - simple reverse map."""

from typing import Dict
from ..exceptions import ProviderNotFoundError

# Global registry: model name -> provider instance
providers: Dict[str, 'Provider'] = {}

def _initialize_providers():
    """Initialize all providers and populate registry."""
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider
    
    for provider_class in [AnthropicProvider, OpenAIProvider]:
        try:
            provider = provider_class(auto_register=False)
            for model_name in provider.models:
                providers[model_name] = provider
        except Exception:
            # Skip providers that can't be initialized (e.g., missing API keys)
            continue

def get_provider(model: str) -> 'Provider':
    """Get provider for model."""
    if not providers:
        _initialize_providers()
    
    provider = providers.get(model)
    if not provider:
        available = list(providers.keys())
        raise ProviderNotFoundError(
            f"No provider for model: {model}. Available: {', '.join(available)}"
        )
    return provider