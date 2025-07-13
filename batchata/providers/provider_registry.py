"""Provider registry - simple reverse map."""

from typing import Dict
from ..exceptions import ProviderNotFoundError

# Import providers to populate registry
from .anthropic import AnthropicProvider

# Global registry: model name -> provider instance
providers: Dict[str, 'Provider'] = {}

# Auto-register providers
_anthropic = AnthropicProvider(auto_register=False)
for model_name in _anthropic.models:
    providers[model_name] = _anthropic


def get_provider(model: str) -> 'Provider':
    """Get provider for model."""
    provider = providers.get(model)
    if not provider:
        available = list(providers.keys())
        raise ProviderNotFoundError(
            f"No provider for model: {model}. Available: {', '.join(available)}"
        )
    return provider