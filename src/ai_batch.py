"""
AI Batch Processing Module

A wrapper around AI providers' batch APIs for structured output.
"""

from typing import List, Type, TypeVar, Optional, Union
from pydantic import BaseModel
from dotenv import load_dotenv
from .providers.anthropic import AnthropicBatchProvider

# Load environment variables
load_dotenv()

T = TypeVar('T', bound=BaseModel)


def batch(
    messages: List[List[dict]], 
    model: str, 
    response_model: Optional[Type[T]] = None, 
    provider: str = "anthropic",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    wait_for_completion: bool = True,
    poll_interval: int = 10,
    verbose: bool = False
) -> Union[List[T], List[str]]:
    """
    Process multiple message conversations using AI providers' batch processing APIs.
    
    Args:
        messages: List of message conversations, each conversation is a list of message dicts
        model: Model name (e.g., "claude-3-haiku-20240307")
        response_model: Optional Pydantic model class for structured response. If None, returns raw text.
        provider: AI provider name ("anthropic", etc.)
        max_tokens: Maximum tokens per response (default: 1024)
        temperature: Temperature for response generation (default: 0.0)
        wait_for_completion: Whether to wait for batch completion (default: True)
        poll_interval: Polling interval in seconds when waiting (default: 10)
        verbose: Whether to print status updates (default: False)
        
    Returns:
        List of response_model instances if response_model provided, otherwise list of raw text strings
        
    Raises:
        ValueError: If API key is missing, unsupported provider, or batch validation fails
        RuntimeError: If batch processing fails
    """
    if not messages:
        return []
    
    # Get provider instance
    if provider == "anthropic":
        provider_instance = AnthropicBatchProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Provider handles all complexity
    provider_instance.validate_batch(messages, response_model)
    batch_requests = provider_instance.prepare_batch_requests(
        messages, response_model, model=model, max_tokens=max_tokens, temperature=temperature
    )
    batch_id = provider_instance.create_batch(batch_requests)
    
    if not wait_for_completion:
        return batch_id
    
    provider_instance.wait_for_completion(batch_id, poll_interval, verbose)
    results = provider_instance.get_results(batch_id)
    return provider_instance.parse_results(results, response_model)