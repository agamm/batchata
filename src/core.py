"""
Core Batch Processing Module

A wrapper around AI providers' batch APIs for structured output.
"""

from pathlib import Path
from typing import List, Type, TypeVar, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from .providers.anthropic import AnthropicBatchProvider
from .batch_job import BatchJob

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

T = TypeVar('T', bound=BaseModel)


def batch(
    messages: List[List[dict]], 
    model: str, 
    response_model: Optional[Type[T]] = None, 
    provider: str = "anthropic",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    verbose: bool = False,
    raw_results_dir: Optional[str] = None
) -> BatchJob:
    """
    Process multiple message conversations using AI providers' batch processing APIs.
    
    Args:
        messages: List of message conversations, each conversation is a list of message dicts
        model: Model name (e.g., "claude-3-haiku-20240307")
        response_model: Optional Pydantic model class for structured response. If None, returns raw text.
        provider: AI provider name ("anthropic", etc.)
        max_tokens: Maximum tokens per response (default: 1024)
        temperature: Temperature for response generation (default: 0.0)
        verbose: Whether to show warnings when accessing incomplete results (default: False)
        raw_results_dir: Optional directory to save raw API responses as JSON files (default: None)
        
    Returns:
        BatchJob instance that can be used to check status and get results
        
    Raises:
        ValueError: If API key is missing, unsupported provider, or batch validation fails
        RuntimeError: If batch creation fails
    """
    if not messages:
        # Return empty BatchJob for consistency
        provider_instance = AnthropicBatchProvider()
        fake_batch_id = "empty_batch"
        return BatchJob(provider_instance, fake_batch_id, response_model, verbose, False, raw_results_dir)
    
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
    
    # Check if citations are enabled
    enable_citations = provider_instance.has_citations_enabled(messages)
    
    return BatchJob(provider_instance, batch_id, response_model, verbose, enable_citations, raw_results_dir)