"""
Core Batch Processing Module

A wrapper around AI providers' batch APIs for structured output.
"""

import base64
from pathlib import Path
from typing import List, Type, TypeVar, Optional, Union, Dict, Any
from pydantic import BaseModel
from .providers import get_provider_for_model
from .batch_job import BatchJob

T = TypeVar('T', bound=BaseModel)

# Input type aliases for better clarity
MessageConversations = List[List[dict]]  # List of conversations, each conversation is a list of message dicts
FileInputs = Union[List[str], List[Path], List[bytes]]  # List of file paths (str/Path) or file content (bytes)



def pdf_to_document_block(pdf_bytes: bytes, enable_citations: bool = False) -> Dict[str, Any]:
    """Convert PDF bytes to Anthropic document content block format.
    
    Args:
        pdf_bytes: Raw PDF file bytes
        enable_citations: Whether to enable citations for this document
        
    Returns:
        Document content block dict
    """
    doc_block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode('utf-8')
        }
    }
    
    if enable_citations:
        doc_block["citations"] = {"enabled": True}
        
    return doc_block


def batch(
    model: str,
    messages: Optional[MessageConversations] = None,
    files: Optional[FileInputs] = None,
    prompt: Optional[str] = None,
    response_model: Optional[Type[T]] = None,
    enable_citations: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    verbose: bool = False,
    raw_results_dir: Optional[str] = None
) -> BatchJob:
    """
    Process conversations or files using AI providers' batch processing APIs.
    
    Either messages OR files must be provided, not both.
    When using files, prompt is required.
    Provider is automatically determined from the model name.
    
    Args:
        model: Model name (e.g., "claude-3-haiku-20240307")
        messages: List of message conversations, each conversation is a list of message dicts
        files: List of file paths (str or Path) OR list of file bytes
        prompt: Prompt to use for each file (required when files is provided)
        response_model: Optional Pydantic model class for structured response. If None, returns raw text.
        enable_citations: Whether to enable citations for documents (only applies to files)
        max_tokens: Maximum tokens per response (default: 1024)
        temperature: Temperature for response generation (default: 0.0)
        verbose: Whether to show warnings when accessing incomplete results (default: False)
        raw_results_dir: Optional directory to save raw API responses as JSON files (default: None)
        
    Returns:
        BatchJob instance that can be used to check status and get results
        
    Raises:
        ValueError: If both messages and files are provided, neither are provided,
                   files are provided without prompt, unsupported model, or other validation fails
        RuntimeError: If batch creation fails
        
    Examples:
        # Using messages
        job = batch(
            messages=[[{"role": "user", "content": "Hello"}]],
            model="claude-3-haiku-20240307"
        )
        
        # Using files
        job = batch(
            files=["doc1.pdf", "doc2.pdf"],
            prompt="Summarize this document",
            model="claude-3-haiku-20240307",
            enable_citations=True
        )
    """
    # Validate either/or parameters
    if messages is not None and files is not None:
        raise ValueError("Cannot provide both messages and files. Use either messages or files, not both.")
    
    if messages is None and files is None:
        raise ValueError("Must provide either messages or files.")
    
    if files is not None and prompt is None:
        raise ValueError("prompt is required when using files.")
    
    # Validate flat model requirement for citation mapping
    from .utils import check_flat_model_for_citation_mapping
    check_flat_model_for_citation_mapping(response_model, enable_citations)
    
    # Store original input types for validation
    original_has_files = files is not None
    original_has_messages = messages is not None
    
    # Get provider instance automatically based on model
    provider_instance = get_provider_for_model(model)
    
    # Validate model capabilities first (before conversion)
    provider_instance.validate_model_capabilities(
        model=model,
        has_files=original_has_files,
        has_messages=original_has_messages
    )
    
    # If using files, convert to messages format
    if files is not None:
        messages = []
        for file in files:
            if isinstance(file, bytes):
                pdf_bytes = file
            else:
                pdf_path = Path(file)
                if not pdf_path.exists():
                    raise FileNotFoundError(f"File not found: {pdf_path}")
                pdf_bytes = pdf_path.read_bytes()
            
            doc_block = pdf_to_document_block(pdf_bytes, enable_citations=enable_citations)
            
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    doc_block
                ]
            }])
    
    # Handle empty lists
    if not messages:
        # Return empty BatchJob for consistency
        fake_batch_id = "empty_batch"
        return BatchJob(provider_instance, fake_batch_id, response_model, verbose, False, raw_results_dir, model)
    
    # Provider handles all complexity
    provider_instance.validate_batch(messages, response_model)
    
    # Check if citations are enabled
    batch_requests = provider_instance.prepare_batch_requests(
        messages, response_model, model=model, max_tokens=max_tokens, temperature=temperature, enable_citations=enable_citations
    )
    batch_id = provider_instance.create_batch(batch_requests)
    
    return BatchJob(provider_instance, batch_id, response_model, verbose, enable_citations, raw_results_dir, model)