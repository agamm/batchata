"""
File Processing Module

Handles PDF and other file formats for batch processing.
"""

import base64
from pathlib import Path
from typing import List, Type, TypeVar, Optional, Union, overload
from pydantic import BaseModel
from .core import batch
from .batch_job import BatchJob

T = TypeVar('T', bound=BaseModel)


def pdf_to_document_block(pdf_bytes: bytes, enable_citations: bool = False) -> dict:
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


def batch_files(
    files: Union[List[str], List[Path], List[bytes]],
    prompt: str,
    model: str,
    response_model: Optional[Type[T]] = None,
    enable_citations: bool = False,
    raw_results_dir: Optional[str] = None,
    **kwargs
) -> BatchJob:
    """Process multiple PDF files using batch API.
    
    Args:
        files: List of file paths (str or Path) OR list of file bytes.
               All items must be of the same type.
        prompt: Prompt to use for each file
        model: Model name
        response_model: Optional Pydantic model for structured output
        enable_citations: Whether to enable citations for documents
        raw_results_dir: Optional directory to save raw API responses as JSON files
        **kwargs: Additional arguments passed to batch()
        
    Returns:
        BatchJob instance that can be used to check status and get results
        
    Examples:
        # Using file paths
        job = batch_files(
            files=["doc1.pdf", "doc2.pdf"],
            prompt="Summarize this document",
            model="claude-3-haiku-20240307"
        )
        results = job.results()  # Returns empty list until complete
        
        # Using Path objects with structured output
        job = batch_files(
            files=[Path("doc1.pdf"), Path("doc2.pdf")],
            prompt="Extract data",
            model="claude-3-haiku-20240307",
            response_model=MyModel
        )
        if job.is_complete():
            results = job.results()
        
        # Using bytes with citations and raw response saving
        pdf_bytes = [open("doc.pdf", "rb").read()]
        job = batch_files(
            files=pdf_bytes,
            prompt="Analyze",
            model="claude-3-haiku-20240307",
            enable_citations=True,
            raw_results_dir="./raw_responses"
        )
        citations = job.citations()  # Returns List[Citation]
    """
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
    
    return batch(messages, model, response_model, raw_results_dir=raw_results_dir, **kwargs)