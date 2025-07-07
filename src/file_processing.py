"""
File Processing Module

Handles PDF and other file formats for batch processing.
"""

import base64
from pathlib import Path
from typing import List, Type, TypeVar, Optional, Union, overload
from pydantic import BaseModel
from .core import batch

T = TypeVar('T', bound=BaseModel)


def pdf_to_document_block(pdf_bytes: bytes) -> dict:
    """Convert PDF bytes to Anthropic document content block format.
    
    Args:
        pdf_bytes: Raw PDF file bytes
        
    Returns:
        Document content block dict
    """
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode('utf-8')
        }
    }


@overload
def batch_files(
    files: List[str],
    prompt: str,
    model: str,
    response_model: Type[T],
    **kwargs
) -> List[T]: ...


@overload
def batch_files(
    files: List[str],
    prompt: str,
    model: str,
    response_model: None = None,
    **kwargs
) -> List[str]: ...


@overload
def batch_files(
    files: List[Path],
    prompt: str,
    model: str,
    response_model: Type[T],
    **kwargs
) -> List[T]: ...


@overload
def batch_files(
    files: List[Path],
    prompt: str,
    model: str,
    response_model: None = None,
    **kwargs
) -> List[str]: ...


@overload
def batch_files(
    files: List[bytes],
    prompt: str,
    model: str,
    response_model: Type[T],
    **kwargs
) -> List[T]: ...


@overload
def batch_files(
    files: List[bytes],
    prompt: str,
    model: str,
    response_model: None = None,
    **kwargs
) -> List[str]: ...


def batch_files(
    files: Union[List[str], List[Path], List[bytes]],
    prompt: str,
    model: str,
    response_model: Optional[Type[T]] = None,
    **kwargs
) -> Union[List[T], List[str], str]:
    """Process multiple PDF files using batch API.
    
    Args:
        files: List of file paths (str or Path) OR list of file bytes.
               All items must be of the same type.
        prompt: Prompt to use for each file
        model: Model name
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments passed to batch()
        
    Returns:
        List of responses
        
    Examples:
        # Using file paths
        results = batch_files(
            files=["doc1.pdf", "doc2.pdf"],
            prompt="Summarize this document",
            model="claude-3-haiku-20240307"
        )
        
        # Using Path objects
        results = batch_files(
            files=[Path("doc1.pdf"), Path("doc2.pdf")],
            prompt="Extract data",
            model="claude-3-haiku-20240307",
            response_model=MyModel
        )
        
        # Using bytes
        pdf_bytes = [open("doc.pdf", "rb").read()]
        results = batch_files(
            files=pdf_bytes,
            prompt="Analyze",
            model="claude-3-haiku-20240307"
        )
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
        
        doc_block = pdf_to_document_block(pdf_bytes)
        
        messages.append([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                doc_block
            ]
        }])
    
    return batch(messages, model, response_model, **kwargs)