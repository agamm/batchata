"""
Citations Module

Provides citation models for document references.
"""

from typing import Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """Model for a citation reference."""
    type: str
    cited_text: str
    document_index: int
    document_title: Optional[str] = None
    
    # Location fields
    start_page_number: Optional[int] = None
    end_page_number: Optional[int] = None
    start_char_index: Optional[int] = None
    end_char_index: Optional[int] = None
    start_content_block_index: Optional[int] = None
    end_content_block_index: Optional[int] = None