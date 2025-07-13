"""Type definitions for the batch processing library."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any


@dataclass
class Citation:
    """Represents a citation extracted from an AI response."""
    
    text: str  # The cited text
    source: str  # Source identifier (e.g., page number, section)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


@dataclass
class FileContent:
    """Represents processed file content."""
    
    text: str  # Extracted text content
    mime_type: str  # MIME type of the file
    metadata: Optional[Dict[str, Any]] = None  # File metadata (pages, size, etc.)


MessageRole = str  # "user", "assistant", "system"
MessageContent = Union[str, List[Dict[str, Any]]]  # Text or multi-modal content
Message = Dict[str, Any]  # Complete message dict