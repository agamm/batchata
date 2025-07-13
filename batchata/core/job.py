"""Job data model."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type
from pydantic import BaseModel

from ..types import Message


@dataclass
class Job:
    """Configuration for a single AI job.
    
    Either provide messages OR file+prompt, not both.
    
    Attributes:
        id: Unique identifier for the job
        messages: Chat messages for direct message input
        file: File path for file-based input
        prompt: Prompt to use with file input
        model: Model name (e.g., "claude-3-sonnet")
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        response_model: Pydantic model for structured output
        enable_citations: Whether to extract citations from response
    """
    
    id: str  # Unique identifier
    model: str  # Model name (e.g., "claude-3-sonnet")
    messages: Optional[List[Message]] = None  # Chat messages
    file: Optional[Path] = None  # File input
    prompt: Optional[str] = None  # Prompt for file
    temperature: float = 0.7
    max_tokens: int = 1000
    response_model: Optional[Type[BaseModel]] = None  # For structured output
    enable_citations: bool = False
    
    def __post_init__(self):
        """Validate job configuration."""
        if self.messages and (self.file or self.prompt):
            raise ValueError("Provide either messages OR file+prompt, not both")
        
        if self.file and not self.prompt:
            raise ValueError("File input requires a prompt")
        
        if not self.messages and not (self.file and self.prompt):
            raise ValueError("Must provide either messages or file+prompt")