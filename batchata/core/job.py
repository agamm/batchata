"""Job data model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel

from ..types import Message


@dataclass
class Job:
    """Configuration for a single AI job.
    
    Either provide messages OR prompt (with optional file), not both.
    
    Attributes:
        id: Unique identifier for the job
        messages: Chat messages for direct message input
        file: Optional file path for file-based input
        prompt: Prompt text (can be used alone or with file)
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
        
        if not self.messages and not self.prompt:
            raise ValueError("Must provide either messages or prompt")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for state persistence."""
        return {
            "id": self.id,
            "model": self.model,
            "messages": self.messages,
            "file": str(self.file) if self.file else None,
            "prompt": self.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_model": self.response_model.__name__ if self.response_model else None,
            "enable_citations": self.enable_citations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Deserialize from state."""
        # Convert file string back to Path if present
        file_path = None
        if data.get("file"):
            file_path = Path(data["file"])
        
        # Note: response_model reconstruction would need additional logic
        # For now, we'll set it to None during deserialization
        return cls(
            id=data["id"],
            model=data["model"],
            messages=data.get("messages"),
            file=file_path,
            prompt=data.get("prompt"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1000),
            response_model=None,  # Cannot reconstruct from string
            enable_citations=data.get("enable_citations", False)
        )