"""Model configuration data class."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Configuration for a specific model.
    
    Attributes:
        name: Model identifier (e.g., "claude-3-sonnet")
        max_input_tokens: Maximum input context length
        max_output_tokens: Maximum tokens that can be generated
        batch_discount: Discount factor for batch processing (e.g., 0.5 for 50% off)
        supports_images: Whether the model accepts image inputs
        supports_files: Whether the model accepts file inputs (PDFs, docs, etc.)
        supports_citations: Whether the model supports citation extraction
        supports_structured_output: Whether the model supports structured output
        file_types: List of supported file extensions
    """
    
    name: str  # e.g., "claude-3-sonnet"
    max_input_tokens: int
    max_output_tokens: int
    batch_discount: float  # e.g., 0.5 for 50% off
    supports_images: bool = False
    supports_files: bool = False  # PDFs, docs, etc.
    supports_citations: bool = False
    supports_structured_output: bool = True
    file_types: List[str] = field(default_factory=list)  # [".pdf", ".docx"]