"""JobResult data model."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

from ..types import Citation


@dataclass
class JobResult:
    """Result from a completed AI job.
    
    Attributes:
        job_id: ID of the job this result is for
        response: Raw text response from the model
        parsed_response: Structured output (if response_model was used)
        citations: Extracted citations (if enable_citations was True)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cost_usd: Total cost in USD
        error: Error message if job failed
    """
    
    job_id: str
    response: str  # Raw text response
    parsed_response: Optional[Union[BaseModel, Dict]] = None  # Structured output or error dict
    citations: Optional[List[Citation]] = None  # Extracted citations
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None  # Error message if failed
    
    @property
    def is_success(self) -> bool:
        """Whether the job completed successfully."""
        return self.error is None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens