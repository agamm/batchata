"""JobResult data model."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for state persistence."""
        return {
            "job_id": self.job_id,
            "response": self.response,
            "parsed_response": self.parsed_response if isinstance(self.parsed_response, dict) else None,
            "citations": [asdict(c) for c in self.citations] if self.citations else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobResult':
        """Deserialize from state."""
        # Reconstruct citations if present
        citations = None
        if data.get("citations"):
            citations = [Citation(**c) for c in data["citations"]]
        
        return cls(
            job_id=data["job_id"],
            response=data["response"],
            parsed_response=data.get("parsed_response"),
            citations=citations,
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            error=data.get("error")
        )