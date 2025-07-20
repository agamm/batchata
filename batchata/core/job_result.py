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
        raw_response: Raw text response from the model (None for failed jobs)
        parsed_response: Structured output (if response_model was used)
        citations: Extracted citations (if enable_citations was True)
        citation_mappings: Maps field names to relevant citations (if response_model used)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cost_usd: Total cost in USD
        error: Error message if job failed
        batch_id: ID of the batch this job was part of (for mapping to raw files)
    """
    
    job_id: str
    raw_response: Optional[str] = None  # Raw text response (None for failed jobs)
    parsed_response: Optional[Union[BaseModel, Dict]] = None  # Structured output or error dict
    citations: Optional[List[Citation]] = None  # Extracted citations
    citation_mappings: Optional[Dict[str, List[Citation]]] = None  # Field -> citations mapping
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None  # Error message if failed
    batch_id: Optional[str] = None  # Batch ID for mapping to raw files
    
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
        # Handle parsed_response serialization
        parsed_response = None
        if self.parsed_response is not None:
            if isinstance(self.parsed_response, dict):
                parsed_response = self.parsed_response
            elif isinstance(self.parsed_response, BaseModel):
                parsed_response = self.parsed_response.model_dump()
            else:
                parsed_response = str(self.parsed_response)
        
        # Handle citation_mappings serialization
        citation_mappings = None
        if self.citation_mappings:
            citation_mappings = {
                field: [asdict(c) for c in citations]
                for field, citations in self.citation_mappings.items()
            }
        
        return {
            "job_id": self.job_id,
            "raw_response": self.raw_response,
            "parsed_response": parsed_response,
            "citations": [asdict(c) for c in self.citations] if self.citations else None,
            "citation_mappings": citation_mappings,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "batch_id": self.batch_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobResult':
        """Deserialize from state."""
        # Reconstruct citations if present
        citations = None
        if data.get("citations"):
            citations = [Citation(**c) for c in data["citations"]]
        
        # Reconstruct citation_mappings if present
        citation_mappings = None
        if data.get("citation_mappings"):
            citation_mappings = {
                field: [Citation(**c) for c in citations]
                for field, citations in data["citation_mappings"].items()
            }
        
        return cls(
            job_id=data["job_id"],
            raw_response=data["raw_response"],
            parsed_response=data.get("parsed_response"),
            citations=citations,
            citation_mappings=citation_mappings,
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            error=data.get("error"),
            batch_id=data.get("batch_id")
        )