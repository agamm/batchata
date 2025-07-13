"""BatchRequest class for tracking batch submissions."""

from dataclasses import dataclass
from datetime import datetime
from typing import List

from ..core.job_result import JobResult


@dataclass
class BatchRequest:
    """Represents a submitted batch request to a provider.
    
    Tracks the lifecycle of a batch from submission to completion,
    providing status updates and result retrieval.
    
    Attributes:
        id: Unique identifier for this batch
        provider_batch_id: Provider's internal batch ID
        submitted_at: When the batch was submitted
        job_ids: List of job IDs in this batch
        status: Current status ("pending", "running", "complete", "failed")
    """
    
    id: str  # Our internal batch ID
    provider_batch_id: str  # Provider's batch ID
    submitted_at: datetime
    job_ids: List[str]
    status: str = "pending"
    
    @property
    def is_complete(self) -> bool:
        """Whether the batch has completed (successfully or failed)."""
        return self.status in ["complete", "failed"]
    
    @property
    def is_failed(self) -> bool:
        """Whether the batch has failed."""
        return self.status == "failed"
    
    def get_status(self) -> str:
        """Get current batch status.
        
        This method should be overridden by provider implementations
        to actually check status with the provider's API.
        
        Returns:
            Status string
        """
        return self.status
    
    def get_results(self) -> List[JobResult]:
        """Get batch results.
        
        This method should be overridden by provider implementations
        to actually retrieve results from the provider's API.
        
        Returns:
            List of JobResult objects
        """
        return []