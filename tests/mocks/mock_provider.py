"""Mock provider for testing."""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set

from batchata.core.job import Job
from batchata.core.job_result import JobResult
from batchata.exceptions import BatchSubmissionError, ValidationError
from batchata.providers.provider_registry import ProviderRegistry
from batchata.providers.batch_request import BatchRequest
from batchata.providers.model_config import ModelConfig
from batchata.providers.provider import Provider


class MockProvider(Provider):
    """Mock provider for testing.
    
    Allows configuration of responses, delays, and failures for testing
    various scenarios without making real API calls.
    """
    
    def __init__(self, auto_register: bool = True):
        """Initialize mock provider.
        
        Args:
            auto_register: Whether to automatically register with ProviderRegistry
        """
        super().__init__()
        
        # Mock configuration
        self.mock_responses: Dict[str, JobResult] = {}
        self.mock_delays: Dict[str, float] = {}
        self.mock_failures: Dict[str, Exception] = {}
        self.batch_failures: Set[str] = set()
        
        # Track batches
        self.batches: Dict[str, Dict] = {}
        
        if auto_register:
            ProviderRegistry.register(self)
    
    def _setup_models(self):
        """Setup mock model configurations."""
        self.models = {
            "mock-model-basic": ModelConfig(
                name="mock-model-basic",
                max_input_tokens=100000,
                max_output_tokens=4096,
                batch_discount=0.5,
                supports_structured_output=True
            ),
            "mock-model-advanced": ModelConfig(
                name="mock-model-advanced",
                max_input_tokens=200000,
                max_output_tokens=8192,
                batch_discount=0.5,
                supports_images=True,
                supports_files=True,
                supports_citations=True,
                supports_structured_output=True,
                file_types=[".pdf", ".txt", ".docx"]
            ),
            "mock-model-simple": ModelConfig(
                name="mock-model-simple",
                max_input_tokens=50000,
                max_output_tokens=2048,
                batch_discount=0.3,
                supports_structured_output=False
            )
        }
    
    def set_mock_response(self, job_id: str, result: JobResult):
        """Configure mock response for a job.
        
        Args:
            job_id: Job ID to mock
            result: Result to return for this job
        """
        self.mock_responses[job_id] = result
    
    def set_mock_delay(self, job_id: str, delay_seconds: float):
        """Configure delay for a job.
        
        Args:
            job_id: Job ID to delay
            delay_seconds: Seconds to delay before returning result
        """
        self.mock_delays[job_id] = delay_seconds
    
    def set_mock_failure(self, job_id: str, exception: Exception):
        """Configure failure for a job.
        
        Args:
            job_id: Job ID to fail
            exception: Exception to raise
        """
        self.mock_failures[job_id] = exception
    
    def set_batch_failure(self, batch_id: str):
        """Configure a batch to fail.
        
        Args:
            batch_id: Batch ID that should fail
        """
        self.batch_failures.add(batch_id)
    
    def validate_job(self, job: Job) -> None:
        """Validate job against mock constraints.
        
        Args:
            job: Job to validate
            
        Raises:
            ValidationError: If job is invalid
        """
        if job.model not in self.models:
            raise ValidationError(f"Model {job.model} not supported by MockProvider")
        
        model = self.models[job.model]
        
        # Check file support
        if job.file and not model.supports_files:
            raise ValidationError(f"Model {job.model} does not support file inputs")
        
        # Check structured output support
        if job.response_model and not model.supports_structured_output:
            raise ValidationError(f"Model {job.model} does not support structured output")
        
        # Check citation support
        if job.enable_citations and not model.supports_citations:
            raise ValidationError(f"Model {job.model} does not support citations")
    
    def create_batch(self, jobs: List[Job]) -> BatchRequest:
        """Create a mock batch.
        
        Args:
            jobs: Jobs to include in batch
            
        Returns:
            BatchRequest with mock batch ID
            
        Raises:
            BatchSubmissionError: If configured to fail
        """
        # Validate all jobs
        for job in jobs:
            self.validate_job(job)
        
        # Generate batch ID
        batch_id = f"mock-batch-{uuid.uuid4().hex[:8]}"
        
        # Check if batch should fail
        if batch_id in self.batch_failures:
            raise BatchSubmissionError(f"Mock batch submission failed: {batch_id}")
        
        # Store batch info
        self.batches[batch_id] = {
            "jobs": jobs,
            "status": "pending",
            "created_at": datetime.now(),
            "completed_at": None
        }
        
        # Create batch request
        batch_request = BatchRequest(
            id=batch_id,
            provider=self,
            jobs=jobs,
            status="pending"
        )
        
        return batch_request
    
    def get_batch_status(self, batch_id: str) -> str:
        """Get mock batch status.
        
        Args:
            batch_id: Batch to check
            
        Returns:
            Mock status string
        """
        if batch_id not in self.batches:
            return "failed"
        
        batch_info = self.batches[batch_id]
        
        # Simulate processing time
        elapsed = (datetime.now() - batch_info["created_at"]).total_seconds()
        
        # Check for configured delays
        max_delay = 0.0
        for job in batch_info["jobs"]:
            if job.id in self.mock_delays:
                max_delay = max(max_delay, self.mock_delays[job.id])
        
        # Update status based on elapsed time
        if elapsed < max_delay:
            batch_info["status"] = "running"
        else:
            batch_info["status"] = "complete"
            batch_info["completed_at"] = datetime.now()
        
        return batch_info["status"]
    
    def get_batch_results(self, batch_id: str) -> List[JobResult]:
        """Get mock batch results.
        
        Args:
            batch_id: Batch to get results for
            
        Returns:
            List of mock JobResult objects
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch not found: {batch_id}")
        
        batch_info = self.batches[batch_id]
        results = []
        
        for job in batch_info["jobs"]:
            # Check for configured response
            if job.id in self.mock_responses:
                results.append(self.mock_responses[job.id])
            else:
                # Generate default response
                result = JobResult(
                    job_id=job.id,
                    response=f"Mock response for job {job.id}",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.005
                )
                
                # Add structured output if requested
                if job.response_model:
                    try:
                        # Create a simple instance
                        result.parsed_response = {"mocked": True, "job_id": job.id}
                    except Exception:
                        result.parsed_response = {"error": "Mock parsing failed"}
                
                results.append(result)
        
        return results
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for jobs.
        
        Args:
            jobs: Jobs to estimate
            
        Returns:
            Mock cost estimate
        """
        # Simple mock pricing: $0.01 per job
        return len(jobs) * 0.01