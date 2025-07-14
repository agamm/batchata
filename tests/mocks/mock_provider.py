"""Mock provider for testing without API calls."""

from typing import Dict, List, Optional, Any
import time
import uuid

from batchata.providers import Provider
from batchata.providers.model_config import ModelConfig
from batchata.core.job import Job
from batchata.core.job_result import JobResult
from batchata.exceptions import ValidationError, ProviderError


class MockProvider(Provider):
    """Mock provider that simulates API behavior without real calls."""
    
    def __init__(self, delay: float = 0.1, auto_register: bool = True):
        """Initialize mock provider.
        
        Args:
            delay: Simulated processing delay in seconds
            auto_register: Whether to register in provider registry
        """
        super().__init__()
        self.delay = delay
        self.batches: Dict[str, Dict[str, Any]] = {}
        self.call_history: List[Dict[str, Any]] = []
        
        # Define mock models
        self.models = {
            "mock-model-basic": ModelConfig(
                name="mock-model-basic",
                max_input_tokens=100000,
                max_output_tokens=4096,
                batch_discount=0.5,
                supports_citations=False,
                supports_images=False,
                supports_files=False
            ),
            "mock-model-advanced": ModelConfig(
                name="mock-model-advanced",
                max_input_tokens=200000,
                max_output_tokens=8192,
                batch_discount=0.5,
                supports_citations=True,
                supports_images=True,
                supports_files=True,
                file_types=[".pdf", ".txt", ".docx"]
            )
        }
        
        # Configure responses
        self.responses: Dict[str, str] = {}
        self.should_fail = False
        self.failure_message = "Mock failure"
    
    def set_response(self, job_id: str, response: str) -> None:
        """Set a specific response for a job ID."""
        self.responses[job_id] = response
    
    def set_failure(self, should_fail: bool = True, message: str = "Mock failure") -> None:
        """Configure the provider to fail."""
        self.should_fail = should_fail
        self.failure_message = message
    
    def validate_job(self, job: Job) -> None:
        """Validate job constraints."""
        self.call_history.append({"method": "validate_job", "job_id": job.id})
        
        if self.should_fail:
            raise ValidationError(self.failure_message)
        
        # Get model from job attributes
        model = job.model
        if not model:
            raise ValidationError("Model parameter is required")
        
        if model not in self.models:
            raise ValidationError(f"Unknown model: {model}")
        
        # Validate message format
        if job.messages:
            for msg in job.messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValidationError("Invalid message format")
    
    def create_batch(self, jobs: List[Job]) -> str:
        """Create a mock batch."""
        self.call_history.append({"method": "create_batch", "job_count": len(jobs)})
        
        if self.should_fail:
            raise ProviderError(self.failure_message)
        
        batch_id = f"mock-batch-{uuid.uuid4().hex[:8]}"
        
        # Store batch info
        self.batches[batch_id] = {
            "jobs": jobs,
            "status": "pending",
            "created_at": time.time(),
            "results": []
        }
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> str:
        """Get mock batch status."""
        self.call_history.append({"method": "get_batch_status", "batch_id": batch_id})
        
        if batch_id not in self.batches:
            raise ProviderError(f"Batch not found: {batch_id}")
        
        batch = self.batches[batch_id]
        
        # Simulate processing delay
        elapsed = time.time() - batch["created_at"]
        if elapsed < self.delay:
            return "running"
        
        # Mark as complete and generate results if needed
        if batch["status"] == "pending":
            batch["status"] = "complete"
            self._generate_results(batch_id)
        
        return batch["status"]
    
    def get_batch_results(self, batch_id: str, raw_responses_dir: Optional[str] = None) -> List[JobResult]:
        """Get mock batch results."""
        self.call_history.append({"method": "get_batch_results", "batch_id": batch_id})
        
        if batch_id not in self.batches:
            raise ProviderError(f"Batch not found: {batch_id}")
        
        batch = self.batches[batch_id]
        
        # Ensure batch is complete
        if self.get_batch_status(batch_id) != "complete":
            raise ProviderError("Batch not complete")
        
        return batch["results"]
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a mock batch."""
        self.call_history.append({"method": "cancel_batch", "batch_id": batch_id})
        
        if batch_id not in self.batches:
            return False
        
        batch = self.batches[batch_id]
        if batch["status"] in ["complete", "failed"]:
            return False
        
        batch["status"] = "cancelled"
        return True
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for jobs."""
        self.call_history.append({"method": "estimate_cost", "job_count": len(jobs)})
        
        total_cost = 0.0
        for job in jobs:
            model = job.model
            config = self.models.get(model)
            if not config:
                continue
            
            # Simple estimation based on message length
            input_tokens = sum(len(msg.get("content", "").split()) * 1.5 
                             for msg in job.messages or [])
            output_tokens = job.max_tokens * 0.5
            
            # Mock cost calculation (since ModelConfig doesn't have cost fields)
            # Using arbitrary costs for testing
            input_cost_per_1k = 0.003 if model == "mock-model-basic" else 0.01
            output_cost_per_1k = 0.015 if model == "mock-model-basic" else 0.03
            
            cost = (input_tokens / 1000 * input_cost_per_1k +
                   output_tokens / 1000 * output_cost_per_1k)
            
            # Apply batch discount
            total_cost += cost * config.batch_discount
        
        return total_cost
    
    def _generate_results(self, batch_id: str) -> None:
        """Generate mock results for a batch."""
        batch = self.batches[batch_id]
        jobs = batch["jobs"]
        
        results = []
        for job in jobs:
            # Use configured response or generate default
            content = self.responses.get(job.id, f"Mock response for job {job.id}")
            
            # Calculate mock token counts
            input_tokens = sum(len(msg.get("content", "").split()) * 1.5 
                             for msg in job.messages or [])
            output_tokens = len(content.split()) * 1.5
            
            # Calculate cost
            model = job.model
            config = self.models[model]
            
            # Mock cost calculation
            input_cost_per_1k = 0.003 if model == "mock-model-basic" else 0.01
            output_cost_per_1k = 0.015 if model == "mock-model-basic" else 0.03
            
            cost = (input_tokens / 1000 * input_cost_per_1k +
                   output_tokens / 1000 * output_cost_per_1k) * config.batch_discount
            
            result = JobResult(
                job_id=job.id,
                response=content,
                cost_usd=cost,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                error=None
            )
            
            # Add citations if supported and requested
            if config.supports_citations and job.enable_citations:
                from batchata.types import Citation
                result.citations = [Citation(text="Mock citation", source="test", page=1)]
            
            results.append(result)
        
        batch["results"] = results
    
    def reset(self) -> None:
        """Reset the mock provider state."""
        self.batches.clear()
        self.call_history.clear()
        self.responses.clear()
        self.should_fail = False