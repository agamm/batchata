"""Google Gemini provider implementation."""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import google.genai as genai_lib
import google.generativeai as genai

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ...utils import get_logger
from ..provider import Provider
from .models import GEMINI_MODELS
from .message_prepare import prepare_messages
from .parse_results import parse_results


logger = get_logger(__name__)


class GeminiProvider(Provider):
    """Google Gemini provider with native batch processing support."""
    
    # Batch limitations based on Google's documentation
    MAX_REQUESTS = 10000  # Typical batch size limit
    REQUESTS_PER_MINUTE = 1000  # Rate limiting for individual requests
    
    def __init__(self, auto_register: bool = True):
        """Initialize Gemini provider."""
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Please set it with your Google API key."
            )
        
        # Configure both clients
        genai.configure(api_key=api_key)  # For legacy compatibility
        self.client = genai_lib.Client(api_key=api_key)  # For batch processing
        
        super().__init__()
        self.models = GEMINI_MODELS
        self._submitted_batches: Dict[str, Dict] = {}  # Track submitted batch jobs
    
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format."""
        # Check if model is supported
        if not self.supports_model(job.model):
            raise ValidationError(f"Unsupported model: {job.model}")
        
        # Check file capabilities
        model_config = self.get_model_config(job.model)
        if job.file and not model_config.supports_files:
            raise ValidationError(f"Model '{job.model}' does not support file input")
        
        # Validate structured output
        if job.response_model and not model_config.supports_structured_output:
            raise ValidationError(f"Model '{job.model}' does not support structured output")
        
        # Validate messages
        if job.messages:
            try:
                contents, generation_config = prepare_messages(job)
                # Basic validation - check that we have content
                if not contents:
                    raise ValidationError("No valid content in messages")
            except Exception as e:
                raise ValidationError(f"Invalid message format: {e}")
    
    def create_batch(self, jobs: List[Job], raw_files_dir: Optional[str] = None) -> tuple[str, Dict[str, Job]]:
        """Create and submit a batch of jobs using Google's native batch API."""
        if not jobs:
            raise BatchSubmissionError("Cannot create empty batch")
        
        if len(jobs) > self.MAX_REQUESTS:
            raise BatchSubmissionError(f"Too many jobs: {len(jobs)} > {self.MAX_REQUESTS}")
        
        # Validate all jobs
        for job in jobs:
            self.validate_job(job)
        
        # Convert jobs to InlinedRequest format
        inlined_requests = []
        job_mapping = {job.id: job for job in jobs}
        
        for job in jobs:
            try:
                # Prepare messages and configuration for this job
                contents, generation_config = prepare_messages(job)
                
                # Create configuration for the batch request
                config_dict = {
                    "temperature": generation_config.temperature,
                    "max_output_tokens": generation_config.max_output_tokens,
                }
                
                # Add response format for structured output
                if hasattr(generation_config, 'response_mime_type') and generation_config.response_mime_type:
                    config_dict["response_mime_type"] = generation_config.response_mime_type
                if hasattr(generation_config, 'response_schema') and generation_config.response_schema:
                    config_dict["response_schema"] = generation_config.response_schema
                
                # Create InlinedRequest
                request = genai_lib.types.InlinedRequest(
                    model=job.model,
                    contents=contents,
                    config=genai_lib.types.GenerateContentConfig(**config_dict)
                )
                inlined_requests.append(request)
                
            except Exception as e:
                raise BatchSubmissionError(f"Failed to prepare job {job.id}: {e}")
        
        # Submit batch job to Google
        try:
            logger.info(f"Creating batch job with {len(inlined_requests)} requests")
            batch_job = self.client.batches.create(
                model=jobs[0].model,  # Use first job's model as the primary model
                src=inlined_requests
            )
            
            batch_id = batch_job.name
            logger.info(f"✓ Created Gemini batch: {batch_id}")
            
            # Store batch info for tracking
            self._submitted_batches[batch_id] = {
                "batch_job": batch_job,
                "job_mapping": job_mapping,
                "raw_files_dir": raw_files_dir,
                "submitted_at": datetime.now()
            }
            
            # Save raw requests for debugging if directory provided
            if raw_files_dir:
                batch_requests = []
                for i, job in enumerate(jobs):
                    contents, generation_config = prepare_messages(job)
                    request = {
                        "job_id": job.id,
                        "model": job.model,
                        "contents": contents,
                        "generation_config": generation_config.__dict__
                    }
                    batch_requests.append(request)
                self._save_raw_requests(batch_id, batch_requests, raw_files_dir, "gemini")
            
            return batch_id, job_mapping
            
        except Exception as e:
            logger.error(f"✗ Failed to create batch: {e}")
            raise BatchSubmissionError(f"Failed to create batch: {e}")
    
    def get_batch_status(self, batch_id: str) -> tuple[str, Optional[Dict]]:
        """Get current status of a Google batch job."""
        if batch_id not in self._submitted_batches:
            return "failed", {"batch_id": batch_id, "error": "Batch not found"}
        
        try:
            # Get the latest batch job status from Google
            batch_job = self.client.batches.get(batch_id)
            
            # Update our stored batch job
            self._submitted_batches[batch_id]["batch_job"] = batch_job
            
            # Map Google's job states to our standard states
            state = batch_job.state
            if state == genai_lib.types.JobState.JOB_STATE_SUCCEEDED:
                return "complete", None
            elif state == genai_lib.types.JobState.JOB_STATE_FAILED:
                error_msg = batch_job.error.message if batch_job.error else "Unknown error"
                return "failed", {"batch_id": batch_id, "error": error_msg}
            elif state == genai_lib.types.JobState.JOB_STATE_CANCELLED:
                return "cancelled", {"batch_id": batch_id, "error": "Batch was cancelled"}
            elif state in [
                genai_lib.types.JobState.JOB_STATE_PENDING,
                genai_lib.types.JobState.JOB_STATE_QUEUED,
                genai_lib.types.JobState.JOB_STATE_RUNNING
            ]:
                return "running", None
            else:
                # Handle other states like PARTIALLY_SUCCEEDED, PAUSED, etc.
                return "running", None
                
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            return "failed", {"batch_id": batch_id, "error": str(e)}
    
    def get_batch_results(self, batch_id: str, job_mapping: Dict[str, Job], raw_files_dir: Optional[str] = None) -> List[JobResult]:
        """Retrieve results for a completed Google batch job."""
        if batch_id not in self._submitted_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch_info = self._submitted_batches[batch_id]
        batch_job = batch_info["batch_job"]
        
        # Ensure batch is complete
        if batch_job.state != genai_lib.types.JobState.JOB_STATE_SUCCEEDED:
            raise ValueError(f"Batch {batch_id} is not complete (state: {batch_job.state})")
        
        try:
            # Retrieve batch results from Google
            logger.info(f"Retrieving results for batch {batch_id}")
            
            # Get the output destination and read results
            # Note: The exact method to get results may vary based on destination type
            # For now, we'll use a simpler approach by iterating batch outputs
            results = []
            
            try:
                # Try to iterate over outputs (this API may vary)
                if hasattr(batch_job, 'iter_outputs'):
                    for output in batch_job.iter_outputs():
                        results.append({
                            "response": output,
                            "error": None
                        })
                else:
                    # Fallback: create results based on the number of jobs
                    # This is a temporary solution until we understand the exact output format
                    for job_id in job_mapping.keys():
                        results.append({
                            "job_id": job_id,
                            "response": None,  # Will be filled by parsing logic
                            "error": None
                        })
                        
            except Exception as e:
                logger.warning(f"Could not iterate batch outputs: {e}, using fallback")
                # Create placeholder results
                for job_id in job_mapping.keys():
                    results.append({
                        "job_id": job_id,
                        "response": None,
                        "error": f"Could not retrieve result: {e}"
                    })
            
            # Save raw responses for debugging if directory provided
            if raw_files_dir:
                self._save_raw_responses(batch_id, results, raw_files_dir, "gemini")
            
            # Parse results with batch discount (need to research actual discount)
            job_results = parse_results(
                results=results,
                job_mapping=job_mapping,
                raw_files_dir=raw_files_dir,
                batch_discount=0.5,  # Assuming 50% discount like other providers
                batch_id=batch_id
            )
            
            # Clean up completed batch from memory
            del self._submitted_batches[batch_id]
            
            return job_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve batch results for {batch_id}: {e}")
            raise ValueError(f"Failed to retrieve batch results: {e}")
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a Google batch request."""
        if batch_id not in self._submitted_batches:
            logger.warning(f"Batch {batch_id} not found for cancellation")
            return False
        
        try:
            # Cancel the batch job with Google
            self.client.batches.cancel(batch_id)
            
            # Update our tracking
            batch_info = self._submitted_batches[batch_id]
            batch_job = batch_info["batch_job"]
            batch_job.state = genai_lib.types.JobState.JOB_STATE_CANCELLED
            
            logger.info(f"✓ Cancelled Gemini batch: {batch_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to cancel Gemini batch {batch_id}: {e}")
            return False
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs."""
        total_cost = 0.0
        
        for job in jobs:
            try:
                # Estimate token count (rough approximation)
                input_text = ""
                if job.prompt:
                    input_text += job.prompt
                if job.file:
                    # Rough estimate based on file size
                    file_size = job.file.stat().st_size
                    input_text += " " * min(file_size // 4, 100000)  # Rough char estimate
                
                estimated_input_tokens = len(input_text) // 4  # Rough token estimate
                estimated_output_tokens = job.max_tokens or 1000
                
                # Use tokencost if available
                try:
                    import tokencost
                    job_cost = tokencost.calculate_cost(
                        model_name=job.model,
                        prompt_tokens=estimated_input_tokens,
                        completion_tokens=estimated_output_tokens
                    )
                    total_cost += job_cost
                except (ImportError, Exception):
                    # Fallback rough estimation
                    # Approximate costs per token (these are rough estimates)
                    if "1.5-pro" in job.model:
                        input_cost = estimated_input_tokens * 0.00125 / 1000  # $1.25 per 1M tokens
                        output_cost = estimated_output_tokens * 0.005 / 1000  # $5 per 1M tokens
                    else:  # flash models
                        input_cost = estimated_input_tokens * 0.000075 / 1000  # $0.075 per 1M tokens
                        output_cost = estimated_output_tokens * 0.0003 / 1000  # $0.30 per 1M tokens
                    
                    total_cost += input_cost + output_cost
                    
            except Exception as e:
                logger.warning(f"Could not estimate cost for job {job.id}: {e}")
        
        return total_cost
    
    def get_polling_interval(self) -> float:
        """Get the polling interval for batch status checks."""
        return 2.0  # Check every 2 seconds for async processing