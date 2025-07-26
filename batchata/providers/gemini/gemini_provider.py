"""Simple Google Gemini provider with batch processing."""

import os
import warnings
from typing import Dict, List, Optional

import google.genai as genai_lib

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ..provider import Provider
from .models import GEMINI_MODELS
from .message_prepare import prepare_messages
from .parse_results import parse_results

# Suppress known Google SDK warning about BATCH_STATE_RUNNING
# The API returns BATCH_STATE_* but SDK expects JOB_STATE_*
warnings.filterwarnings('ignore', message='.*is not a valid JobState')


class GeminiProvider(Provider):
    """Google Gemini provider with batch processing support."""
    
    MAX_REQUESTS = 10000
    
    def __init__(self, auto_register: bool = True):
        """Initialize with Google API key."""
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.client = genai_lib.Client(api_key=api_key)
        super().__init__()
        self.models = GEMINI_MODELS
        self._batches: Dict[str, Dict] = {}
    
    def validate_job(self, job: Job) -> None:
        """Validate job configuration."""
        if not job:
            raise ValidationError("Job cannot be None")
        
        if not self.supports_model(job.model):
            raise ValidationError(f"Unsupported model: {job.model}")
        
        model_config = self.get_model_config(job.model)
        
        if job.file and not model_config.supports_files:
            raise ValidationError(f"Model '{job.model}' does not support file input")
        
        if job.response_model and not model_config.supports_structured_output:
            raise ValidationError(f"Model '{job.model}' does not support structured output")
        
        if job.messages:
            contents, _ = prepare_messages(job)
            if not contents:
                raise ValidationError("No valid content in messages")
    
    def create_batch(self, jobs: List[Job], raw_files_dir: Optional[str] = None) -> tuple[str, Dict[str, Job]]:
        """Create batch using Google's inline requests API."""
        if not jobs:
            raise BatchSubmissionError("Cannot create empty batch")
        
        if len(jobs) > self.MAX_REQUESTS:
            raise BatchSubmissionError(f"Too many jobs: {len(jobs)} > {self.MAX_REQUESTS}")
        
        # Validate jobs
        for job in jobs:
            self.validate_job(job)
        
        # Create inline requests
        inlined_requests = []
        job_mapping = {job.id: job for job in jobs}
        
        for job in jobs:
            contents, generation_config = prepare_messages(job)
            
            config = generation_config or {}
            if job.temperature is not None:
                config["temperature"] = job.temperature
            if job.max_tokens is not None:
                config["max_output_tokens"] = job.max_tokens
            
            request = genai_lib.types.InlinedRequest(
                model=job.model,
                contents=contents,
                config=genai_lib.types.GenerateContentConfig(**config) if config else None
            )
            inlined_requests.append(request)
        
        # Submit batch
        batch_job = self.client.batches.create(
            model=jobs[0].model,
            src=inlined_requests
        )
        
        batch_id = batch_job.name
        self._batches[batch_id] = {
            "batch_job": batch_job,
            "job_mapping": job_mapping,
            "raw_files_dir": raw_files_dir
        }
        
        # Save raw requests for debugging
        if raw_files_dir:
            batch_requests = []
            for job in jobs:
                contents, generation_config = prepare_messages(job)
                batch_requests.append({
                    "job_id": job.id,
                    "model": job.model,
                    "contents": contents,
                    "generation_config": generation_config
                })
            self._save_raw_requests(batch_id, batch_requests, raw_files_dir, "gemini")
        
        return batch_id, job_mapping
    
    def get_batch_status(self, batch_id: str) -> tuple[str, Optional[Dict]]:
        """Get batch status from Google."""
        if batch_id not in self._batches:
            return "failed", {"batch_id": batch_id, "error": "Batch not found"}
        
        try:
            batch_job = self.client.batches.get(name=batch_id)
            self._batches[batch_id]["batch_job"] = batch_job
            
            # Handle state by name to avoid enum validation issues
            state_name = getattr(batch_job.state, 'name', str(batch_job.state))
            
            if state_name == 'JOB_STATE_SUCCEEDED':
                return "complete", None
            elif state_name == 'JOB_STATE_FAILED':
                error_msg = batch_job.error.message if batch_job.error else "Unknown error"
                return "failed", {"batch_id": batch_id, "error": error_msg}
            elif state_name == 'JOB_STATE_CANCELLED':
                return "cancelled", {"batch_id": batch_id, "error": "Batch was cancelled"}
            else:
                # Handle running states (including BATCH_STATE_RUNNING)
                return "running", None
                
        except Exception as e:
            return "failed", {"batch_id": batch_id, "error": str(e)}
    
    def get_batch_results(self, batch_id: str, job_mapping: Dict[str, Job], raw_files_dir: Optional[str] = None) -> List[JobResult]:
        """Retrieve batch results from Google."""
        if batch_id not in self._batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        if not job_mapping:
            return []
        
        batch_info = self._batches[batch_id]
        batch_job = batch_info["batch_job"]
        
        # Check if batch is complete using state name
        state_name = getattr(batch_job.state, 'name', str(batch_job.state))
        if state_name != 'JOB_STATE_SUCCEEDED':
            raise ValueError(f"Batch {batch_id} is not complete (state: {state_name})")
        
        # Get results from inline responses (following official docs)
        results = []
        job_ids = list(job_mapping.keys())
        
        if not job_ids:
            return []
        
        try:
            if batch_job.dest and batch_job.dest.inlined_responses:
                inline_responses = batch_job.dest.inlined_responses
                for idx, inline_response in enumerate(inline_responses):
                    if idx >= len(job_ids):
                        break
                    
                    if inline_response.response:
                        results.append({
                            "job_id": job_ids[idx],
                            "response": inline_response.response,
                            "error": None
                        })
                    elif inline_response.error:
                        results.append({
                            "job_id": job_ids[idx],
                            "response": None,
                            "error": str(inline_response.error)
                        })
            else:
                # No inline responses found
                for job_id in job_ids:
                    results.append({
                        "job_id": job_id,
                        "response": None,
                        "error": "No inline responses found in batch result"
                    })
        except Exception as e:
            for job_id in job_ids:
                results.append({
                    "job_id": job_id,
                    "response": None,
                    "error": f"Could not retrieve result: {e}"
                })
        
        # Save raw responses for debugging
        if raw_files_dir:
            self._save_raw_responses(batch_id, results, raw_files_dir, "gemini")
        
        # Parse results
        # Get batch discount from model config (all jobs in batch use same model)
        first_job = next(iter(job_mapping.values()))
        model_config = self.get_model_config(first_job.model)
        
        job_results = parse_results(
            results=results,
            job_mapping=job_mapping,
            raw_files_dir=raw_files_dir,
            batch_discount=model_config.batch_discount,
            batch_id=batch_id
        )
        
        # Clean up
        del self._batches[batch_id]
        
        return job_results
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel batch job."""
        if batch_id not in self._batches:
            return False
        
        try:
            self.client.batches.cancel(name=batch_id)
            return True
        except Exception:
            return False
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for jobs using Google's token counting API."""
        total_cost = 0.0
        
        for job in jobs:
            model_config = self.get_model_config(job.model)
            batch_discount = model_config.batch_discount
            
            # Get accurate token count using Google's API
            input_tokens = self._count_tokens(job)
            output_tokens = job.max_tokens or 1000
            
            try:
                import tokencost
                cost = tokencost.calculate_cost_by_tokens(
                    model=job.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                total_cost += cost * (1 - batch_discount)
            except Exception:
                # Fallback with rough pricing estimates
                if "pro" in job.model.lower():
                    cost = (input_tokens * 0.00125 + output_tokens * 0.005) / 1000
                else:
                    cost = (input_tokens * 0.000075 + output_tokens * 0.0003) / 1000
                total_cost += cost * (1 - batch_discount)
        
        return total_cost
    
    def _count_tokens(self, job: Job) -> int:
        """Count tokens using Google's official API."""
        try:
            # Prepare content for token counting
            contents, _ = prepare_messages(job)
            
            # Use Google's token counting API
            response = self.client.models.count_tokens(
                model=job.model,
                contents=contents
            )
            return getattr(response, 'total_tokens', 0)
            
        except Exception:
            # Fallback to rough estimation
            input_text = job.prompt or ""
            if job.file:
                file_size = job.file.stat().st_size
                input_text += " " * min(file_size // 4, 100000)
            return len(input_text) // 4
    
    def get_polling_interval(self) -> float:
        """Polling interval for status checks."""
        return 2.0