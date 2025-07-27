"""Simple Google Gemini provider with batch processing."""

import os
import time
import warnings
from decimal import Decimal
from typing import Dict, List, Optional

import google.genai as genai_lib

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ...utils import get_logger
from ..provider import Provider
from .models import GEMINI_MODELS
from .message_prepare import prepare_messages
from .parse_results import parse_results


logger = get_logger(__name__)

# Suppress known Google SDK warning about BATCH_STATE_RUNNING
# The API returns BATCH_STATE_* but SDK expects JOB_STATE_*
warnings.filterwarnings('ignore', message='.*is not a valid JobState')

# Token estimation constants
CHARS_PER_TOKEN = 4
MAX_FILE_ESTIMATION_TOKENS = 100000
# Token counting retry configuration
TOKEN_COUNT_MAX_RETRIES = 3
TOKEN_COUNT_RETRY_DELAY = 1.0  # seconds

# Google batch job states
JOB_STATE_SUCCEEDED = 'JOB_STATE_SUCCEEDED'
JOB_STATE_FAILED = 'JOB_STATE_FAILED'
JOB_STATE_CANCELLED = 'JOB_STATE_CANCELLED'


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
        
        if job.file and model_config.supports_files:
            # Validate file type
            file_ext = job.file.suffix.lower()
            if hasattr(model_config, 'file_types') and model_config.file_types:
                if file_ext not in model_config.file_types:
                    supported = ', '.join(model_config.file_types)
                    raise ValidationError(f"Unsupported file type '{file_ext}'. Supported: {supported}")
        
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
            
            if state_name == JOB_STATE_SUCCEEDED:
                return "complete", None
            elif state_name == JOB_STATE_FAILED:
                error_msg = batch_job.error.message if batch_job.error else "Unknown error"
                return "failed", {"batch_id": batch_id, "error": error_msg}
            elif state_name == JOB_STATE_CANCELLED:
                return "cancelled", {"batch_id": batch_id, "error": "Batch was cancelled"}
            else:
                # Handle running states (including BATCH_STATE_RUNNING)
                return "running", None
                
        except (ValueError, AttributeError, ConnectionError) as e:
            return "failed", {"batch_id": batch_id, "error": str(e)}
        except Exception as e:
            return "failed", {"batch_id": batch_id, "error": f"Unexpected error: {str(e)}"}
    
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
        except (AttributeError, KeyError, IndexError) as e:
            for job_id in job_ids:
                results.append({
                    "job_id": job_id,
                    "response": None,
                    "error": f"Could not retrieve result: {e}"
                })
        except Exception as e:
            for job_id in job_ids:
                results.append({
                    "job_id": job_id,
                    "response": None,
                    "error": f"Unexpected error retrieving result: {e}"
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
        except (ConnectionError, ValueError, AttributeError):
            return False
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs using tokencost and Google's token counting API.
        
        WARNING: This is an estimation. Actual costs may vary due to:
        - Token counting differences between the estimator and Google's tokenizer
        - Dynamic pricing changes
        - Additional fees or discounts not captured here
        
        Uses Google's official token counting API for accurate token estimates.
        """
        total_cost = Decimal('0.0')
        
        for job in jobs:
            model_config = self.get_model_config(job.model)
            batch_discount = Decimal(str(model_config.batch_discount))
            
            # Get accurate token count using Google's API
            input_tokens = self._count_tokens(job)
            output_tokens = job.max_tokens or 1000
            
            try:
                import tokencost
                
                # Calculate input and output costs separately
                input_cost = tokencost.calculate_cost_by_tokens(
                    num_tokens=input_tokens,
                    model=job.model,
                    token_type='input'
                )
                output_cost = tokencost.calculate_cost_by_tokens(
                    num_tokens=output_tokens,
                    model=job.model,
                    token_type='output'
                )
                
                cost = Decimal(str(input_cost + output_cost))
                total_cost += cost * (Decimal('1') - batch_discount)
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.warning("tokencost not available, returning 0 cost estimate")
                return 0.0
        
        return float(total_cost)
    
    def _count_tokens(self, job: Job) -> int:
        """Count tokens using Google's official API with retry mechanism."""
        # Prepare content for token counting
        contents, _ = prepare_messages(job)
        
        last_exception = None
        for attempt in range(TOKEN_COUNT_MAX_RETRIES):
            try:
                response = self.client.models.count_tokens(
                    model=job.model,
                    contents=contents
                )
                return getattr(response, 'total_tokens', 0)
                
            except (ConnectionError, ValueError, AttributeError) as e:
                last_exception = e
                if attempt < TOKEN_COUNT_MAX_RETRIES - 1:
                    time.sleep(TOKEN_COUNT_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                break
        
        # If all retries failed, raise a clear error instead of returning a meaningless fallback
        raise ValidationError(
            f"Failed to count tokens for model {job.model} after {TOKEN_COUNT_MAX_RETRIES} attempts. "
            f"Last error: {last_exception}. Please check your Google API key and connection."
        )
    
    def get_polling_interval(self) -> float:
        """Polling interval for status checks."""
        return 2.0