"""Anthropic provider implementation."""

import os
from datetime import datetime
from typing import Dict, List, Optional

from anthropic import Anthropic

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ...utils import get_logger
from ..provider import Provider
from .models import ANTHROPIC_MODELS
from .message_prepare import prepare_messages
from .parse_results import parse_results


logger = get_logger(__name__)


class AnthropicProvider(Provider):
    """Anthropic provider for batch processing."""
    
    # Batch limitations
    MAX_REQUESTS = 100_000
    MAX_TOTAL_SIZE_MB = 256
    
    def __init__(self, auto_register: bool = True):
        """Initialize Anthropic provider."""
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Please set it with your Anthropic API key."
            )
        
        self.client = Anthropic()
        super().__init__()
        self.models = ANTHROPIC_MODELS
    
    
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format."""
        # Check if model is supported
        if not self.supports_model(job.model):
            raise ValidationError(f"Unsupported model: {job.model}")
        
        # Check file capabilities
        model_config = self.get_model_config(job.model)
        if job.file and not model_config.supports_files:
            raise ValidationError(f"Model '{job.model}' does not support file input")
        
        # Validate PDF textual compatibility with citations (Anthropic-specific)
        if job.file and job.enable_citations and job.file.suffix.lower() == '.pdf':
            from ...utils.pdf import is_textual_pdf
            
            textual_score = is_textual_pdf(job.file)
            
            if textual_score == 0.0:
                raise ValidationError(
                    f"PDF '{job.file}' appears to be image-only (no extractable text). "
                    "Citations will not work with scanned/image PDFs. "
                    "Please use a text-based PDF or disable citations."
                )
            elif textual_score < 0.1:
                logger.warning(
                    f"PDF '{job.file}' has very low text content (score: {textual_score:.2f}). "
                    "Citations may not work well with primarily image-based PDFs."
                )
        
        # Validate messages using pydantic
        if job.messages:
            try:
                messages, _ = prepare_messages(job)
                
                # Anthropic-specific validation: no consecutive messages from same role
                if len(messages) > 1:
                    for i in range(1, len(messages)):
                        if messages[i].get("role") == messages[i-1].get("role"):
                            raise ValidationError("Anthropic does not allow consecutive messages from same role")
                            
            except Exception as e:
                raise ValidationError(f"Invalid message format: {e}")
    
    def create_batch(self, jobs: List[Job]) -> tuple[str, Dict[str, Job]]:
        """Create and submit a batch of jobs."""
        if not jobs:
            raise BatchSubmissionError("Cannot create empty batch")
        
        if len(jobs) > self.MAX_REQUESTS:
            raise BatchSubmissionError(f"Too many jobs: {len(jobs)} > {self.MAX_REQUESTS}")
        
        # Validate all jobs
        for job in jobs:
            self.validate_job(job)
        
        # Prepare batch requests and create batch-specific job mapping
        batch_requests = []
        job_mapping = {}
        
        for job in jobs:
            messages, system_prompt = prepare_messages(job)
            
            request = {
                "custom_id": job.id,
                "params": {
                    "model": job.model,
                    "messages": messages,
                    "max_tokens": job.max_tokens,
                    "temperature": job.temperature
                }
            }
            
            if system_prompt:
                request["params"]["system"] = system_prompt
            
            batch_requests.append(request)
            job_mapping[job.id] = job
        
        # Submit to Anthropic
        try:
            logger.info(f"Submitting batch with {len(batch_requests)} requests to Anthropic API")
            batch_response = self.client.messages.batches.create(requests=batch_requests)
            provider_batch_id = batch_response.id
            logger.info(f"✓ Anthropic batch created successfully: {provider_batch_id}")
        except Exception as e:
            logger.error(f"✗ Failed to create Anthropic batch: {e}")
            raise BatchSubmissionError(f"Failed to create batch: {e}")
        
        return provider_batch_id, job_mapping
    
    def get_batch_status(self, batch_id: str) -> tuple[str, Optional[Dict]]:
        """Get current status of a batch."""
        try:
            batch_status = self.client.messages.batches.retrieve(batch_id)
            
            # Check if this is an error response
            if hasattr(batch_status, 'type') and batch_status.type == 'error':
                error_details = {
                    "batch_id": batch_id,
                    "error": batch_status.error.message,
                    "error_type": batch_status.error.type
                }
                logger.error(f"Batch {batch_id} retrieval failed: {batch_status.error.message}")
                return "failed", error_details
            
            status = batch_status.processing_status
            
            # Map Anthropic statuses to our standard statuses
            if status == "ended":
                # Check if there were any errors
                if hasattr(batch_status, 'request_counts') and batch_status.request_counts.errored > 0:
                    # Calculate total from succeeded + errored
                    total_count = getattr(batch_status.request_counts, 'succeeded', 0) + batch_status.request_counts.errored
                    error_details = {
                        "batch_id": batch_id,
                        "errored_count": batch_status.request_counts.errored,
                        "succeeded_count": getattr(batch_status.request_counts, 'succeeded', 0),
                        "total_count": total_count
                    }
                    logger.error(f"Batch {batch_id} completed with {batch_status.request_counts.errored} errors out of {total_count} total")
                    return "failed", error_details
                return "complete", None
            elif status in ["canceled", "expired"]:
                return "failed", {"batch_id": batch_id, "reason": f"Batch {status}"}
            elif status == "in_progress":
                return "running", None
            else:
                return "pending", None
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            return "failed", {"batch_id": batch_id, "error": str(e)}
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch request.
        
        Args:
            batch_id: The batch ID to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            logger.info(f"Cancelling Anthropic batch: {batch_id}")
            self.client.messages.batches.cancel(batch_id)
            logger.info(f"✓ Successfully cancelled Anthropic batch: {batch_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to cancel Anthropic batch {batch_id}: {e}")
            return False
    
    def get_batch_results(self, batch_id: str, job_mapping: Dict[str, Job], raw_responses_dir: Optional[str] = None) -> List[JobResult]:
        """Retrieve results for a completed batch."""
        try:
            results = list(self.client.messages.batches.results(batch_id))
            return parse_results(results, job_mapping, raw_responses_dir)
        except Exception as e:
            raise ValidationError(f"Failed to get batch results: {e}")
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs using tokencost.
        
        WARNING: This is an estimation. Actual costs may vary due to:
        - Token counting differences between the estimator and Anthropic's tokenizer
        - Dynamic pricing changes
        - Additional fees or discounts not captured here
        
        Uses conservative token estimation for Claude models.
        """
        try:
            from tokencost import calculate_cost_by_tokens
            from ...utils.llm import token_count_simple
        except ImportError:
            logger.warning("tokencost not available, returning 0 cost estimate")
            return 0.0
        
        total_cost = 0.0
        
        for job in jobs:
            try:
                # Prepare messages to get actual input
                messages, system_prompt = prepare_messages(job)
                
                # Build full text for token estimation
                full_text = ""
                if system_prompt:
                    full_text += system_prompt + "\n\n"
                
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    full_text += f"{role}: {content}\n\n"
                
                # Estimate tokens using Claude-specific estimator
                input_tokens = token_count_simple(full_text)
                
                # Calculate costs using tokencost with actual Claude model
                input_cost = float(calculate_cost_by_tokens(
                    input_tokens,
                    job.model,
                    token_type="input"
                ))
                
                output_cost = float(calculate_cost_by_tokens(
                    job.max_tokens,
                    job.model,
                    token_type="output"
                ))
                
                # Apply batch discount
                model_config = self.get_model_config(job.model)
                discount = model_config.batch_discount if model_config else 0.5
                job_cost = (input_cost + output_cost) * discount
                
                logger.info(
                    f"Job {job.id}: ~{input_tokens} input tokens, "
                    f"{job.max_tokens} max output tokens, "
                    f"cost: ${job_cost:.6f} (with {int(discount*100)}% batch discount)"
                )
                
                total_cost += job_cost
                
            except Exception as e:
                logger.warning(f"Failed to estimate cost for job {job.id}: {e}")
                continue
        
        return total_cost