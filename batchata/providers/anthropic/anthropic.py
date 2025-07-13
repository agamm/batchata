"""Anthropic provider implementation."""

import os
from datetime import datetime
from typing import Dict, List

from anthropic import Anthropic

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ..batch_request import BatchRequest
from ..provider import Provider
from .models import ANTHROPIC_MODELS
from .message_prepare import prepare_messages
from .parse_results import parse_results


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
        self._job_mapping: Dict[str, Job] = {}  # Track jobs for parsing
        super().__init__()
        self.models = ANTHROPIC_MODELS
    
    
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format."""
        # Check if model is supported
        if not self.supports_model(job.model):
            raise ValidationError(f"Model '{job.model}' is not supported by Anthropic provider")
        
        # Check file capabilities
        model_config = self.get_model_config(job.model)
        if job.file and not model_config.supports_files:
            raise ValidationError(f"Model '{job.model}' does not support file input")
        
        # Validate messages using pydantic
        if job.messages:
            try:
                messages, _ = prepare_messages(job)
            except Exception as e:
                raise ValidationError(f"Invalid message format: {e}")
    
    def create_batch(self, jobs: List[Job]) -> BatchRequest:
        """Create and submit a batch of jobs."""
        if not jobs:
            raise BatchSubmissionError("Cannot create empty batch")
        
        if len(jobs) > self.MAX_REQUESTS:
            raise BatchSubmissionError(f"Too many jobs: {len(jobs)} > {self.MAX_REQUESTS}")
        
        # Validate all jobs
        for job in jobs:
            self.validate_job(job)
        
        # Prepare batch requests
        batch_requests = []
        self._job_mapping.clear()
        
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
            self._job_mapping[job.id] = job
        
        # Submit to Anthropic
        try:
            batch_response = self.client.messages.batches.create(requests=batch_requests)
            provider_batch_id = batch_response.id
        except Exception as e:
            raise BatchSubmissionError(f"Failed to create batch: {e}")
        
        return BatchRequest(
            id=provider_batch_id,
            provider_batch_id=provider_batch_id,
            submitted_at=datetime.now(),
            job_ids=[job.id for job in jobs]
        )
    
    def get_batch_status(self, batch_id: str) -> str:
        """Get current status of a batch."""
        try:
            batch_status = self.client.messages.batches.retrieve(batch_id)
            status = batch_status.processing_status
            
            # Map Anthropic statuses to our standard statuses
            if status == "ended":
                return "complete"
            elif status in ["canceled", "expired"]:
                return "failed"
            elif status in ["in_progress"]:
                return "running"
            else:
                return "pending"
        except Exception as e:
            raise ValidationError(f"Failed to get batch status: {e}")
    
    def get_batch_results(self, batch_id: str) -> List[JobResult]:
        """Retrieve results for a completed batch."""
        try:
            results = list(self.client.messages.batches.results(batch_id))
            return parse_results(results, self._job_mapping)
        except Exception as e:
            raise ValidationError(f"Failed to get batch results: {e}")
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs using tokencost."""
        try:
            from tokencost import calculate_prompt_cost, calculate_cost_by_tokens
        except ImportError:
            return 0.0  # Return 0 if tokencost not available
        
        total_cost = 0.0
        
        for job in jobs:
            try:
                # Prepare messages to get actual input
                messages, system_prompt = prepare_messages(job)
                
                # Add system prompt to messages for cost calculation
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                
                # Calculate input cost
                input_cost = float(calculate_prompt_cost(messages, job.model))
                
                # Calculate output cost
                output_cost = float(calculate_cost_by_tokens(
                    job.max_tokens, 
                    job.model, 
                    token_type="output"
                ))
                
                # Apply batch discount
                model_config = self.get_model_config(job.model)
                discount = model_config.batch_discount if model_config else 0.5
                job_cost = (input_cost + output_cost) * discount
                
                total_cost += job_cost
                
            except Exception:
                # Skip job if estimation fails
                continue
        
        return total_cost