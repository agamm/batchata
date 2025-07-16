"""OpenAI provider implementation."""

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from ...core.job import Job
from ...core.job_result import JobResult
from ...exceptions import BatchSubmissionError, ValidationError
from ...utils import get_logger
from ..provider import Provider
from .models import OPENAI_MODELS
from .message_prepare import prepare_jsonl_request
from .parse_results import parse_results


logger = get_logger(__name__)


class OpenAIProvider(Provider):
    """OpenAI provider for batch processing."""
    
    # Batch limitations
    MAX_REQUESTS = 50_000
    MAX_FILE_SIZE_MB = 200
    
    # Batch discount constant
    BATCH_DISCOUNT = 0.5
    
    def __init__(self, auto_register: bool = True):
        """Initialize OpenAI provider."""
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it with your OpenAI API key."
            )
        
        self.client = OpenAI()
        super().__init__()
        self.models = OPENAI_MODELS
    
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format."""
        # Check if model is supported
        if not self.supports_model(job.model):
            raise ValidationError(f"Unsupported model: {job.model}")
        
        # Check file capabilities
        model_config = self.get_model_config(job.model)
        if job.file and not model_config.supports_files:
            raise ValidationError(f"Model '{job.model}' does not support file input")
        
        # OpenAI doesn't support citations
        if job.enable_citations:
            logger.warning(
                f"Job {job.id}: Citations are enabled but OpenAI doesn't support citations. "
                "Citations will be ignored."
            )
        
        # Validate messages by preparing them
        if job.messages or (job.file and job.prompt):
            try:
                from .message_prepare import prepare_messages
                prepare_messages(job)
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
        
        # Create batch-specific job mapping
        job_mapping = {job.id: job for job in jobs}
        
        # Prepare JSONL content
        jsonl_lines = []
        for job in jobs:
            request = prepare_jsonl_request(job)
            jsonl_lines.append(json.dumps(request))
        
        jsonl_content = "\n".join(jsonl_lines)
        
        # Check file size
        file_size_mb = len(jsonl_content.encode('utf-8')) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise BatchSubmissionError(f"Batch file too large: {file_size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB")
        
        try:
            # Create temporary file and upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
                tmp_file.write(jsonl_content)
                tmp_file_path = tmp_file.name
            
            logger.info(f"Uploading batch file with {len(jobs)} requests to OpenAI")
            
            # Upload file to OpenAI
            with open(tmp_file_path, 'rb') as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Create batch
            logger.info(f"Creating batch with file {uploaded_file.id}")
            batch_response = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            provider_batch_id = batch_response.id
            logger.info(f"✓ OpenAI batch created successfully: {provider_batch_id}")
            
        except Exception as e:
            # Clean up temp file if it exists
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            logger.error(f"✗ Failed to create OpenAI batch: {e}")
            raise BatchSubmissionError(f"Failed to create batch: {e}")
        
        return provider_batch_id, job_mapping
    
    def get_batch_status(self, batch_id: str) -> tuple[str, Optional[Dict]]:
        """Get current status of a batch."""
        try:
            batch_status = self.client.batches.retrieve(batch_id)
            
            status = batch_status.status
            
            # Map OpenAI statuses to our standard statuses
            if status == "completed":
                return "complete", None
            elif status in ["failed", "expired", "cancelled"]:
                error_details = {
                    "batch_id": batch_id,
                    "reason": f"Batch {status}",
                    "status": status
                }
                if hasattr(batch_status, 'errors') and batch_status.errors:
                    error_details["errors"] = batch_status.errors
                return "failed", error_details
            elif status in ["in_progress", "finalizing"]:
                return "running", None
            elif status == "cancelling":
                return "cancelled", {"batch_id": batch_id, "reason": "Batch is being cancelled"}
            else:  # validating or other
                return "pending", None
                
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            return "failed", {"batch_id": batch_id, "error": str(e)}
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch request."""
        try:
            logger.info(f"Cancelling OpenAI batch: {batch_id}")
            self.client.batches.cancel(batch_id)
            logger.info(f"✓ Successfully cancelled OpenAI batch: {batch_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to cancel OpenAI batch {batch_id}: {e}")
            return False
    
    def get_batch_results(self, batch_id: str, job_mapping: Dict[str, Job], raw_responses_dir: Optional[str] = None) -> List[JobResult]:
        """Retrieve results for a completed batch."""
        try:
            # Get batch info to get output file ID
            batch_info = self.client.batches.retrieve(batch_id)
            
            if not batch_info.output_file_id:
                raise ValidationError(f"Batch {batch_id} has no output file")
            
            # Download results file
            logger.info(f"Downloading results for batch {batch_id}")
            file_response = self.client.files.content(batch_info.output_file_id)
            jsonl_content = file_response.text
            
            # Parse JSONL content
            results = []
            for line in jsonl_content.strip().split('\n'):
                if line.strip():
                    results.append(json.loads(line))
            
            # Parse results using our parser
            return parse_results(results, job_mapping, raw_responses_dir, self.BATCH_DISCOUNT)
            
        except Exception as e:
            raise ValidationError(f"Failed to get batch results: {e}")
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs using tokencost."""
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
                from .message_prepare import prepare_messages
                messages, response_format = prepare_messages(job)
                
                # Build full text for token estimation
                full_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle multipart content (images, etc.)
                        for part in content:
                            if part.get("type") == "text":
                                full_text += f"{role}: {part.get('text', '')}\\n\\n"
                    else:
                        full_text += f"{role}: {content}\\n\\n"
                
                # Add response format to token count if structured output
                if response_format:
                    full_text += json.dumps(response_format)
                
                # Estimate tokens
                input_tokens = token_count_simple(full_text)
                
                # Calculate costs using tokencost
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
                job_cost = (input_cost + output_cost) * self.BATCH_DISCOUNT
                
                logger.info(
                    f"Job {job.id}: ~{input_tokens} input tokens, "
                    f"{job.max_tokens} max output tokens, "
                    f"cost: ${job_cost:.6f} (with {int(self.BATCH_DISCOUNT*100)}% batch discount)"
                )
                
                total_cost += job_cost
                
            except Exception as e:
                logger.warning(f"Failed to estimate cost for job {job.id}: {e}")
                continue
        
        return total_cost