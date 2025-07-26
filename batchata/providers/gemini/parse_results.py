"""Result parsing for Google Gemini API responses."""

import json
from pathlib import Path
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel

from ...core.job_result import JobResult
from ...utils import to_dict, get_logger


logger = get_logger(__name__)


def parse_results(results: List[Dict], job_mapping: Dict[str, 'Job'], raw_files_dir: str | None = None, batch_discount: float = 0.0, batch_id: str | None = None) -> List[JobResult]:
    """Parse Gemini API responses into JobResult objects.
    
    Args:
        results: List of response dictionaries from async processing
        job_mapping: Mapping of job ID to Job object
        raw_files_dir: Optional directory to save debug files
        batch_discount: Batch discount factor (0.0 for Gemini since no true batch)
        batch_id: Batch ID for mapping to raw files
        
    Returns:
        List of JobResult objects
    """
    job_results = []
    
    for result_data in results:
        job_id = result_data.get("job_id")
        job = job_mapping.get(job_id)
        
        # Job must exist in mapping
        if not job:
            raise ValueError(f"Job {job_id} not found in mapping")
        
        # Save raw response to disk if directory is provided (before any error handling)
        if raw_files_dir:
            _save_raw_response(result_data, job_id, raw_files_dir)
        
        # Handle failed results
        if result_data.get("error"):
            error_message = result_data["error"]
            
            job_results.append(JobResult(
                job_id=job_id,
                status="failed",
                content="",
                error=error_message,
                usage={},
                cost_usd=0.0,
                provider="gemini",
                model=job.model,
                parsed_response=None,
                citation_mappings={}
            ))
            continue
        
        # Extract the response content
        response = result_data.get("response")
        if not response:
            job_results.append(JobResult(
                job_id=job_id,
                status="failed",
                content="",
                error="No response in result",
                usage={},
                cost_usd=0.0,
                provider="gemini",
                model=job.model,
                parsed_response=None,
                citation_mappings={}
            ))
            continue
        
        # Extract text content from Gemini response
        content = ""
        if hasattr(response, 'text'):
            content = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        content += part.text
        
        # Parse structured output if response model is specified
        parsed_response = None
        if job.response_model and content:
            try:
                # Try to parse as JSON first
                if content.strip().startswith('{') or content.strip().startswith('['):
                    parsed_data = json.loads(content.strip())
                    parsed_response = job.response_model(**parsed_data)
                else:
                    # If not JSON, try to extract JSON from text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_text = content[json_start:json_end]
                        parsed_data = json.loads(json_text)
                        parsed_response = job.response_model(**parsed_data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse structured output for job {job_id}: {e}")
                # Continue with plain text response
        
        # Calculate usage and cost
        usage = {}
        cost_usd = 0.0
        
        if hasattr(response, 'usage_metadata'):
            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, 'prompt_token_count', 0),
                "completion_tokens": getattr(usage_meta, 'candidates_token_count', 0),
                "total_tokens": getattr(usage_meta, 'total_token_count', 0),
            }
            
            # Estimate cost using tokencost (if available)
            try:
                import tokencost
                cost_usd = tokencost.calculate_cost(
                    model_name=job.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
                # No batch discount for Gemini
            except (ImportError, Exception) as e:
                logger.debug(f"Could not calculate cost for job {job_id}: {e}")
        
        # Create JobResult
        job_results.append(JobResult(
            job_id=job_id,
            status="completed",
            content=content,
            error=None,
            usage=usage,
            cost_usd=cost_usd,
            provider="gemini",
            model=job.model,
            parsed_response=parsed_response,
            citation_mappings={}  # TODO: Implement citations if needed
        ))
    
    return job_results


def _save_raw_response(result: Any, job_id: str, raw_files_dir: str) -> None:
    """Save individual raw API response to disk.
    
    Args:
        result: Raw response from API
        job_id: Job ID for filename
        raw_files_dir: Directory to save to
    """
    try:
        raw_files_path = Path(raw_files_dir)
        responses_dir = raw_files_path / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        raw_response_file = responses_dir / f"{job_id}_raw.json"
        
        # Convert to dict using utility function
        raw_data = to_dict(result)
        
        with open(raw_response_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        logger.debug(f"Saved raw response for job {job_id} to {raw_response_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save raw response for job {job_id}: {e}")