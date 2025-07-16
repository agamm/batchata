"""Result parsing for OpenAI API responses."""

import json
from pathlib import Path
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel

from ...core.job_result import JobResult
from ...utils import to_dict, get_logger


logger = get_logger(__name__)


def parse_results(results: List[Dict], job_mapping: Dict[str, 'Job'], raw_responses_dir: str | None = None, batch_discount: float = 0.5) -> List[JobResult]:
    """Parse OpenAI batch results into JobResult objects.
    
    Args:
        results: Parsed JSONL lines from the downloaded batch output file
        job_mapping: Mapping of job ID to Job object
        raw_responses_dir: Optional directory to save raw API responses
        batch_discount: Batch discount factor from provider
        
    Returns:
        List of JobResult objects
    """
    job_results = []
    
    for result in results:
        job_id = result.get("custom_id")
        job = job_mapping.get(job_id)
        
        # Job must exist in mapping
        if not job:
            raise ValueError(f"Job {job_id} not found in mapping")
        
        # Save raw response to disk if directory is provided (before any error handling)
        if raw_responses_dir:
            _save_raw_response(result, job_id, raw_responses_dir)
        
        # Handle failed results
        if result.get("error"):
            error_info = result["error"]
            error_message = f"Request failed: {error_info.get('message', 'Unknown error')}"
            
            job_results.append(JobResult(
                job_id=job_id,
                raw_response="",
                error=error_message
            ))
            continue
        
        try:
            response = result.get("response")
            if not response:
                job_results.append(JobResult(
                    job_id=job_id,
                    raw_response="",
                    error="No response in batch result"
                ))
                continue
            
            # Check HTTP status code
            status_code = response.get("status_code")
            if status_code != 200:
                job_results.append(JobResult(
                    job_id=job_id,
                    raw_response="",
                    error=f"HTTP error: {status_code}"
                ))
                continue
            
            # Extract message content from response body
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                job_results.append(JobResult(
                    job_id=job_id,
                    raw_response="",
                    error="No choices in response"
                ))
                continue
            
            message = choices[0].get("message", {})
            full_text = message.get("content", "")
            
            # Parse structured output if needed
            parsed_response = None
            if job.response_model and full_text:
                parsed_response = _extract_json_model(full_text, job.response_model)
            
            # Extract usage from response body
            usage = body.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # Calculate cost using tokencost with provider's batch discount
            cost_usd = _calculate_cost(input_tokens, output_tokens, job.model, batch_discount)
            
            job_results.append(JobResult(
                job_id=job_id,
                raw_response=full_text,
                parsed_response=parsed_response,
                citations=None,  # OpenAI doesn't support citations
                citation_mappings=None,  # OpenAI doesn't support citations
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd
            ))
            
        except Exception as e:
            job_results.append(JobResult(
                job_id=job_id,
                raw_response="",
                error=f"Failed to parse result: {str(e)}"
            ))
    
    return job_results


def _extract_json_model(text: str, response_model: Type[BaseModel]) -> BaseModel | None:
    """Extract JSON from text and parse into Pydantic model."""
    try:
        # For OpenAI structured output, the response should be direct JSON
        json_data = json.loads(text)
        return response_model(**json_data)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from markdown code blocks
        try:
            import re
            code_block_pattern = r'```(?:json)?\s*\n([\s\S]*?)\n```'
            match = re.search(code_block_pattern, text)
            
            if match:
                json_str = match.group(1)
            else:
                # Fall back to finding JSON in text
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                
                if start_idx == -1 or end_idx <= start_idx:
                    return None
                
                json_str = text[start_idx:end_idx]
            
            json_data = json.loads(json_str)
            return response_model(**json_data)
        except:
            return None
    except:
        return None


def _save_raw_response(result: Any, job_id: str, raw_responses_dir: str) -> None:
    """Save raw API response to disk."""
    try:
        raw_responses_path = Path(raw_responses_dir)
        raw_response_file = raw_responses_path / f"{job_id}_raw.json"
        
        # Convert to dict using utility function
        raw_data = to_dict(result)
        
        with open(raw_response_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        logger.debug(f"Saved raw response for job {job_id} to {raw_response_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save raw response for job {job_id}: {e}")


def _calculate_cost(input_tokens: int, output_tokens: int, model: str, batch_discount: float) -> float:
    """Calculate cost for tokens using tokencost."""
    try:
        from tokencost import calculate_cost_by_tokens
        
        # Calculate costs using tokencost
        input_cost = float(calculate_cost_by_tokens(input_tokens, model, token_type="input"))
        output_cost = float(calculate_cost_by_tokens(output_tokens, model, token_type="output"))
        
        return (input_cost + output_cost) * batch_discount
    except ImportError:
        logger.warning("tokencost not available, returning 0 cost")
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to calculate cost: {e}")
        return 0.0