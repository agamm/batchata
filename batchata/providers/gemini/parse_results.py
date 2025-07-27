"""Simple result parsing for Google Gemini API responses."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ...core.job_result import JobResult
from ...utils import to_dict


def parse_results(results: List[Dict], job_mapping: Dict[str, 'Job'], raw_files_dir: Optional[str] = None, batch_discount: float = 0.5, batch_id: Optional[str] = None) -> List[JobResult]:
    """Parse Gemini batch responses into JobResult objects."""
    job_results = []
    
    for result_data in results:
        job_id = result_data["job_id"]
        job = job_mapping[job_id]
        
        # Save raw response for debugging
        if raw_files_dir:
            _save_raw_response(result_data, job_id, raw_files_dir)
        
        # Handle errors
        if error := result_data.get("error"):
            job_results.append(JobResult(
                job_id=job_id, raw_response="", parsed_response=None,
                citations=None, citation_mappings=None,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                error=error, batch_id=batch_id
            ))
            continue
        
        # Extract response
        response = result_data.get("response")
        if not response:
            job_results.append(JobResult(
                job_id=job_id, raw_response="", parsed_response=None,
                citations=None, citation_mappings=None,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                error="No response in result", batch_id=batch_id
            ))
            continue
        
        # Get text content
        content = _extract_text(response)
        
        # Parse structured output
        parsed_response = None
        if job.response_model and content:
            parsed_response = _parse_structured(content, job.response_model)
        
        # Extract usage
        input_tokens, output_tokens = _extract_usage(response)
        
        # Calculate cost
        cost_usd = _calculate_cost(job.model, input_tokens, output_tokens, batch_discount)
        
        job_results.append(JobResult(
            job_id=job_id, raw_response=content, parsed_response=parsed_response,
            citations=None, citation_mappings=None,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, error=None, batch_id=batch_id
        ))
    
    return job_results


def _extract_text(response) -> str:
    """Extract text from Gemini response."""
    if hasattr(response, 'text') and response.text:
        return response.text
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
    
    return ""


def _parse_structured(content: str, response_model):
    """Parse structured output from content."""
    try:
        content = content.strip()
        
        # Try direct JSON parsing
        if content.startswith(('{', '[')):
            return response_model(**json.loads(content))
        
        # Extract JSON from text
        if (start := content.find('{')) != -1 and (end := content.rfind('}') + 1) > start:
            json_text = content[start:end]
            return response_model(**json.loads(json_text))
    
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    return None


def _extract_usage(response) -> tuple[int, int]:
    """Extract token usage from response."""
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        return (
            getattr(usage, 'prompt_token_count', 0),
            getattr(usage, 'candidates_token_count', 0)
        )
    return 0, 0


def _calculate_cost(model: str, input_tokens: int, output_tokens: int, batch_discount: float) -> float:
    """Calculate cost with batch discount."""
    try:
        import tokencost
        
        # Calculate input and output costs separately using correct signature
        input_cost = tokencost.calculate_cost_by_tokens(
            num_tokens=input_tokens,
            model=model,
            token_type='input'
        )
        output_cost = tokencost.calculate_cost_by_tokens(
            num_tokens=output_tokens,
            model=model,
            token_type='output'
        )
        
        total_cost = float(input_cost + output_cost)
        return total_cost * (1 - batch_discount)
        
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        # Return zero cost if tokencost library unavailable or calculation fails
        return 0.0


def _save_raw_response(result, job_id: str, raw_files_dir: str) -> None:
    """Save raw response for debugging."""
    try:
        responses_dir = Path(raw_files_dir) / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to convert to dict, fall back to string representation
        try:
            serializable_data = to_dict(result)
            with open(responses_dir / f"{job_id}_raw.json", 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except (TypeError, AttributeError):
            # If JSON serialization fails, save as text representation
            with open(responses_dir / f"{job_id}_raw.txt", 'w') as f:
                f.write(str(result))
                
    except (OSError, PermissionError):
        # Ignore file saving errors - not critical for functionality
        pass