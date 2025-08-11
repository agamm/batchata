"""Result parsing for Anthropic API responses."""

import json
from pathlib import Path
from typing import List, Dict, Any, Type, Tuple, Optional
from pydantic import BaseModel

from ...core.job_result import JobResult
from ...types import Citation
from ...utils import to_dict, get_logger


logger = get_logger(__name__)


def parse_results(results: List[Any], job_mapping: Dict[str, 'Job'], raw_files_dir: str | None = None, batch_discount: float = 0.5, batch_id: str | None = None) -> List[JobResult]:
    """Parse Anthropic batch results into JobResult objects.
    
    Args:
        results: Raw results from Anthropic API
        job_mapping: Mapping of job ID to Job object
        raw_files_dir: Optional directory to save debug files
        batch_discount: Batch discount factor from provider
        batch_id: Batch ID for mapping to raw files
        
    Returns:
        List of JobResult objects
    """
    job_results = []
    
    for result in results:
        job_id = result.custom_id
        job = job_mapping.get(job_id)
        
        # Job must exist in mapping
        if not job:
            raise ValueError(f"Job {job_id} not found in mapping")
        
        # Save raw response to disk if directory is provided (before any error handling)
        if raw_files_dir:
            _save_raw_response(result, job_id, raw_files_dir)
        
        # Handle failed results
        if result.result.type != "succeeded":
            error_message = f"Request failed: {result.result.type}"
            if hasattr(result.result, 'error') and result.result.error:
                # Try to get nested error message first, fall back to direct message
                if hasattr(result.result.error, 'error') and hasattr(result.result.error.error, 'message'):
                    error_message = f"Request failed: {result.result.error.error.message}"
                elif hasattr(result.result.error, 'message'):
                    error_message = f"Request failed: {result.result.error.message}"
            
            job_results.append(JobResult(
                job_id=job_id,
                raw_response="",
                error=error_message,
                batch_id=batch_id
            ))
            continue
        
        try:
            message = result.result.message
            
            # Extract text and citation blocks
            full_text, citation_blocks = _parse_content(message.content, job)
            
            # Parse structured output if needed
            parsed_response = None
            if job.response_model:
                parsed_response = _extract_json_model(full_text, job.response_model)
            
            # Process citations
            final_citations = None
            citation_mappings = None
            
            if citation_blocks:
                # Always populate citations list
                final_citations = [citation for _, citation in citation_blocks]
                
                # Try to map citations to fields if we have a response model
                if parsed_response:
                    from .citation_mapper import map_citations_to_fields
                    mappings, warning = map_citations_to_fields(
                        citation_blocks, 
                        parsed_response
                    )
                    citation_mappings = mappings if mappings else None
                    
                    if warning:
                        logger.warning(f"Job {job_id}: {warning}")
            
            # Extract usage
            usage = getattr(message, 'usage', None)
            input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
            output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
            
            # Calculate cost using tokencost with provided batch discount
            cost_usd = _calculate_cost(input_tokens, output_tokens, job.model, batch_discount)
            
            job_results.append(JobResult(
                job_id=job_id,
                raw_response=full_text,
                parsed_response=parsed_response,
                citations=final_citations,
                citation_mappings=citation_mappings,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                batch_id=batch_id
            ))
            
        except Exception as e:
            job_results.append(JobResult(
                job_id=job_id,
                raw_response="",
                error=f"Failed to parse result: {str(e)}",
                batch_id=batch_id
            ))
    
    return job_results


def _parse_content(content: Any, job: Optional['Job']) -> Tuple[str, List[Tuple[str, Citation]]]:
    """Parse content blocks to extract text and citation blocks.
    
    Returns:
        Tuple of (full_text, citation_blocks) where citation_blocks is
        a list of (block_text, citation) tuples.
    """
    if isinstance(content, str):
        return content, []
    
    if not isinstance(content, list):
        return str(content), []
    
    text_parts = []
    citation_blocks = []
    
    for block in content:
        block_text = ""
        
        # Extract text
        if hasattr(block, 'text'):
            block_text = block.text
            text_parts.append(block_text)
        
        # Extract citations if enabled  
        if job and job.enable_citations and hasattr(block, 'citations'):
            for cit in block.citations or []:
                citation = Citation(
                    text=getattr(cit, 'cited_text', ''),
                    source=getattr(cit, 'document_title', 'Document'),
                    page=getattr(cit, 'start_page_number', None),  # Set page directly
                    metadata={
                        'type': getattr(cit, 'type', ''),
                        'document_index': getattr(cit, 'document_index', 0),
                        'start_page_number': getattr(cit, 'start_page_number', None),
                        'end_page_number': getattr(cit, 'end_page_number', None)
                    }
                )
                citation_blocks.append((block_text, citation))
    
    return "".join(text_parts), citation_blocks


def _extract_json_model(text: str, response_model: Type[BaseModel]) -> BaseModel | Dict | None:
    """Extract JSON from text and parse into Pydantic model.
    
    Returns:
        - Pydantic model instance if validation succeeds
        - Dict with raw data and error info if Pydantic validation fails
        - None if JSON extraction fails completely
    """
    import re
    from pydantic import ValidationError
    
    json_str = None
    extraction_error = None
    
    # Try multiple patterns to extract JSON
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # More flexible: allows any whitespace
        r'```(?:json)?\s*\n([\s\S]*?)\n```',  # Original pattern
        r'```\s*([\s\S]*?)\s*```',  # Any code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            json_str = match.group(1).strip()
            logger.debug(f"Extracted JSON using pattern: {pattern}")
            break
    
    if not json_str:
        # Fall back to finding JSON object in text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            logger.warning("No JSON structure found in response text")
            return None
        
        json_str = text[start_idx:end_idx]
        logger.debug("Extracted JSON by finding braces")
    
    # Try to parse JSON
    try:
        json_data = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {list(json_data.keys())}")
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error at position {e.pos}: {e.msg}"
        logger.error(error_msg)
        # Return dict with error info
        return {
            "_error": "json_decode_failed", 
            "_error_details": error_msg,
            "_raw_json_attempt": json_str[:500]
        }
    
    # Try to create Pydantic model
    try:
        model_instance = response_model(**json_data)
        logger.debug(f"Successfully created {response_model.__name__} instance")
        return model_instance
    except ValidationError as e:
        # Log detailed validation errors
        error_details = []
        for error in e.errors():
            field = '.'.join(str(f) for f in error['loc'])
            msg = error['msg']
            error_details.append(f"{field}: {msg}")
        
        error_msg = f"Pydantic validation failed: {'; '.join(error_details)}"
        logger.warning(error_msg)
        
        # Return the raw dict with error info as fallback
        json_data["_error"] = "validation_failed"
        json_data["_error_details"] = error_details
        json_data["_expected_model"] = response_model.__name__
        return json_data
    except Exception as e:
        error_msg = f"Unexpected error creating model: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        # Return dict with error and raw data
        return {
            "_error": "model_creation_failed",
            "_error_details": error_msg,
            "_raw_data": json_data
        }


def _save_raw_response(result: Any, job_id: str, raw_files_dir: str) -> None:
    """Save raw API response to disk."""
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


def _calculate_cost(input_tokens: int, output_tokens: int, model: str, batch_discount: float = 0.5) -> float:
    """Calculate cost for tokens using tokencost."""
    from tokencost import calculate_cost_by_tokens
    
    # Calculate costs using tokencost
    input_cost = float(calculate_cost_by_tokens(input_tokens, model, token_type="input"))
    output_cost = float(calculate_cost_by_tokens(output_tokens, model, token_type="output"))
    
    return (input_cost + output_cost) * batch_discount