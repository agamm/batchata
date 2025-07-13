"""Result parsing for Anthropic API responses."""

import json
from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import BaseModel

from ...core.job_result import JobResult
from ...types import Citation


def parse_results(results: List[Any], job_mapping: Dict[str, 'Job']) -> List[JobResult]:
    """Parse Anthropic batch results into JobResult objects.
    
    Args:
        results: Raw results from Anthropic API
        job_mapping: Mapping of job ID to Job object
        
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
        
        # Handle failed results
        if result.result.type != "succeeded":
            job_results.append(JobResult(
                job_id=job_id,
                response="",
                error=f"Request failed: {result.result.type}"
            ))
            continue
        
        try:
            message = result.result.message
            
            # Extract text and citations
            full_text, citations = _parse_content(message.content, job)
            
            # Parse structured output if needed
            parsed_response = None
            if job.response_model:
                parsed_response = _extract_json_model(full_text, job.response_model)
            
            # Extract usage
            usage = getattr(message, 'usage', None)
            input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
            output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
            
            # Calculate cost using tokencost
            # Get batch discount from model (would need provider reference for model config)
            batch_discount = 0.5  # Default Anthropic batch discount
            cost_usd = _calculate_cost(input_tokens, output_tokens, job.model, batch_discount)
            
            job_results.append(JobResult(
                job_id=job_id,
                response=full_text,
                parsed_response=parsed_response,
                citations=citations if citations else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd
            ))
            
        except Exception as e:
            job_results.append(JobResult(
                job_id=job_id,
                response="",
                error=f"Failed to parse result: {str(e)}"
            ))
    
    return job_results


def _parse_content(content: Any, job: Optional['Job']) -> Tuple[str, List[Citation]]:
    """Parse content blocks to extract text and citations."""
    if isinstance(content, str):
        return content, []
    
    if not isinstance(content, list):
        return str(content), []
    
    text_parts = []
    citations = []
    
    for block in content:
        # Extract text
        if hasattr(block, 'text'):
            text_parts.append(block.text)
        
        # Extract citations if enabled  
        if job.enable_citations and hasattr(block, 'citations'):
            for cit in block.citations or []:
                citation = Citation(
                    text=getattr(cit, 'cited_text', ''),
                    source=getattr(cit, 'document_title', 'Document'),
                    metadata={
                        'type': getattr(cit, 'type', ''),
                        'document_index': getattr(cit, 'document_index', 0),
                        'start_page_number': getattr(cit, 'start_page_number', None),
                        'end_page_number': getattr(cit, 'end_page_number', None)
                    }
                )
                citations.append(citation)
    
    return "".join(text_parts), citations


def _extract_json_model(text: str, response_model: Type[BaseModel]) -> Optional[BaseModel]:
    """Extract JSON from text and parse into Pydantic model."""
    try:
        # Find JSON in text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            return None
        
        json_str = text[start_idx:end_idx]
        json_data = json.loads(json_str)
        return response_model(**json_data)
    except:
        return None


def _calculate_cost(input_tokens: int, output_tokens: int, model: str, batch_discount: float = 0.5) -> float:
    """Calculate cost for tokens using tokencost."""
    from tokencost import calculate_cost_by_tokens
    
    # Calculate costs using tokencost
    input_cost = float(calculate_cost_by_tokens(input_tokens, model, token_type="input"))
    output_cost = float(calculate_cost_by_tokens(output_tokens, model, token_type="output"))
    
    return (input_cost + output_cost) * batch_discount