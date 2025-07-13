"""Utility functions for bachata."""

import json
import os
import threading
from pathlib import Path
from typing import Type, Optional, get_origin, get_args, Callable, List, Dict, Any
from pydantic import BaseModel


def check_flat_model_for_citation_mapping(response_model: Optional[Type[BaseModel]], enable_citations: bool) -> None:
    """
    Validate that response model is flat when citation mapping is enabled.
    Citation mapping only works with flat Pydantic models, not nested ones.
    
    Raises ValueError if nested models are used with citations enabled.
    """
    if not (response_model and enable_citations):
        return
    
    def has_nested_model(field_type: Type) -> bool:
        # Direct BaseModel check
        if (hasattr(field_type, '__mro__') and 
            BaseModel in field_type.__mro__ and 
            field_type != BaseModel):
            return True
        
        # Check generic types (List[Model], Optional[Model], etc.)
        args = get_args(field_type)
        return args and any(has_nested_model(arg) for arg in args)
    
    for field_name, field_info in response_model.model_fields.items():
        if has_nested_model(field_info.annotation):
            raise ValueError(
                f"Citation mapping requires flat Pydantic models. "
                f"Field '{field_name}' contains nested model(s). "
                f"Please flatten your model structure when using citations."
            )


def run_jobs_with_conditional_parallel(
    max_parallel: int, 
    condition_fn: Callable[[], bool], 
    jobs: list, 
    job_processor_fn: Callable,
    shared_lock: Optional[threading.Lock] = None
) -> None:
    """
    Execute jobs in parallel with atomic cost checking to prevent race conditions.
    
    This implements the following logic:
    1. Start initial batch of jobs (up to max_parallel)
    2. Jobs run in parallel  
    3. When one finishes, atomically check condition and start new job if allowed
    4. Continue until all jobs complete or condition prevents new jobs
    
    Args:
        max_parallel: Maximum number of concurrent jobs
        condition_fn: Function that returns True if no more jobs should start (called under lock)
        jobs: List of jobs to process
        job_processor_fn: Function to process each job
        shared_lock: Optional shared lock for atomic condition checking (creates one if None)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # Use provided lock or create a new one for atomic operations
    atomic_lock = shared_lock if shared_lock is not None else threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        remaining_jobs = jobs.copy()
        
        # Start initial batch of jobs (up to max_parallel) without cost checking
        # Costs are only known after job completion, so we start optimistically
        while remaining_jobs and len(futures) < max_parallel:
            job = remaining_jobs.pop(0)
            future = executor.submit(job_processor_fn, job)
            futures[future] = job
        
        # Process completions and start new jobs atomically
        while futures:
            # Wait for next job completion
            completed_future = None
            for future in as_completed(futures):
                completed_future = future
                break
            
            if completed_future:
                # CRITICAL SECTION: Atomic job completion → condition check → new job decision
                with atomic_lock:
                    # Process the completed job (this updates costs)
                    try:
                        completed_future.result()
                    except Exception:
                        pass  # Let caller handle errors
                    
                    # Remove completed job from active set
                    del futures[completed_future]
                    
                    # Check if we can start a new job (condition is checked AFTER job completes)
                    # This ensures the condition sees the most up-to-date cost
                    if remaining_jobs and not condition_fn():
                        job = remaining_jobs.pop(0)
                        new_future = executor.submit(job_processor_fn, job)
                        futures[new_future] = job


def load_results_from_disk(
    results_dir: str, 
    response_model: Optional[Type[BaseModel]] = None
) -> List[Dict[str, Any]]:
    """
    Load and reconstruct BatchManager results from disk.
    
    Args:
        results_dir: Directory containing processed results (should have 'processed' subdir)
        response_model: Pydantic model class to reconstruct results with
        
    Returns:
        List of result entries in unified format: [{"result": ..., "citations": ...}, ...]
        - If response_model provided: result field contains Pydantic model instances
        - If no response_model: result field contains raw data (str, dict, etc.)
        
    Example:
        ```python
        from batchata.utils import load_results_from_disk
        from pydantic import BaseModel
        
        class Invoice(BaseModel):
            company_name: str
            total_amount: float
            
        # Load results and reconstruct Pydantic models
        results = load_results_from_disk("./results", Invoice)
        
        for entry in results:
            invoice = entry["result"]  # This is an Invoice instance
            citations = entry["citations"]  # This contains Citation objects
            print(f"Company: {invoice.company_name}, Amount: {invoice.total_amount}")
        ```
    """
    processed_dir = Path(results_dir) / "processed"
    
    if not processed_dir.exists():
        return []
    
    all_results = []
    
    # Load all result files in order
    result_files = sorted([f for f in processed_dir.glob("job_*_results.json")])
    
    for file_path in result_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            job_results = json.load(f)
            
        # Reconstruct each result entry
        for entry in job_results:
            reconstructed_entry = _reconstruct_result_entry(entry, response_model)
            all_results.append(reconstructed_entry)
    
    return all_results


def _reconstruct_result_entry(entry: Dict[str, Any], response_model: Optional[Type[BaseModel]]) -> Dict[str, Any]:
    """
    Reconstruct a single result entry from JSON data.
    
    Args:
        entry: JSON data for one result entry
        response_model: Optional Pydantic model to reconstruct
        
    Returns:
        Result entry in unified format with Pydantic models and Citation objects
    """
    result_data = entry["result"]
    citations_data = entry.get("citations")
    
    # Reconstruct the result based on response_model
    if response_model and isinstance(result_data, dict):
        # Reconstruct Pydantic model from dictionary
        reconstructed_result = response_model(**result_data)
    else:
        # Keep as-is (string, number, etc.)
        reconstructed_result = result_data
    
    # Reconstruct citations if present
    reconstructed_citations = None
    if citations_data is not None:
        reconstructed_citations = _reconstruct_citations(citations_data)
    
    return {
        "result": reconstructed_result,
        "citations": reconstructed_citations
    }


def _reconstruct_citations(citations_data: Any) -> Any:
    """
    Reconstruct Citation objects from JSON data.
    
    Args:
        citations_data: JSON representation of citations
        
    Returns:
        Reconstructed citations (dict of lists for field citations, or list for text citations)
    """
    # Import here to avoid circular imports
    from .citations import Citation
    
    def _safe_create_citation(citation_dict: dict) -> Citation:
        """Safely create Citation with required defaults."""
        if not isinstance(citation_dict, dict):
            return citation_dict
        
        # Ensure required fields are present with sensible defaults
        citation_data = citation_dict.copy()
        if 'type' not in citation_data:
            citation_data['type'] = 'text'
        if 'document_index' not in citation_data:
            citation_data['document_index'] = 0
            
        try:
            return Citation(**citation_data)
        except Exception:
            # If Citation creation fails, return the raw dict
            return citation_dict
    
    if isinstance(citations_data, dict):
        # Field citations: {"field_name": [citation_dicts], ...}
        reconstructed = {}
        for field_name, citation_list in citations_data.items():
            if isinstance(citation_list, list):
                reconstructed[field_name] = [
                    _safe_create_citation(citation_dict)
                    for citation_dict in citation_list
                ]
            else:
                reconstructed[field_name] = citation_list
        return reconstructed
    elif isinstance(citations_data, list):
        # List of citations
        return [
            _safe_create_citation(citation_dict)
            for citation_dict in citations_data
        ]
    else:
        return citations_data
