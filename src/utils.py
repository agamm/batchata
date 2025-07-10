"""Utility functions for ai-batch."""

import threading
from typing import Type, Optional, get_origin, get_args, Callable
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


def run_jobs_with_conditional_parallel(max_parallel: int, condition_fn: Callable[[], bool], jobs: list, job_processor_fn: Callable) -> None:
    """
    Execute jobs in parallel, checking condition before starting each new job.
    
    Args:
        max_parallel: Maximum number of concurrent jobs
        condition_fn: Function that returns True if no more jobs should start
        jobs: List of jobs to process
        job_processor_fn: Function to process each job
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        remaining_jobs = jobs.copy()
        
        # Submit initial batch of jobs
        while remaining_jobs and len(futures) < max_parallel:
            if condition_fn():
                break
            job = remaining_jobs.pop(0)
            future = executor.submit(job_processor_fn, job)
            futures[future] = job
        
        # Process completed jobs and submit new ones
        while futures:
            # Get the next completed future
            completed_future = None
            for future in as_completed(futures):
                completed_future = future
                break
            
            if completed_future:
                try:
                    completed_future.result()
                except Exception:
                    pass  # Let caller handle errors
                
                # Remove completed job
                del futures[completed_future]
                
                # Submit next job if available and condition allows
                if remaining_jobs and not condition_fn():
                    job = remaining_jobs.pop(0)
                    new_future = executor.submit(job_processor_fn, job)
                    futures[new_future] = job