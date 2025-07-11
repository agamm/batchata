"""Utility functions for bachata."""

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