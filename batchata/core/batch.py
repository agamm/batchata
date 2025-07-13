"""Batch builder."""

import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .batch_config import BatchConfig
from .job import Job
from ..providers import get_provider
from ..types import Message


class Batch:
    """Builder for batch job configuration.
    
    Provides a fluent interface for configuring batch jobs with sensible defaults
    and validation. The batch can be configured with cost limits, default parameters,
    and progress callbacks.
    
    Example:
        >>> batch = Batch("./state", "./results", max_concurrent=10)
        ...     .defaults(model="claude-3-sonnet", temperature=0.7)
        ...     .add_cost_limit(usd=15.0)
        ...     .add_job(messages=[{"role": "user", "content": "Hello"}])
        ...     .add_job(file="./path/to/file.pdf", prompt: "Generate summary of file")
        >>> run = batch.run(wait=True)
    """
    
    def __init__(self, state_file: str, results_dir: str, max_concurrent: int = 10):
        """Initialize batch configuration.
        
        Args:
            state_file: Path to state file for persistence
            results_dir: Directory to store results
            max_concurrent: Maximum concurrent batch requests
        """
        self.config = BatchConfig(
            state_file=state_file,
            results_dir=results_dir,
            max_concurrent=max_concurrent
        )
    
    def defaults(self, **kwargs) -> 'Batch':
        """Set default parameters for all jobs.
        
        These defaults will be applied to all jobs unless overridden
        by job-specific parameters.
        
        Args:
            **kwargs: Default parameters (model, temperature, max_tokens, etc.)
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.defaults(model="claude-3-sonnet", temperature=0.7)
        """
        # Validate if model is provided
        if "model" in kwargs:
            self.config.validate_default_params(kwargs["model"])
        
        self.config.default_params.update(kwargs)
        return self
    
    def add_cost_limit(self, usd: float) -> 'Batch':
        """Add cost limit for the batch.
        
        The batch will stop accepting new jobs once the cost limit is reached.
        Active jobs will be allowed to complete.
        
        Args:
            usd: Cost limit in USD
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.add_cost_limit(usd=50.0)
        """
        if usd <= 0:
            raise ValueError("Cost limit must be positive")
        self.config.cost_limit_usd = usd
        return self
    
    def on_progress(self, callback: Callable[[Dict, float], None], interval: float = 3.0) -> 'Batch':
        """Set progress callback.
        
        The callback will be called periodically with progress statistics
        including completed jobs, total jobs, current cost, etc.
        
        Args:
            callback: Function that receives progress statistics and elapsed time
            interval: Interval in seconds between progress updates (default: 3.0)
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.on_progress(lambda stats, time: print(f"Progress: {stats['completed']}/{stats['total']}, {time}s"))
        """
        self.config.progress_callback = callback
        self.config.progress_interval = interval
        return self
    
    def add_job(
        self,
        messages: Optional[List[Message]] = None,
        file: Optional[Union[str, Path]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        enable_citations: bool = False,
        **kwargs
    ) -> 'Batch':
        """Add a job to the batch.
        
        Either provide messages OR file+prompt, not both. Parameters not provided
        will use the defaults set via the defaults() method.
        
        Args:
            messages: Chat messages for direct input
            file: File path for file-based input
            prompt: Prompt to use with file input
            model: Model to use (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens to generate (overrides default)
            response_model: Pydantic model for structured output
            enable_citations: Whether to extract citations
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.add_job(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     model="gpt-4"
            ... )
        """
        # Generate unique job ID
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        
        # Merge with defaults
        params = self.config.default_params.copy()
        
        # Update with provided parameters
        if model is not None:
            params["model"] = model
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add other kwargs
        params.update(kwargs)
        
        # Ensure model is provided
        if "model" not in params:
            raise ValueError("Model must be provided either in defaults or job parameters")
        
        # Validate parameters
        provider = get_provider(params["model"])
        # Extract params without model to avoid duplicate
        param_subset = {k: v for k, v in params.items() if k != "model"}
        provider.validate_params(params["model"], **param_subset)
        
        # Convert file path if string
        if isinstance(file, str):
            file = Path(file)
        
        # Create job
        job = Job(
            id=job_id,
            messages=messages,
            file=file,
            prompt=prompt,
            response_model=response_model,
            enable_citations=enable_citations,
            **params
        )
        
        self.config.jobs.append(job)
        return self
    
    def run(self, wait: bool = False) -> 'BatchRun':
        """Execute the batch.
        
        Creates a BatchRun instance and starts processing the jobs.
        
        Args:
            wait: If True, block until all jobs complete
            
        Returns:
            BatchRun instance for monitoring progress
            
        Raises:
            ValueError: If no jobs have been added
        """
        if not self.config.jobs:
            raise ValueError("No jobs added to batch")
        
        # Import here to avoid circular dependency
        from .batch_run import BatchRun
        
        # Create and start the run
        run = BatchRun(self.config)
        run.start()
        
        if wait:
            run.wait()
        
        return run
    
    def __len__(self) -> int:
        """Get the number of jobs in the batch."""
        return len(self.config.jobs)
    
    def __repr__(self) -> str:
        """String representation of the batch."""
        return (
            f"Batch(jobs={len(self.config.jobs)}, "
            f"max_concurrent={self.config.max_concurrent}, "
            f"cost_limit=${self.config.cost_limit_usd or 'None'})"
        )