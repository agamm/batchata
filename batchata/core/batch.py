"""Batch builder."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Type, Optional, Union

from pydantic import BaseModel

from .batch_params import BatchParams
from .job import Job
from ..providers import get_provider
from ..types import Message


class Batch:
    """Builder for batch job configuration.
    
    Provides a fluent interface for configuring batch jobs with sensible defaults
    and validation. The batch can be configured with cost limits, default parameters,
    and progress callbacks.
    
    Example:
        >>> batch = Batch("./results", max_concurrent=10, items_per_batch=10)
        ...     .set_state(file="./state.json", reuse_previous=True)
        ...     .set_default_params(model="claude-3-sonnet", temperature=0.7)
        ...     .add_cost_limit(usd=15.0)
        ...     .add_job(messages=[{"role": "user", "content": "Hello"}])
        ...     .add_job(file="./path/to/file.pdf", prompt: "Generate summary of file")
        >>> run = batch.run(wait=True)
    """
    
    def __init__(self, results_dir: str, max_parallel_batches: int = 10, items_per_batch: int = 10, save_raw_responses: Optional[bool] = None):
        """Initialize batch configuration.
        
        Args:
            results_dir: Directory to store results
            max_parallel_batches: Maximum parallel batch requests
            items_per_batch: Number of jobs per provider batch
            save_raw_responses: Whether to save raw API responses from providers (default: True if results_dir is set, False otherwise)
        """
        # Auto-determine save_raw_responses based on results_dir if not explicitly set
        if save_raw_responses is None:
            save_raw_responses = bool(results_dir and results_dir.strip())
        
        self.config = BatchParams(
            state_file=None,
            results_dir=results_dir,
            max_parallel_batches=max_parallel_batches,
            items_per_batch=items_per_batch,
            reuse_state=True,
            save_raw_responses=save_raw_responses
        )
        self.jobs: List[Job] = []
    
    def set_default_params(self, **kwargs) -> 'Batch':
        """Set default parameters for all jobs.
        
        These defaults will be applied to all jobs unless overridden
        by job-specific parameters.
        
        Args:
            **kwargs: Default parameters (model, temperature, max_tokens, etc.)
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.set_default_params(model="claude-3-sonnet", temperature=0.7)
        """
        # Validate if model is provided
        if "model" in kwargs:
            self.config.validate_default_params(kwargs["model"])
        
        self.config.default_params.update(kwargs)
        return self
    
    def set_state(self, file: Optional[str] = None, reuse_previous: bool = True) -> 'Batch':
        """Set state file configuration.
        
        Args:
            file: Path to state file for persistence (default: None)
            reuse_previous: Whether to resume from existing state file (default: True)
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.set_state(file="./state.json", reuse_previous=True)
        """
        self.config.state_file = file
        self.config.reuse_state = reuse_previous
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
    
    def save_raw_responses(self, enabled: bool = True) -> 'Batch':
        """Enable or disable saving raw API responses from providers.
        
        When enabled, the raw API responses will be saved alongside the parsed results
        in a 'raw_responses' subdirectory within the results directory.
        This is useful for debugging, auditing, or accessing provider-specific metadata.
        
        Args:
            enabled: Whether to save raw responses (default: True)
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.save_raw_responses(True)
        """
        self.config.save_raw_responses = enabled
        return self
    
    def set_verbosity(self, level: str) -> 'Batch':
        """Set logging verbosity level.
        
        Args:
            level: Verbosity level ("debug", "info", "warn", "error")
            
        Returns:
            Self for chaining
            
        Example:
            >>> batch.set_verbosity("error")  # For production
            >>> batch.set_verbosity("debug")  # For debugging
        """
        valid_levels = {"debug", "info", "warn", "error"}
        if level.lower() not in valid_levels:
            raise ValueError(f"Invalid verbosity level: {level}. Must be one of {valid_levels}")
        self.config.verbosity = level.lower()
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
        
        # Validate citation compatibility
        if response_model and enable_citations:
            from ..utils.validation import validate_flat_model
            validate_flat_model(response_model)
        
        # Validate job with provider (includes PDF validation for Anthropic)
        provider.validate_job(job)
        
        
        self.jobs.append(job)
        return self
    
    def run(self, on_progress: Optional[Callable[[Dict, float, Dict], None]] = None, progress_interval: float = 1.0, print_status: bool = False) -> 'BatchRun':
        """Execute the batch.
        
        Creates a BatchRun instance and starts processing the jobs synchronously.
        
        Args:
            wait: Legacy parameter, ignored (execution is always synchronous)
            on_progress: Optional progress callback function that receives
                        (stats_dict, elapsed_time_seconds, batch_data)
            progress_interval: Interval in seconds between progress updates (default: 1.0)
            print_status: Whether to show rich progress display (default: False)
            
        Returns:
            BatchRun instance with completed results
            
        Raises:
            ValueError: If no jobs have been added
        """
        if not self.jobs:
            raise ValueError("No jobs added to batch")
        
        # Import here to avoid circular dependency
        from .batch_run import BatchRun
        
        # Create and start the run
        run = BatchRun(self.config, self.jobs)
        
        # Set progress callback - either rich display or custom callback
        if print_status:
            # Use rich progress display
            from ..utils.rich_progress import RichBatchProgressDisplay
            display = RichBatchProgressDisplay()
            
            def rich_progress_callback(stats, elapsed_time, batch_data):
                # Start display on first call
                if not hasattr(rich_progress_callback, '_started'):
                    config_dict = {
                        'results_dir': self.config.results_dir,
                        'state_file': self.config.state_file,
                        'items_per_batch': self.config.items_per_batch,
                        'max_parallel_batches': self.config.max_parallel_batches
                    }
                    display.start(stats, config_dict)
                    rich_progress_callback._started = True
                
                # Update display
                display.update(stats, batch_data, elapsed_time)
            
            run.set_on_progress(rich_progress_callback, interval=progress_interval)
            
            # Ensure display is stopped when done
            try:
                run.start()
                
                # Show final status with all batches completed
                stats = run.status()
                display.update(stats, run.batch_tracking, (datetime.now() - run._start_time).total_seconds())
                
                # Small delay to ensure display updates
                import time
                time.sleep(0.2)
                
            except KeyboardInterrupt:
                # Update batch tracking to show cancelled status for pending/running batches
                with run._state_lock:
                    for batch_id, batch_info in run.batch_tracking.items():
                        if batch_info['status'] == 'running':
                            batch_info['status'] = 'cancelled'
                        elif batch_info['status'] == 'pending':
                            batch_info['status'] = 'cancelled'
                
                # Show final status with cancelled batches
                stats = run.status()
                display.update(stats, run.batch_tracking, 0.0)
                
                # Add a small delay to ensure the display updates
                import time
                time.sleep(0.1)
                
                display.stop()
                raise
            finally:
                if display.live:  # Only stop if not already stopped
                    display.stop()
        else:
            # Use custom progress callback if provided
            if on_progress:
                run.set_on_progress(on_progress, interval=progress_interval)
            
            run.start()
        
        return run
    
    def __len__(self) -> int:
        """Get the number of jobs in the batch."""
        return len(self.jobs)
    
    def __repr__(self) -> str:
        """String representation of the batch."""
        return (
            f"Batch(jobs={len(self.jobs)}, "
            f"max_parallel_batches={self.config.max_parallel_batches}, "
            f"cost_limit=${self.config.cost_limit_usd or 'None'})"
        )