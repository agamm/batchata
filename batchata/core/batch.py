"""Batch builder."""

import logging
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
        ```python
        batch = Batch("./results", max_parallel_batches=10, items_per_batch=10)
            .set_state(file="./state.json", reuse_state=True)
            .set_default_params(model="claude-sonnet-4-20250514", temperature=0.7)
            .add_cost_limit(usd=15.0)
            .add_job(messages=[{"role": "user", "content": "Hello"}])
            .add_job(file="./path/to/file.pdf", prompt="Generate summary of file")
        
        run = batch.run()
        ```
    """
    
    def __init__(self, results_dir: str, max_parallel_batches: int = 10, items_per_batch: int = 10, raw_files: Optional[bool] = None):
        """Initialize batch configuration.
        
        Args:
            results_dir: Directory to store results
            max_parallel_batches: Maximum parallel batch requests
            items_per_batch: Number of jobs per provider batch
            raw_files: Whether to save debug files (raw responses, JSONL files) from providers (default: True if results_dir is set, False otherwise)
        """
        # Auto-determine raw_files based on results_dir if not explicitly set
        if raw_files is None:
            raw_files = bool(results_dir and results_dir.strip())
        
        self.config = BatchParams(
            state_file=None,
            results_dir=results_dir,
            max_parallel_batches=max_parallel_batches,
            items_per_batch=items_per_batch,
            reuse_state=True,
            raw_files=raw_files
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
            ```python
            batch.set_default_params(model="claude-3-sonnet", temperature=0.7)
            ```
        """
        # Validate if model is provided
        if "model" in kwargs:
            self.config.validate_default_params(kwargs["model"])
        
        self.config.default_params.update(kwargs)
        return self
    
    def set_state(self, file: Optional[str] = None, reuse_state: bool = True) -> 'Batch':
        """Set state file configuration.
        
        Args:
            file: Path to state file for persistence (default: None)
            reuse_state: Whether to resume from existing state file (default: True)
            
        Returns:
            Self for chaining
            
        Example:
            ```python
            batch.set_state(file="./state.json", reuse_state=True)
            ```
        """
        self.config.state_file = file
        self.config.reuse_state = reuse_state
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
            ```python
            batch.add_cost_limit(usd=50.0)
            ```
        """
        if usd <= 0:
            raise ValueError("Cost limit must be positive")
        self.config.cost_limit_usd = usd
        return self
    
    def raw_files(self, enabled: bool = True) -> 'Batch':
        """Enable or disable saving debug files from providers.
        
        When enabled, debug files (raw API responses, JSONL files) will be saved
        in a 'raw_files' subdirectory within the results directory.
        This is useful for debugging, auditing, or accessing provider-specific metadata.
        
        Args:
            enabled: Whether to save debug files (default: True)
            
        Returns:
            Self for chaining
            
        Example:
            ```python
            batch.raw_files(True)
            ```
        """
        self.config.raw_files = enabled
        return self
    
    def set_verbosity(self, level: str) -> 'Batch':
        """Set logging verbosity level.
        
        Args:
            level: Verbosity level ("debug", "info", "warn", "error")
            
        Returns:
            Self for chaining
            
        Example:
            ```python
            batch.set_verbosity("error")  # For production
            batch.set_verbosity("debug")  # For debugging
            ```
        """
        valid_levels = {"debug", "info", "warn", "error"}
        if level.lower() not in valid_levels:
            raise ValueError(f"Invalid verbosity level: {level}. Must be one of {valid_levels}")
        self.config.verbosity = level.lower()
        return self
    
    def add_time_limit(self, seconds: Optional[float] = None, minutes: Optional[float] = None, hours: Optional[float] = None) -> 'Batch':
        """Add time limit for the entire batch execution.
        
        When time limit is reached, all active provider batches are cancelled and 
        remaining unprocessed jobs are marked as failed. The batch execution 
        completes normally without throwing exceptions.
        
        Args:
            seconds: Time limit in seconds (optional)
            minutes: Time limit in minutes (optional)
            hours: Time limit in hours (optional)
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If no time units specified, or if total time is outside 
                       valid range (min: 10 seconds, max: 24 hours)
            
        Note:
            - Can combine multiple time units
            - Time limit is checked every second by a background watchdog thread
            - Jobs that exceed time limit appear in results()["failed"] with time limit error message
            - No exceptions are thrown when time limit is reached
            
        Example:
            ```python
            batch.add_time_limit(seconds=30)  # 30 seconds
            batch.add_time_limit(minutes=5)   # 5 minutes
            batch.add_time_limit(hours=2)     # 2 hours
            batch.add_time_limit(hours=1, minutes=30, seconds=15)  # 5415 seconds total
            ```
        """
        time_limit_seconds = 0.0
        
        if seconds is not None:
            time_limit_seconds += seconds
        if minutes is not None:
            time_limit_seconds += minutes * 60
        if hours is not None:
            time_limit_seconds += hours * 3600
            
        if time_limit_seconds == 0:
            raise ValueError("Must specify at least one of seconds, minutes, or hours")
            
        self.config.time_limit_seconds = time_limit_seconds
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
            ```python
            batch.add_job(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )
            ```
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
        
        # Warn about temporary file paths that may not persist
        if file:
            file_str = str(file)
            if "/tmp/" in file_str or "/var/folders/" in file_str or "temp" in file_str.lower():
                logger = logging.getLogger("batchata")
                logger.debug(f"File path appears to be in a temporary directory: {file}")
                logger.debug("This may cause issues when resuming from state if temp files are cleaned up")
        
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
        
        Creates a BatchRun instance and executes the jobs synchronously.
        
        Args:
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
            return self._run_with_rich_display(run, progress_interval)
        else:
            return self._run_with_custom_callback(run, on_progress, progress_interval)
    
    def _run_with_rich_display(self, run: 'BatchRun', progress_interval: float) -> 'BatchRun':
        """Execute batch run with rich progress display.
        
        Args:
            run: BatchRun instance to execute
            progress_interval: Interval between progress updates
            
        Returns:
            Completed BatchRun instance
        """
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
        
        # Execute with proper cleanup
        try:
            run.execute()
            
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
        
        return run
    
    def _run_with_custom_callback(self, run: 'BatchRun', on_progress: Optional[Callable[[Dict, float, Dict], None]], progress_interval: float) -> 'BatchRun':
        """Execute batch run with custom progress callback.
        
        Args:
            run: BatchRun instance to execute
            on_progress: Optional custom progress callback
            progress_interval: Interval between progress updates
            
        Returns:
            Completed BatchRun instance
        """
        # Use custom progress callback if provided
        if on_progress:
            run.set_on_progress(on_progress, interval=progress_interval)
        
        run.execute()
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