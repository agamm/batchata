"""Batch run execution management."""

import json
import logging
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .batch_params import BatchParams
from .job import Job
from .job_result import JobResult
from ..providers.provider_registry import get_provider
from ..utils import CostTracker, StateManager, get_logger, set_log_level


logger = get_logger(__name__)


class BatchRun:
    """Manages the execution of a batch job synchronously.
    
    Processes jobs in batches based on items_per_batch configuration.
    Simpler synchronous execution for clear logging and debugging.
    
    Example:
        >>> config = BatchParams(...)
        >>> run = BatchRun(config, jobs)
        >>> run.start()
        >>> results = run.results()
    """
    
    def __init__(self, config: BatchParams, jobs: List[Job]):
        """Initialize batch run.
        
        Args:
            config: Batch configuration
            jobs: List of jobs to execute
        """
        self.config = config
        self.jobs = jobs
        
        # Set logging level based on config
        set_log_level(level=config.verbosity.upper())
        
        # Initialize components
        self.cost_tracker = CostTracker(limit_usd=config.cost_limit_usd)
        self.state_manager = StateManager(config.state_file)
        
        # State tracking
        self.pending_jobs: List[Job] = []
        self.completed_results: Dict[str, JobResult] = {}  # job_id -> result
        self.failed_jobs: Dict[str, str] = {}  # job_id -> error
        
        # Batch tracking
        self.total_batches = 0
        self.completed_batches = 0
        self.current_batch_index = 0
        self.current_batch_size = 0
        
        # Execution control
        self._started = False
        self._start_time: Optional[datetime] = None
        self._progress_callback: Optional[Callable[[Dict, float], None]] = None
        self._progress_interval: float = 1.0  # Default to 1 second
        
        # Results directory
        self.results_dir = Path(config.results_dir)
        
        # If not reusing state, clear the results directory
        if not config.reuse_state and self.results_dir.exists():
            import shutil
            shutil.rmtree(self.results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Raw responses directory (if enabled)
        self.raw_responses_dir = None
        if config.save_raw_responses:
            self.raw_responses_dir = self.results_dir / "raw_responses"
            self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to resume from saved state
        self._resume_from_state()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._shutdown_event.set()  # Signal shutdown to execution loop
            self.shutdown(wait_for_active=False)  # Don't wait for active jobs on Ctrl+C
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _resume_from_state(self):
        """Resume from saved state if available."""
        # Check if we should reuse state
        if not self.config.reuse_state:
            # Clear any existing state and start fresh
            self.state_manager.clear()
            self.pending_jobs = list(self.jobs)
            return
            
        state = self.state_manager.load()
        if state is None:
            # No saved state, use jobs passed to constructor
            self.pending_jobs = list(self.jobs)
            return
        
        logger.info("Resuming batch run from saved state")
        
        # Restore pending jobs
        self.pending_jobs = []
        for job_data in state.pending_jobs:
            job = Job.from_dict(job_data)
            self.pending_jobs.append(job)
        
        # Restore completed results
        for result_data in state.completed_results:
            result = JobResult.from_dict(result_data)
            self.completed_results[result.job_id] = result
        
        # Restore failed jobs
        for job_data in state.failed_jobs:
            self.failed_jobs[job_data["id"]] = job_data.get("error", "Unknown error")
        
        # Restore cost tracker
        self.cost_tracker.track_spend(state.total_cost_usd)
        
        logger.info(
            f"Resumed with {len(self.pending_jobs)} pending, "
            f"{len(self.completed_results)} completed, "
            f"{len(self.failed_jobs)} failed"
        )
    
    def to_json(self) -> Dict:
        """Convert current state to JSON-serializable dict."""
        return {
            "created_at": datetime.now().isoformat(),
            "pending_jobs": [job.to_dict() for job in self.pending_jobs],
            "completed_results": [result.to_dict() for result in self.completed_results.values()],
            "failed_jobs": [
                {
                    "id": job_id, 
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                } for job_id, error in self.failed_jobs.items()
            ],
            "total_cost_usd": self.cost_tracker.used_usd,
            "config": {
                "state_file": self.config.state_file,
                "results_dir": self.config.results_dir,
                "max_concurrent": self.config.max_concurrent,
                "items_per_batch": self.config.items_per_batch,
                "cost_limit_usd": self.config.cost_limit_usd,
                "default_params": self.config.default_params,
                "save_raw_responses": self.config.save_raw_responses
            }
        }
    
    def start(self):
        """Start synchronous batch execution."""
        if self._started:
            raise RuntimeError("Batch run already started")
        
        self._started = True
        self._start_time = datetime.now()
        
        logger.info("Starting batch run")
        
        # Call initial progress
        if self._progress_callback:
            stats = self.status()
            self._progress_callback(stats, 0.0)
        
        # Process all jobs synchronously
        self._process_all_jobs()
        
        logger.info("Batch run completed")
    
    def set_on_progress(self, callback: Callable[[Dict, float], None], interval: float = 1.0) -> 'BatchRun':
        """Set progress callback for execution monitoring.
        
        The callback will be called periodically with progress statistics
        including completed jobs, total jobs, current cost, etc.
        
        Args:
            callback: Function that receives (stats_dict, elapsed_time_seconds)
                     - stats_dict: Progress statistics dictionary
                     - elapsed_time_seconds: Time elapsed since batch started (float)
            interval: Interval in seconds between progress updates (default: 1.0)
            
        Returns:
            Self for chaining
            
        Example:
            >>> run.set_on_progress(lambda stats, time: print(f"Progress: {stats['completed']}/{stats['total']}, {time:.1f}s"))
        """
        self._progress_callback = callback
        self._progress_interval = interval
        return self
    
    def _process_all_jobs(self):
        """Process all jobs synchronously."""
        # Group jobs by provider
        jobs_by_provider = self._group_jobs_by_provider()
        
        # Calculate total batches
        self.total_batches = 0
        for provider_jobs in jobs_by_provider.values():
            batches = self._split_into_batches(provider_jobs)
            self.total_batches += len(batches)
        
        for provider_name, provider_jobs in jobs_by_provider.items():
            logger.info(f"Processing {len(provider_jobs)} jobs for {provider_name}")
            
            # Split into batches based on items_per_batch
            batches = self._split_into_batches(provider_jobs)
            
            for i, batch_jobs in enumerate(batches):
                self.current_batch_index += 1
                self.current_batch_size = len(batch_jobs)
                logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch_jobs)} jobs")
                self._process_batch(batch_jobs)
                
                # Call progress callback after each batch
                if self._progress_callback:
                    stats = self.status()
                    elapsed_time = round((datetime.now() - self._start_time).total_seconds())
                    self._progress_callback(stats, elapsed_time)
                
                # Save state after each batch
                self.state_manager.save(self)
    
    def _group_jobs_by_provider(self) -> Dict[str, List[Job]]:
        """Group jobs by provider."""
        jobs_by_provider = {}
        
        for job in self.pending_jobs[:]:  # Copy to avoid modification during iteration
            try:
                provider = get_provider(job.model)
                provider_name = provider.__class__.__name__
                
                if provider_name not in jobs_by_provider:
                    jobs_by_provider[provider_name] = []
                
                jobs_by_provider[provider_name].append(job)
                
            except Exception as e:
                logger.error(f"Failed to get provider for job {job.id}: {e}")
                self.failed_jobs[job.id] = str(e)
                self.pending_jobs.remove(job)
        
        return jobs_by_provider
    
    def _split_into_batches(self, jobs: List[Job]) -> List[List[Job]]:
        """Split jobs into batches based on items_per_batch."""
        batches = []
        batch_size = self.config.items_per_batch
        
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_batch(self, jobs: List[Job]):
        """Process a single batch of jobs synchronously."""
        if not jobs:
            return
        
        # Get provider
        provider = get_provider(jobs[0].model)
        
        # Check cost limit
        logger.info(f"Estimating cost for batch of {len(jobs)} jobs...")
        estimated_cost = provider.estimate_cost(jobs)
        logger.info(f"Total estimated cost: ${estimated_cost:.4f}, remaining budget: ${self.cost_tracker.remaining():.4f}")
        
        if not self.cost_tracker.can_afford(estimated_cost):
            logger.warning(f"Cost limit would be exceeded, skipping batch of {len(jobs)} jobs")
            for job in jobs:
                self.failed_jobs[job.id] = "Cost limit exceeded"
                self.pending_jobs.remove(job)
            return
        
        try:
            # Create batch
            logger.info(f"Creating batch with {len(jobs)} jobs...")
            batch_id = provider.create_batch(jobs)
            
            # Poll for completion
            logger.info(f"Polling for batch {batch_id} completion...")
            status, error_details = provider.get_batch_status(batch_id)
            logger.info(f"Initial batch status: {status}")
            poll_count = 0
            
            while status not in ["complete", "failed"]:
                poll_count += 1
                logger.debug(f"Polling attempt {poll_count}, current status: {status}")
                
                # Sleep for the progress interval duration
                time.sleep(self._progress_interval)
                status, error_details = provider.get_batch_status(batch_id)
                
                # Call progress callback after each interval
                if self._progress_callback:
                    stats = self.status()
                    elapsed_time = round((datetime.now() - self._start_time).total_seconds())
                    self._progress_callback(stats, elapsed_time)
                
                elapsed_seconds = poll_count * self._progress_interval
                logger.info(f"Batch {batch_id} status: {status} (polling for {elapsed_seconds:.1f}s)")
            
            if status == "failed":
                error_msg = f"Batch failed: {batch_id}"
                if error_details:
                    error_msg = f"Batch failed: {error_details.get('error', error_details.get('reason', 'Unknown error'))}"
                    logger.error(f"Batch {batch_id} failed with details: {error_details}")
                else:
                    logger.error(f"Batch {batch_id} failed")
                
                for job in jobs:
                    self.failed_jobs[job.id] = error_msg
                    self.pending_jobs.remove(job)
                
                # Save raw responses even on failure if configured
                if self.raw_responses_dir and error_details:
                    self._save_batch_error_details(batch_id, error_details)
                
                return
            
            # Get results
            logger.info(f"Getting results for batch {batch_id}")
            raw_responses_path = str(self.raw_responses_dir) if self.raw_responses_dir else None
            results = provider.get_batch_results(batch_id, raw_responses_path)
            
            # Track actual cost
            actual_cost = sum(r.cost_usd for r in results)
            self.cost_tracker.track_spend(actual_cost)
            
            # Process results
            for result in results:
                if result.is_success:
                    self.completed_results[result.job_id] = result
                    self._save_result_to_file(result)
                    logger.info(f"✓ Job {result.job_id} completed successfully")
                else:
                    self.failed_jobs[result.job_id] = result.error or "Unknown error"
                    # Save failed result to file as well for debugging
                    self._save_result_to_file(result)
                    logger.error(f"✗ Job {result.job_id} failed: {result.error}")
                
                # Remove from pending
                for job in jobs:
                    if job.id == result.job_id:
                        self.pending_jobs.remove(job)
                        break
            
            self.completed_batches += 1
            logger.info(
                f"✓ Batch {batch_id} completed: "
                f"{len([r for r in results if r.is_success])} success, "
                f"{len([r for r in results if not r.is_success])} failed, "
                f"cost: ${actual_cost:.6f}"
            )
            
        except Exception as e:
            logger.error(f"✗ Batch execution failed: {e}")
            for job in jobs:
                self.failed_jobs[job.id] = str(e)
                self.pending_jobs.remove(job)
    
    def _save_result_to_file(self, result: JobResult):
        """Save individual result to file."""
        result_file = self.results_dir / f"{result.job_id}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save result for {result.job_id}: {e}")
    
    def _save_batch_error_details(self, batch_id: str, error_details: Dict):
        """Save batch error details to raw responses directory."""
        try:
            error_file = self.raw_responses_dir / f"batch_{batch_id}_error.json"
            with open(error_file, 'w') as f:
                json.dump({
                    "batch_id": batch_id,
                    "timestamp": datetime.now().isoformat(),
                    "error_details": error_details
                }, f, indent=2)
            logger.info(f"Saved batch error details to {error_file}")
        except Exception as e:
            logger.error(f"Failed to save batch error details: {e}")
    
    def _is_complete(self) -> bool:
        """Check if all jobs are complete."""
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_results) + len(self.failed_jobs)
        return len(self.pending_jobs) == 0 and completed_count == total_jobs
    
    @property
    def is_complete(self) -> bool:
        """Whether all jobs are complete."""
        return self._is_complete()
    
    def wait(self, timeout: Optional[float] = None):
        """Wait for batch to complete (no-op for synchronous execution)."""
        pass
    
    def status(self, print_status: bool = False) -> Dict:
        """Get current execution statistics."""
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_results) + len(self.failed_jobs)
        remaining_count = total_jobs - completed_count
        
        stats = {
            "total": total_jobs,
            "pending": remaining_count,
            "active": 0,  # Always 0 for synchronous execution
            "completed": len(self.completed_results),
            "failed": len(self.failed_jobs),
            "cost_usd": self.cost_tracker.used_usd,
            "cost_limit_usd": self.cost_tracker.limit_usd,
            "is_complete": self._is_complete(),
            "batches_total": self.total_batches,
            "batches_completed": self.completed_batches,
            "batches_pending": self.total_batches - self.completed_batches,
            "current_batch_index": self.current_batch_index,
            "current_batch_size": self.current_batch_size,
            "items_per_batch": self.config.items_per_batch
        }
        
        if print_status:
            logger.info("\nBatch Run Status:")
            logger.info(f"  Total jobs: {stats['total']}")
            logger.info(f"  Pending: {stats['pending']}")
            logger.info(f"  Active: {stats['active']}")
            logger.info(f"  Completed: {stats['completed']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Cost: ${stats['cost_usd']:.6f}")
            if stats['cost_limit_usd']:
                logger.info(f"  Cost limit: ${stats['cost_limit_usd']:.2f}")
            logger.info(f"  Complete: {stats['is_complete']}")
        
        return stats
    
    def results(self) -> Dict[str, JobResult]:
        """Get all completed results."""
        return dict(self.completed_results)
    
    def get_failed_jobs(self) -> Dict[str, str]:
        """Get failed jobs with error messages."""
        return dict(self.failed_jobs)
    
    def shutdown(self):
        """Shutdown (no-op for synchronous execution)."""
        pass