"""Batch run execution management."""

import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
        
        # Use temp file for state if not provided
        state_file = config.state_file
        if not state_file:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            temp_file.close()
            state_file = temp_file.name
        
        self.state_manager = StateManager(state_file)
        
        # State tracking
        self.pending_jobs: List[Job] = []
        self.completed_results: Dict[str, JobResult] = {}  # job_id -> result
        self.failed_jobs: Dict[str, str] = {}  # job_id -> error
        self.cancelled_jobs: Dict[str, str] = {}  # job_id -> reason
        
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
        
        # Threading primitives
        self._state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Batch tracking for progress display
        self.batch_tracking: Dict[str, Dict] = {}  # batch_id -> batch_info
        
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
        
        # Restore cancelled jobs (if they exist in state)
        for job_data in getattr(state, 'cancelled_jobs', []):
            self.cancelled_jobs[job_data["id"]] = job_data.get("reason", "Cancelled")
        
        # Restore cost tracker
        self.cost_tracker.used_usd = state.total_cost_usd
        
        logger.info(
            f"Resumed with {len(self.pending_jobs)} pending, "
            f"{len(self.completed_results)} completed, "
            f"{len(self.failed_jobs)} failed, "
            f"{len(self.cancelled_jobs)} cancelled"
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
            "cancelled_jobs": [
                {
                    "id": job_id, 
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                } for job_id, reason in self.cancelled_jobs.items()
            ],
            "total_cost_usd": self.cost_tracker.used_usd,
            "config": {
                "state_file": self.config.state_file,
                "results_dir": self.config.results_dir,
                "max_parallel_batches": self.config.max_parallel_batches,
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
        
        # Register signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.warning("Received interrupt signal, shutting down gracefully...")
            self._shutdown_event.set()
        
        # Store original handler to restore later
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            logger.info("Starting batch run")
            
            # Call initial progress
            if self._progress_callback:
                stats = self.status()
                batch_data = dict(self.batch_tracking)
                self._progress_callback(stats, 0.0, batch_data)
            
            # Process all jobs synchronously
            self._process_all_jobs()
            
            logger.info("Batch run completed")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
    
    def set_on_progress(self, callback: Callable[[Dict, float, Dict], None], interval: float = 1.0) -> 'BatchRun':
        """Set progress callback for execution monitoring.
        
        The callback will be called periodically with progress statistics
        including completed jobs, total jobs, current cost, etc.
        
        Args:
            callback: Function that receives (stats_dict, elapsed_time_seconds, batch_data)
                     - stats_dict: Progress statistics dictionary
                     - elapsed_time_seconds: Time elapsed since batch started (float)
                     - batch_data: Dictionary mapping batch_id to batch information
            interval: Interval in seconds between progress updates (default: 1.0)
            
        Returns:
            Self for chaining
            
        Example:
            >>> run.set_on_progress(lambda stats, time, batch_data: print(f"Progress: {stats['completed']}/{stats['total']}, {time:.1f}s"))
        """
        self._progress_callback = callback
        self._progress_interval = interval
        return self
    
    def _process_all_jobs(self):
        """Process all jobs with parallel execution."""
        # Prepare all batches
        batches = self._prepare_batches()
        self.total_batches = len(batches)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_batches) as executor:
            futures = [executor.submit(self._execute_batch_wrapped, provider, batch_jobs) 
                      for _, provider, batch_jobs in batches]
            
            try:
                for future in as_completed(futures):
                    # Stop if shutdown event detected
                    if self._shutdown_event.is_set():
                        break
                    future.result()  # Re-raise any exceptions
            except KeyboardInterrupt:
                self._shutdown_event.set()
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
                raise
    
    def _execute_batch_wrapped(self, provider, batch_jobs):
        """Thread-safe wrapper for _execute_batch."""
        try:
            result = self._execute_batch(provider, batch_jobs)
            with self._state_lock:
                self._update_batch_results(result)
                # Remove jobs from pending_jobs if specified
                jobs_to_remove = result.get("jobs_to_remove", [])
                for job in jobs_to_remove:
                    if job in self.pending_jobs:
                        self.pending_jobs.remove(job)
        except KeyboardInterrupt:
            self._shutdown_event.set()
            # Handle cancelled jobs with proper locking
            with self._state_lock:
                for job in batch_jobs:
                    self.cancelled_jobs[job.id] = "Cancelled by user"
                    if job in self.pending_jobs:
                        self.pending_jobs.remove(job)
                self.state_manager.save(self)
            raise
    
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
                with self._state_lock:
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
    
    def _prepare_batches(self) -> List[Tuple[str, object, List[Job]]]:
        """Prepare all batches as simple list of (provider_name, provider, jobs)."""
        batches = []
        jobs_by_provider = self._group_jobs_by_provider()
        
        for provider_name, provider_jobs in jobs_by_provider.items():
            provider = get_provider(provider_jobs[0].model)
            job_batches = self._split_into_batches(provider_jobs)
            
            for batch_jobs in job_batches:
                batches.append((provider_name, provider, batch_jobs))
                
                # Pre-populate batch tracking for pending batches
                batch_id = f"pending_{len(self.batch_tracking)}"
                self.batch_tracking[batch_id] = {
                    'start_time': None,
                    'status': 'pending',
                    'total': len(batch_jobs),
                    'completed': 0,
                    'cost': 0.0,
                    'jobs': batch_jobs
                }
        
        return batches
    
    def _poll_batch_status(self, provider, batch_id: str) -> Tuple[str, Optional[Dict]]:
        """Poll until batch completes."""
        status, error_details = provider.get_batch_status(batch_id)
        logger.info(f"Initial batch status: {status}")
        poll_count = 0
        
        while status not in ["complete", "failed"]:
            poll_count += 1
            logger.debug(f"Polling attempt {poll_count}, current status: {status}")
            
            # Interruptible wait - will wake up immediately if shutdown event is set
            if self._shutdown_event.wait(self._progress_interval):
                logger.info(f"Batch {batch_id} polling interrupted by shutdown")
                raise KeyboardInterrupt("Batch cancelled by user")
            
            status, error_details = provider.get_batch_status(batch_id)
            
            if self._progress_callback:
                with self._state_lock:
                    stats = self.status()
                    elapsed_time = round((datetime.now() - self._start_time).total_seconds())
                    batch_data = dict(self.batch_tracking)
                self._progress_callback(stats, elapsed_time, batch_data)
            
            elapsed_seconds = poll_count * self._progress_interval
            logger.info(f"Batch {batch_id} status: {status} (polling for {elapsed_seconds:.1f}s)")
        
        return status, error_details
    
    def _update_batch_results(self, batch_result: Dict):
        """Update state from batch results."""
        results = batch_result.get("results", [])
        failed = batch_result.get("failed", {})
        cost = batch_result.get("cost", 0.0)
        
        # Note: Cost already tracked by adjust_reserved_cost in _execute_batch
        
        # Update completed results
        for result in results:
            if result.is_success:
                self.completed_results[result.job_id] = result
                self._save_result_to_file(result)
                logger.info(f"✓ Job {result.job_id} completed successfully")
            else:
                self.failed_jobs[result.job_id] = result.error or "Unknown error"
                self._save_result_to_file(result)
                logger.error(f"✗ Job {result.job_id} failed: {result.error}")
        
        # Update failed jobs
        for job_id, error in failed.items():
            self.failed_jobs[job_id] = error
            logger.error(f"✗ Job {job_id} failed: {error}")
        
        # Update batch tracking
        self.completed_batches += 1
        
        # Save state
        self.state_manager.save(self)
    
    def _execute_batch(self, provider, batch_jobs: List[Job]) -> Dict:
        """Execute one batch, return results dict with jobs/costs/errors."""
        if not batch_jobs:
            return {"results": [], "failed": {}, "cost": 0.0}
        
        # Reserve cost limit
        logger.info(f"Estimating cost for batch of {len(batch_jobs)} jobs...")
        estimated_cost = provider.estimate_cost(batch_jobs)
        logger.info(f"Total estimated cost: ${estimated_cost:.4f}, remaining budget: ${self.cost_tracker.remaining():.4f}")
        
        if not self.cost_tracker.reserve_cost(estimated_cost):
            logger.warning(f"Cost limit would be exceeded, skipping batch of {len(batch_jobs)} jobs")
            failed = {}
            for job in batch_jobs:
                failed[job.id] = "Cost limit exceeded"
            return {"results": [], "failed": failed, "cost": 0.0, "jobs_to_remove": list(batch_jobs)}
        
        batch_id = None
        job_mapping = None
        try:
            # Create batch
            logger.info(f"Creating batch with {len(batch_jobs)} jobs...")
            batch_id, job_mapping = provider.create_batch(batch_jobs)
            
            # Track batch creation
            with self._state_lock:
                # Remove pending entry if it exists
                pending_keys = [k for k in self.batch_tracking.keys() if k.startswith('pending_')]
                for pending_key in pending_keys:
                    if self.batch_tracking[pending_key]['jobs'] == batch_jobs:
                        del self.batch_tracking[pending_key]
                        break
                
                # Add actual batch tracking
                self.batch_tracking[batch_id] = {
                    'start_time': datetime.now(),
                    'status': 'running',
                    'total': len(batch_jobs),
                    'completed': 0,
                    'cost': 0.0,
                    'jobs': batch_jobs
                }
            
            # Poll for completion
            logger.info(f"Polling for batch {batch_id} completion...")
            status, error_details = self._poll_batch_status(provider, batch_id)
            
            if status == "failed":
                error_msg = f"Batch failed: {batch_id}"
                if error_details:
                    error_msg = f"Batch failed: {error_details.get('error', error_details.get('reason', 'Unknown error'))}"
                    logger.error(f"Batch {batch_id} failed with details: {error_details}")
                else:
                    logger.error(f"Batch {batch_id} failed")
                
                # Update batch tracking for failure
                with self._state_lock:
                    if batch_id in self.batch_tracking:
                        self.batch_tracking[batch_id]['status'] = 'failed'
                        self.batch_tracking[batch_id]['error'] = error_msg
                
                # Release the reservation since batch failed
                self.cost_tracker.adjust_reserved_cost(estimated_cost, 0.0)
                
                failed = {}
                for job in batch_jobs:
                    failed[job.id] = error_msg
                
                # Save error details if configured
                if self.raw_responses_dir and error_details:
                    self._save_batch_error_details(batch_id, error_details)
                
                return {"results": [], "failed": failed, "cost": 0.0, "jobs_to_remove": list(batch_jobs)}
            
            # Get results
            logger.info(f"Getting results for batch {batch_id}")
            raw_responses_path = str(self.raw_responses_dir) if self.raw_responses_dir else None
            results = provider.get_batch_results(batch_id, job_mapping, raw_responses_path)
            
            # Calculate actual cost and adjust reservation
            actual_cost = sum(r.cost_usd for r in results)
            self.cost_tracker.adjust_reserved_cost(estimated_cost, actual_cost)
            
            # Update batch tracking for completion
            with self._state_lock:
                if batch_id in self.batch_tracking:
                    self.batch_tracking[batch_id]['status'] = 'complete'
                    self.batch_tracking[batch_id]['completed'] = len(results)
                    self.batch_tracking[batch_id]['cost'] = actual_cost
            
            logger.info(
                f"✓ Batch {batch_id} completed: "
                f"{len([r for r in results if r.is_success])} success, "
                f"{len([r for r in results if not r.is_success])} failed, "
                f"cost: ${actual_cost:.6f}"
            )
            
            return {"results": results, "failed": {}, "cost": actual_cost, "jobs_to_remove": list(batch_jobs)}
            
        except KeyboardInterrupt:
            logger.warning(f"\nCancelling batch{f' {batch_id}' if batch_id else ''}...")
            if batch_id:
                provider.cancel_batch(batch_id)
                # Update batch tracking for cancellation
                with self._state_lock:
                    if batch_id in self.batch_tracking:
                        self.batch_tracking[batch_id]['status'] = 'cancelled'
                        self.batch_tracking[batch_id]['error'] = 'Cancelled by user'
            # Release the reservation since batch was cancelled
            self.cost_tracker.adjust_reserved_cost(estimated_cost, 0.0)
            # Handle cancellation in the wrapper with proper locking
            raise
            
        except Exception as e:
            logger.error(f"✗ Batch execution failed: {e}")
            # Update batch tracking for exception
            if batch_id:
                with self._state_lock:
                    if batch_id in self.batch_tracking:
                        self.batch_tracking[batch_id]['status'] = 'failed'
                        self.batch_tracking[batch_id]['error'] = str(e)
            # Release the reservation since batch failed
            self.cost_tracker.adjust_reserved_cost(estimated_cost, 0.0)
            failed = {}
            for job in batch_jobs:
                failed[job.id] = str(e)
            return {"results": [], "failed": failed, "cost": 0.0, "jobs_to_remove": list(batch_jobs)}
    
    
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
    
    @property
    def is_complete(self) -> bool:
        """Whether all jobs are complete."""
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_results) + len(self.failed_jobs) + len(self.cancelled_jobs)
        return len(self.pending_jobs) == 0 and completed_count == total_jobs

    
    def status(self, print_status: bool = False) -> Dict:
        """Get current execution statistics."""
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_results) + len(self.failed_jobs) + len(self.cancelled_jobs)
        remaining_count = total_jobs - completed_count
        
        stats = {
            "total": total_jobs,
            "pending": remaining_count,
            "active": 0,  # Always 0 for synchronous execution
            "completed": len(self.completed_results),
            "failed": len(self.failed_jobs),
            "cancelled": len(self.cancelled_jobs),
            "cost_usd": self.cost_tracker.used_usd,
            "cost_limit_usd": self.cost_tracker.limit_usd,
            "is_complete": self.is_complete,
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
            logger.info(f"  Cancelled: {stats['cancelled']}")
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