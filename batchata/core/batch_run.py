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
from ..utils.state import create_temp_state_file
from ..exceptions import TimeoutError


logger = get_logger(__name__)


class BatchRun:
    """Manages the execution of a batch job synchronously.
    
    Processes jobs in batches based on items_per_batch configuration.
    Simpler synchronous execution for clear logging and debugging.
    
    Example:
        ```python
        config = BatchParams(...)
        run = BatchRun(config, jobs)
        run.execute()
        results = run.results()
        ```
    """
    
    def __init__(self, config: BatchParams, jobs: List[Job]):
        """Initialize batch run.
        
        Args:
            config: Batch configuration
            jobs: List of jobs to execute
        """
        self.config = config
        self.jobs = {job.id: job for job in jobs}
        
        # Set logging level based on config
        set_log_level(level=config.verbosity.upper())
        
        # Initialize components
        self.cost_tracker = CostTracker(limit_usd=config.cost_limit_usd)
        
        # Use temp file for state if not provided
        state_file = config.state_file
        if not state_file:
            state_file = create_temp_state_file(config)
            config.reuse_state = False
            logger.info(f"Created temporary state file: {state_file}")
        
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
        self._time_limit_exceeded = False
        self._progress_callback: Optional[Callable[[Dict, float], None]] = None
        self._progress_interval: float = 1.0  # Default to 1 second
        
        # Threading primitives
        self._state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Batch tracking for progress display
        self.batch_tracking: Dict[str, Dict] = {}  # batch_id -> batch_info
        
        # Active batch tracking for cancellation
        self._active_batches: Dict[str, object] = {}  # batch_id -> provider
        self._active_batches_lock = threading.Lock()
        
        # Results directory
        self.results_dir = Path(config.results_dir)
        
        # If not reusing state, clear the results directory
        if not config.reuse_state and self.results_dir.exists():
            import shutil
            shutil.rmtree(self.results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Raw files directory (if enabled)
        self.raw_files_dir = None
        if config.raw_files:
            self.raw_files_dir = self.results_dir / "raw_files"
            self.raw_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to resume from saved state
        self._resume_from_state()
    
    
    def _resume_from_state(self):
        """Resume from saved state if available."""
        # Check if we should reuse state
        if not self.config.reuse_state:
            # Clear any existing state and start fresh
            self.state_manager.clear()
            self.pending_jobs = list(self.jobs.values())
            return
            
        state = self.state_manager.load()
        if state is None:
            # No saved state, use jobs passed to constructor
            self.pending_jobs = list(self.jobs.values())
            return
        
        logger.info("Resuming batch run from saved state")
        
        # Restore pending jobs
        self.pending_jobs = []
        for job_data in state.pending_jobs:
            job = Job.from_dict(job_data)
            # Check if file exists (if job has a file)
            if job.file and not job.file.exists():
                logger.error(f"File not found for job {job.id}: {job.file}")
                logger.error("This may happen if files were in temporary directories that were cleaned up")
                self.failed_jobs[job.id] = f"File not found: {job.file}"
            else:
                self.pending_jobs.append(job)
        
        # Restore completed results from file references
        for result_ref in state.completed_results:
            job_id = result_ref["job_id"]
            file_path = result_ref["file_path"]
            try:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                result = JobResult.from_dict(result_data)
                self.completed_results[job_id] = result
            except Exception as e:
                logger.error(f"Failed to load result for {job_id} from {file_path}: {e}")
                # Move to failed jobs if we can't load the result
                self.failed_jobs[job_id] = f"Failed to load result file: {e}"
        
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
            "completed_results": [
                {"job_id": job_id, "file_path": str(self.results_dir / f"{job_id}.json")}
                for job_id in self.completed_results.keys()
            ],
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
                "raw_files": self.config.raw_files
            }
        }
    
    def execute(self):
        """Execute synchronous batch run and wait for completion."""
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
            
            # Start time limit watchdog if configured
            self._start_time_limit_watchdog()
            
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
            ```python
            run.set_on_progress(
                lambda stats, time, batch_data: print(
                    f"Progress: {stats['completed']}/{stats['total']}, {time:.1f}s"
                )
            )
            ```
        """
        self._progress_callback = callback
        self._progress_interval = interval
        return self
    
    def _start_time_limit_watchdog(self):
        """Start a background thread to check for time limit every second."""
        if not self.config.time_limit_seconds:
            return
        
        def time_limit_watchdog():
            """Check for time limit every second and trigger shutdown if exceeded."""
            while not self._shutdown_event.is_set():
                if self._check_time_limit():
                    logger.warning("Batch execution time limit exceeded")
                    with self._state_lock:
                        self._time_limit_exceeded = True
                    self._shutdown_event.set()
                    break
                time.sleep(1.0)
        
        # Start watchdog as daemon thread
        watchdog_thread = threading.Thread(target=time_limit_watchdog, daemon=True)
        watchdog_thread.start()
        logger.debug(f"Started time limit watchdog thread (time limit: {self.config.time_limit_seconds}s)")
    
    def _check_time_limit(self) -> bool:
        """Check if batch execution has exceeded time limit."""
        if not self.config.time_limit_seconds or not self._start_time:
            return False
        
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return elapsed >= self.config.time_limit_seconds
    
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
                    # Stop if shutdown event detected (includes time limit)
                    if self._shutdown_event.is_set():
                        break
                    future.result()  # Re-raise any exceptions
            except KeyboardInterrupt:
                self._shutdown_event.set()
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
                raise
            finally:
                # Handle time limit or cancellation - mark remaining jobs appropriately
                with self._state_lock:
                    if self._shutdown_event.is_set():
                        # If time limit exceeded, cancel all active batches
                        if self._time_limit_exceeded:
                            self._cancel_all_active_batches()
                        
                        # Mark any unprocessed jobs based on reason for shutdown
                        for _, _, batch_jobs in batches:
                            for job in batch_jobs:
                                # Skip jobs already processed
                                if (job.id in self.completed_results or 
                                    job.id in self.failed_jobs or 
                                    job.id in self.cancelled_jobs):
                                    continue
                                
                                # Mark based on shutdown reason
                                if self._time_limit_exceeded:
                                    self.failed_jobs[job.id] = "Time limit exceeded: batch execution time limit exceeded"
                                else:
                                    self.cancelled_jobs[job.id] = "Cancelled by user"
                                
                                if job in self.pending_jobs:
                                    self.pending_jobs.remove(job)
                        
                        # Save state
                        self.state_manager.save(self)
    
    def _cancel_all_active_batches(self):
        """Cancel all active batches at the provider level."""
        with self._active_batches_lock:
            active_batch_items = list(self._active_batches.items())
            
        logger.info(f"Cancelling {len(active_batch_items)} active batches due to time limit exceeded")
        
        # Cancel outside the lock to avoid blocking
        for batch_id, provider in active_batch_items:
            try:
                provider.cancel_batch(batch_id)
                logger.info(f"Cancelled batch {batch_id} due to time limit exceeded")
            except Exception as e:
                logger.warning(f"Failed to cancel batch {batch_id}: {e}")
        
        # Clear the tracking after cancellation attempts
        with self._active_batches_lock:
            self._active_batches.clear()
    
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
        except TimeoutError:
            # Handle time limit exceeded - mark jobs as failed
            with self._state_lock:
                for job in batch_jobs:
                    self.failed_jobs[job.id] = "Time limit exceeded: batch execution time limit exceeded"
                    if job in self.pending_jobs:
                        self.pending_jobs.remove(job)
                self.state_manager.save(self)
            # Don't re-raise, just return the result
            return
        except KeyboardInterrupt:
            self._shutdown_event.set()
            # Handle user cancellation
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
                estimated_cost = provider.estimate_cost(batch_jobs)
                self.batch_tracking[batch_id] = {
                    'start_time': None,
                    'status': 'pending',
                    'total': len(batch_jobs),
                    'completed': 0,
                    'cost': 0.0,
                    'estimated_cost': estimated_cost,
                    'provider': provider_name,
                    'jobs': batch_jobs
                }
        
        return batches
    
    def _poll_batch_status(self, provider, batch_id: str) -> Tuple[str, Optional[Dict]]:
        """Poll until batch completes."""
        status, error_details = provider.get_batch_status(batch_id)
        logger.info(f"Initial batch status: {status}")
        poll_count = 0
        
        # Use provider-specific polling interval
        provider_polling_interval = provider.get_polling_interval()
        logger.debug(f"Using {provider_polling_interval}s polling interval for {provider.__class__.__name__}")
        
        while status not in ["complete", "failed"]:
            poll_count += 1
            logger.debug(f"Polling attempt {poll_count}, current status: {status}")
            
            # Interruptible wait - will wake up immediately if shutdown event is set (includes time limit)
            if self._shutdown_event.wait(provider_polling_interval):
                # Check if it's time limit exceeded or user cancellation
                with self._state_lock:
                    is_time_limit_exceeded = self._time_limit_exceeded
                
                if is_time_limit_exceeded:
                    logger.info(f"Batch {batch_id} polling interrupted by time limit exceeded")
                    raise TimeoutError("Batch cancelled due to time limit exceeded")
                else:
                    logger.info(f"Batch {batch_id} polling interrupted by user")
                    raise KeyboardInterrupt("Batch cancelled by user")
            
            status, error_details = provider.get_batch_status(batch_id)
            
            if self._progress_callback:
                with self._state_lock:
                    stats = self.status()
                    elapsed_time = (datetime.now() - self._start_time).total_seconds()
                    batch_data = dict(self.batch_tracking)
                self._progress_callback(stats, elapsed_time, batch_data)
            
            elapsed_seconds = poll_count * provider_polling_interval
        
        return status, error_details
    
    
    def _update_batch_results(self, batch_result: Dict):
        """Update state from batch results."""
        results = batch_result.get("results", [])
        failed = batch_result.get("failed", {})
                
        # Update completed results
        for result in results:
            if result.is_success:
                self.completed_results[result.job_id] = result
                self._save_result_to_file(result)
                logger.info(f"✓ Job {result.job_id} completed successfully")
            else:
                error_message = result.error or "Unknown error"
                self.failed_jobs[result.job_id] = error_message
                self._save_result_to_file(result)
                logger.error(f"✗ Job {result.job_id} failed: {result.error}")
            
            # Remove completed/failed job from pending
            self.pending_jobs = [job for job in self.pending_jobs if job.id != result.job_id]
        
        # Update failed jobs
        for job_id, error in failed.items():
            self.failed_jobs[job_id] = error
            # Remove failed job from pending
            self.pending_jobs = [job for job in self.pending_jobs if job.id != job_id]
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
        remaining = self.cost_tracker.remaining()
        remaining_str = f"${remaining:.4f}" if remaining is not None else "unlimited"
        logger.info(f"Total estimated cost: ${estimated_cost:.4f}, remaining budget: {remaining_str}")
        
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
            raw_files_path = str(self.raw_files_dir) if self.raw_files_dir else None
            batch_id, job_mapping = provider.create_batch(batch_jobs, raw_files_path)
            
            # Track active batch for cancellation
            with self._active_batches_lock:
                self._active_batches[batch_id] = provider
            
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
                    'estimated_cost': estimated_cost,
                    'provider': provider.__class__.__name__,
                    'jobs': batch_jobs
                }
            
            # Poll for completion
            logger.info(f"Polling for batch {batch_id} completion...")
            status, error_details = self._poll_batch_status(provider, batch_id)
            
            if status == "failed":
                if error_details:
                    logger.error(f"Batch {batch_id} failed with details: {error_details}")
                else:
                    logger.error(f"Batch {batch_id} failed")
                
                # Save error details if configured
                if self.raw_files_dir and error_details:
                    self._save_batch_error_details(batch_id, error_details)
                
                # Continue to get individual results - some jobs might have succeeded
            
            # Get results
            logger.info(f"Getting results for batch {batch_id}")
            raw_files_path = str(self.raw_files_dir) if self.raw_files_dir else None
            results = provider.get_batch_results(batch_id, job_mapping, raw_files_path)
            
            # Calculate actual cost and adjust reservation
            actual_cost = sum(r.cost_usd for r in results)
            self.cost_tracker.adjust_reserved_cost(estimated_cost, actual_cost)
            
            # Update batch tracking for completion
            success_count = len([r for r in results if r.is_success])
            failed_count = len([r for r in results if not r.is_success])
            batch_status = 'complete' if failed_count == 0 else 'failed'
            
            with self._state_lock:
                if batch_id in self.batch_tracking:
                    self.batch_tracking[batch_id]['status'] = batch_status
                    self.batch_tracking[batch_id]['completed'] = success_count
                    self.batch_tracking[batch_id]['failed'] = failed_count
                    self.batch_tracking[batch_id]['cost'] = actual_cost
                    self.batch_tracking[batch_id]['completion_time'] = datetime.now()
                    if batch_status == 'failed' and failed_count > 0:
                        # Use the first job's error as the batch error summary
                        first_error = next((r.error for r in results if not r.is_success), 'Some jobs failed')
                        self.batch_tracking[batch_id]['error'] = first_error
            
            # Remove from active batches tracking
            with self._active_batches_lock:
                self._active_batches.pop(batch_id, None)
            
            status_symbol = "✓" if batch_status == 'complete' else "⚠"
            logger.info(
                f"{status_symbol} Batch {batch_id} completed: "
                f"{success_count} success, "
                f"{failed_count} failed, "
                f"cost: ${actual_cost:.6f}"
            )
            
            return {"results": results, "failed": {}, "cost": actual_cost, "jobs_to_remove": list(batch_jobs)}
            
        except TimeoutError:
            logger.info(f"Time limit exceeded for batch{f' {batch_id}' if batch_id else ''}")
            if batch_id:
                # Update batch tracking for time limit exceeded
                with self._state_lock:
                    if batch_id in self.batch_tracking:
                        self.batch_tracking[batch_id]['status'] = 'failed'
                        self.batch_tracking[batch_id]['error'] = 'Time limit exceeded: batch execution time limit exceeded'
                        self.batch_tracking[batch_id]['completion_time'] = datetime.now()
                # NOTE: Don't remove from _active_batches - let centralized cancellation handle it
            # Release the reservation since batch exceeded time limit
            self.cost_tracker.adjust_reserved_cost(estimated_cost, 0.0)
            # Re-raise to be handled by wrapper
            raise
            
        except KeyboardInterrupt:
            logger.warning(f"\nCancelling batch{f' {batch_id}' if batch_id else ''}...")
            if batch_id:
                # Update batch tracking for cancellation
                with self._state_lock:
                    if batch_id in self.batch_tracking:
                        self.batch_tracking[batch_id]['status'] = 'cancelled'
                        self.batch_tracking[batch_id]['error'] = 'Cancelled by user'
                        self.batch_tracking[batch_id]['completion_time'] = datetime.now()
                # Remove from active batches tracking
                with self._active_batches_lock:
                    self._active_batches.pop(batch_id, None)
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
                        self.batch_tracking[batch_id]['completion_time'] = datetime.now()
                # Remove from active batches tracking
                with self._active_batches_lock:
                    self._active_batches.pop(batch_id, None)
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
        """Save batch error details to debug files directory."""
        try:
            error_file = self.raw_files_dir / f"batch_{batch_id}_error.json"
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
    
    def results(self) -> Dict[str, List[JobResult]]:
        """Get all results organized by status.
        
        Returns:
            {
                "completed": [JobResult],
                "failed": [JobResult],
                "cancelled": [JobResult]
            }
        """
        return {
            "completed": list(self.completed_results.values()),
            "failed": self._create_failed_results(),
            "cancelled": self._create_cancelled_results()
        }
    
    def get_failed_jobs(self) -> Dict[str, str]:
        """Get failed jobs with error messages.
        
        Note: This method is deprecated. Use results()['failed'] instead.
        """
        return dict(self.failed_jobs)
    
    def _create_failed_results(self) -> List[JobResult]:
        """Convert failed jobs to JobResult objects."""
        failed_results = []
        for job_id, error_msg in self.failed_jobs.items():
            failed_results.append(JobResult(
                job_id=job_id,
                raw_response=None,
                parsed_response=None,
                error=error_msg,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0
            ))
        return failed_results
    
    def _create_cancelled_results(self) -> List[JobResult]:
        """Convert cancelled jobs to JobResult objects."""
        cancelled_results = []
        for job_id, reason in self.cancelled_jobs.items():
            cancelled_results.append(JobResult(
                job_id=job_id,
                raw_response=None,
                parsed_response=None,
                error=reason,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0
            ))
        return cancelled_results
    
    def shutdown(self):
        """Shutdown (no-op for synchronous execution)."""
        pass
    
    def dry_run(self) -> 'BatchRun':
        """Perform a dry run - show cost estimation and job details without executing.
        
        Returns:
            Self for chaining (doesn't actually execute jobs)
        """
        logger.info("=== DRY RUN MODE ===")
        logger.info("This will show cost estimates without executing jobs")
        
        # Load existing state if reuse_state=True
        if self.config.reuse_state:
            self.state_manager.load_state(self)
        
        # Filter out completed jobs from previous runs
        self.pending_jobs = [job for job in self.jobs.values() if job.id not in self.completed_results]
        
        if not self.pending_jobs:
            logger.info("No pending jobs to analyze (all jobs already completed)")
            return self
        
        logger.info(f"Analyzing {len(self.pending_jobs)} pending jobs...")
        
        # Group jobs by provider and analyze costs
        provider_groups = self._group_jobs_by_provider()
        total_estimated_cost = 0.0
        
        logger.info(f"\nJob breakdown:")
        for provider_name, jobs in provider_groups.items():
            provider = get_provider(jobs[0].model)
            logger.info(f"\n{provider_name} ({len(jobs)} jobs):")
            
            job_batches = [jobs[i:i + self.config.items_per_batch] 
                          for i in range(0, len(jobs), self.config.items_per_batch)]
            
            for batch_idx, batch_jobs in enumerate(job_batches, 1):
                estimated_cost = provider.estimate_cost(batch_jobs)
                total_estimated_cost += estimated_cost
                
                logger.info(f"  Batch {batch_idx}: {len(batch_jobs)} jobs, estimated cost: ${estimated_cost:.4f}")
                for job in batch_jobs:
                    if job.file:
                        logger.info(f"    - {job.id}: {job.file.name} (citations: {job.enable_citations})")
                    else:
                        logger.info(f"    - {job.id}: direct messages (citations: {job.enable_citations})")
        
        # Show cost summary
        logger.info(f"\n=== COST SUMMARY ===")
        logger.info(f"Total estimated cost: ${total_estimated_cost:.4f}")
        
        if self.config.cost_limit_usd:
            logger.info(f"Cost limit: ${self.config.cost_limit_usd:.2f}")
            if total_estimated_cost > self.config.cost_limit_usd:
                excess = total_estimated_cost - self.config.cost_limit_usd
                logger.warning(f"⚠️ Estimated cost exceeds limit by ${excess:.4f}")
            else:
                remaining = self.config.cost_limit_usd - total_estimated_cost
                logger.info(f"✅ Within cost limit (${remaining:.4f} remaining)")
        else:
            logger.info("No cost limit set")
        
        # Show execution plan
        logger.info(f"\n=== EXECUTION PLAN ===")
        total_batches = sum(
            len(jobs) // self.config.items_per_batch + (1 if len(jobs) % self.config.items_per_batch else 0)
            for jobs in provider_groups.values()
        )
        logger.info(f"Total batches to process: {total_batches}")
        logger.info(f"Max parallel batches: {self.config.max_parallel_batches}")
        logger.info(f"Items per batch: {self.config.items_per_batch}")
        logger.info(f"Results directory: {self.config.results_dir}")
        
        logger.info("\n=== DRY RUN COMPLETE ===")
        logger.info("To execute for real, call run() without dry_run=True")
        
        return self
    
