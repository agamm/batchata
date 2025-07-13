"""Batch run execution management."""

import json
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .batch_config import BatchConfig
from .concurrent_executor import ConcurrentExecutor
from .job import Job
from .job_result import JobResult
from ..exceptions import StateError
from ..providers.provider_registry import get_provider
from ..utils import CostTracker, StateManager
from ..utils.state import BatchState


logger = logging.getLogger(__name__)


class BatchRun:
    """Manages the execution of a batch job.
    
    Coordinates between providers, executor, state management, and cost tracking
    to run a batch of jobs. Supports graceful shutdown via signal handlers and
    can resume from saved state.
    
    Example:
        >>> config = BatchConfig(...)
        >>> run = BatchRun(config)
        >>> run.start()
        >>> run.wait()
        >>> results = run.results()
    """
    
    def __init__(self, config: BatchConfig):
        """Initialize batch run.
        
        Args:
            config: Batch configuration
        """
        self.config = config
        self.batch_id = f"batch-run-{uuid.uuid4().hex[:8]}"
        
        # Initialize components
        self.cost_tracker = CostTracker(limit_usd=config.cost_limit_usd)
        self.state_manager = StateManager(config.state_file)
        self.executor = ConcurrentExecutor(
            max_concurrent=config.max_concurrent,
            cost_tracker=self.cost_tracker
        )
        
        # State tracking
        self.pending_jobs: List[Job] = []
        self.completed_results: Dict[str, JobResult] = {}  # job_id -> result
        self.failed_jobs: Dict[str, str] = {}  # job_id -> error
        
        # Execution control
        self._started = False
        self._shutdown_event = threading.Event()
        self._execution_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: Optional[datetime] = None
        self._last_progress_time: float = 0
        
        # Results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Try to resume from saved state
        self._resume_from_state()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown(wait_for_active=False)  # Don't wait for active jobs on Ctrl+C
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _resume_from_state(self):
        """Resume from saved state if available."""
        state = self.state_manager.load_state()
        if state is None:
            # No saved state, use jobs from config
            self.pending_jobs = list(self.config.jobs)
            return
        
        logger.info(f"Resuming batch run {state.batch_id} from saved state")
        
        # Restore state
        self.batch_id = state.batch_id
        
        # Restore pending jobs
        self.pending_jobs = []
        for job_data in state.pending_jobs:
            # Simple deserialization - in real implementation would handle response_model
            job = Job(
                id=job_data["id"],
                model=job_data["model"],
                messages=job_data.get("messages"),
                file=Path(job_data["file"]) if job_data.get("file") else None,
                prompt=job_data.get("prompt"),
                temperature=job_data.get("temperature", 0.7),
                max_tokens=job_data.get("max_tokens", 1000),
                enable_citations=job_data.get("enable_citations", False)
            )
            self.pending_jobs.append(job)
        
        # Restore completed results
        for result_data in state.completed_results:
            result = JobResult(
                job_id=result_data["job_id"],
                response=result_data["response"],
                parsed_response=result_data.get("parsed_response"),
                citations=result_data.get("citations"),
                input_tokens=result_data.get("input_tokens", 0),
                output_tokens=result_data.get("output_tokens", 0),
                cost_usd=result_data.get("cost_usd", 0.0),
                error=result_data.get("error")
            )
            self.completed_results[result.job_id] = result
        
        # Restore failed jobs
        for job_data in state.failed_jobs:
            self.failed_jobs[job_data["id"]] = job_data.get("error", "Unknown error")
        
        # Restore cost tracker
        self.cost_tracker.track(state.total_cost_usd)
        
        logger.info(
            f"Resumed with {len(self.pending_jobs)} pending, "
            f"{len(self.completed_results)} completed, "
            f"{len(self.failed_jobs)} failed"
        )
    
    def _save_state(self):
        """Save current state to disk."""
        with self._lock:
            # Serialize jobs
            pending_serialized = []
            for job in self.pending_jobs:
                pending_serialized.append(StateManager.serialize_job(job))
            
            # Note: Active jobs will be handled by the executor's state
            
            # Serialize results
            completed_serialized = []
            for result in self.completed_results.values():
                completed_serialized.append(StateManager.serialize_job_result(result))
            
            # Serialize failed jobs
            failed_serialized = []
            for job_id, error in self.failed_jobs.items():
                failed_serialized.append({"id": job_id, "error": error})
            
            # Create state
            state = BatchState(
                batch_id=self.batch_id,
                created_at=datetime.now().isoformat(),
                pending_jobs=pending_serialized,
                active_batches=[],  # No longer tracking active batches here
                completed_results=completed_serialized,
                failed_jobs=failed_serialized,
                total_cost_usd=self.cost_tracker.used_usd,
                config={
                    "max_concurrent": self.config.max_concurrent,
                    "cost_limit_usd": self.config.cost_limit_usd
                }
            )
            
            self.state_manager.save_state(state)
    
    def start(self):
        """Start the batch execution.
        
        This starts a background thread that processes the jobs.
        """
        if self._started:
            raise RuntimeError("Batch run already started")
        
        self._started = True
        self._start_time = datetime.now()
        self._execution_thread = threading.Thread(target=self._run_execution_loop)
        self._execution_thread.daemon = True
        self._execution_thread.start()
        
        logger.info(f"Started batch run {self.batch_id}")
    
    def _run_execution_loop(self):
        """Main execution loop that runs in background thread."""
        try:
            while not self._shutdown_event.is_set():
                current_time = time.time()
                
                # Check for completed batches
                self._collect_completed_batches()
                
                # Submit new batches if we have capacity and jobs
                self._submit_pending_jobs()
                
                # Call progress callback if set and interval has passed
                if (self.config.progress_callback and 
                    current_time - self._last_progress_time >= self.config.progress_interval):
                    stats = self.status()
                    elapsed_time = (datetime.now() - self._start_time).total_seconds()
                    self.config.progress_callback(stats, elapsed_time)
                    self._last_progress_time = current_time
                
                # Save state periodically
                self._save_state()
                
                # Check if we're done
                if self._is_complete():
                    logger.info("All jobs completed")
                    break
                
                # Small sleep to avoid busy waiting
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Execution loop error: {e}")
            raise
        finally:
            # Final state save
            self._save_state()
    
    def _collect_completed_batches(self):
        """Collect results from completed batches."""
        completed = self.executor.get_completed()
        
        for batch_request, results in completed:
            with self._lock:
                # Process results
                for result in results:
                    if result.is_success:
                        self.completed_results[result.job_id] = result
                        
                        # Save result to file
                        self._save_result_to_file(result)
                    else:
                        self.failed_jobs[result.job_id] = result.error or "Unknown error"
                
                logger.info(
                    f"Batch {batch_request.id} completed: "
                    f"{len([r for r in results if r.is_success])} success, "
                    f"{len([r for r in results if not r.is_success])} failed"
                )
    
    def _submit_pending_jobs(self):
        """Submit pending jobs to executor."""
        if not self.pending_jobs:
            return
        
        # Group jobs by model/provider
        jobs_by_provider: Dict[str, List[Job]] = {}
        
        with self._lock:
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
        
        # Submit batches by provider
        for provider_name, jobs in jobs_by_provider.items():
            if not jobs:
                continue
                
            # Get provider instance
            provider = None
            for job in jobs:
                try:
                    provider = get_provider(job.model)
                    break
                except:
                    pass
            
            if not provider:
                continue
            
            # Try to submit batch
            future = self.executor.submit_if_allowed(provider, jobs)
            
            if future:
                with self._lock:
                    # Remove from pending immediately (they're now being processed)
                    for job in jobs:
                        self.pending_jobs.remove(job)
                    
                    logger.info(f"Submitted {len(jobs)} jobs to {provider_name}")
            else:
                # Could not submit (at capacity or cost limit)
                logger.debug(f"Could not submit {len(jobs)} jobs to {provider_name}")
                break  # Try again later
    
    def _save_result_to_file(self, result: JobResult):
        """Save individual result to file."""
        result_file = self.results_dir / f"{result.job_id}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(StateManager.serialize_job_result(result), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save result for {result.job_id}: {e}")
    
    def _is_complete(self) -> bool:
        """Check if all jobs are complete."""
        with self._lock:
            # Get total job count from config
            total_jobs = len(self.config.jobs)
            completed_count = len(self.completed_results) + len(self.failed_jobs)
            
            return (
                len(self.pending_jobs) == 0 and 
                self.executor.get_active_count() == 0 and
                completed_count == total_jobs
            )
    
    @property
    def is_complete(self) -> bool:
        """Whether all jobs are complete."""
        return self._is_complete()
    
    def wait(self, timeout: Optional[float] = None):
        """Wait for batch to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self._started:
            raise RuntimeError("Batch run not started")
        
        if self._execution_thread:
            self._execution_thread.join(timeout)
    
    def status(self, print_status: bool = False) -> Dict:
        """Get current execution statistics.
        
        Args:
            print_status: If True, print status to console
            
        Returns:
            Dictionary with status information
        """
        with self._lock:
            total_jobs = len(self.config.jobs)
            completed_count = len(self.completed_results) + len(self.failed_jobs)
            active_count = total_jobs - completed_count - len(self.pending_jobs)
            
            stats = {
                "batch_id": self.batch_id,
                "total": total_jobs,
                "pending": len(self.pending_jobs),
                "active": active_count,
                "completed": len(self.completed_results),
                "failed": len(self.failed_jobs),
                "cost_usd": self.cost_tracker.used_usd,
                "cost_limit_usd": self.cost_tracker.limit_usd,
                "is_complete": self._is_complete()
            }
        
        if print_status:
            print(f"\nBatch Run Status ({self.batch_id}):")
            print(f"  Total jobs: {stats['total']}")
            print(f"  Pending: {stats['pending']}")
            print(f"  Active: {stats['active']}")
            print(f"  Completed: {stats['completed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Cost: ${stats['cost_usd']:.2f}")
            if stats['cost_limit_usd']:
                print(f"  Cost limit: ${stats['cost_limit_usd']:.2f}")
            print(f"  Complete: {stats['is_complete']}")
        
        return stats
    
    def results(self) -> Dict[str, JobResult]:
        """Get all completed results.
        
        Returns:
            Dictionary mapping job ID to JobResult
        """
        with self._lock:
            return dict(self.completed_results)
    
    def get_failed_jobs(self) -> Dict[str, str]:
        """Get failed jobs with error messages.
        
        Returns:
            Dictionary mapping job ID to error message
        """
        with self._lock:
            return dict(self.failed_jobs)
    
    def shutdown(self, wait_for_active: bool = True):
        """Gracefully shutdown the batch run.
        
        Args:
            wait_for_active: If True, wait for active batches to complete
        """
        logger.info(f"Shutting down batch run {self.batch_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Shutdown executor
        self.executor.shutdown(wait=wait_for_active)
        
        # Wait for execution thread
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=5.0)
        
        # Final state save
        self._save_state()
        
        logger.info("Batch run shutdown complete")