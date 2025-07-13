"""Concurrent execution management."""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .job import Job
from .job_result import JobResult
from ..providers.batch_request import BatchRequest
from ..providers.provider import Provider
from ..utils.cost import CostTracker


logger = logging.getLogger(__name__)


@dataclass
class ExecutorStats:
    """Statistics for concurrent executor.
    
    Attributes:
        submitted_batches: Total number of batches submitted
        completed_batches: Number of completed batches
        failed_batches: Number of failed batches
        active_batches: Number of currently active batches
        total_jobs: Total number of jobs processed
        completed_jobs: Number of completed jobs
        failed_jobs: Number of failed jobs
    """
    
    submitted_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    active_batches: int = 0
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0


class ConcurrentExecutor:
    """Manages concurrent execution with cost and rate limits.
    
    Uses ThreadPoolExecutor internally to manage parallel batch submissions
    while respecting max_concurrent limit and checking cost limits before
    each submission.
    
    Example:
        >>> executor = ConcurrentExecutor(max_concurrent=5, cost_limit_usd=100.0)
        >>> future = executor.submit_if_allowed(provider, jobs)
        >>> completed = executor.get_completed()
    """
    
    def __init__(
        self,
        max_concurrent: int,
        cost_tracker: Optional[CostTracker] = None
    ):
        """Initialize concurrent executor.
        
        Args:
            max_concurrent: Maximum concurrent batch requests
            cost_tracker: Optional cost tracker for budget management
        """
        self.max_concurrent = max_concurrent
        self.cost_tracker = cost_tracker or CostTracker()
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_futures: Dict[Future, BatchRequest] = {}
        self.completed_batches: List[Tuple[BatchRequest, List[JobResult]]] = []
        
        self._shutdown = False
        self._lock = threading.RLock()
        self._stats = ExecutorStats()
        
        # Track batch to future mapping for status checks
        self._batch_to_future: Dict[str, Future] = {}
    
    def submit_if_allowed(
        self,
        provider: Provider,
        jobs: List[Job]
    ) -> Optional[Future]:
        """Submit jobs if cost limit allows.
        
        Checks cost limit before submission and tracks the batch.
        
        Args:
            provider: Provider to submit to
            jobs: Jobs to submit
            
        Returns:
            Future for the batch submission, or None if not allowed
        """
        if self._shutdown:
            logger.warning("Executor is shutting down, rejecting submission")
            return None
        
        with self._lock:
            # Check if we're at capacity
            if len(self.active_futures) >= self.max_concurrent:
                logger.debug(f"At max concurrent limit ({self.max_concurrent})")
                return None
            
            # Estimate and check cost
            estimated_cost = provider.estimate_cost(jobs)
            
            if not self.cost_tracker.reserve(estimated_cost):
                logger.info(f"Cost limit would be exceeded, estimated: ${estimated_cost:.2f}")
                return None
            
            # Submit the batch
            try:
                # Create a wrapper to pass the future reference
                future = None
                future = self.executor.submit(
                    self._execute_batch_wrapper, provider, jobs, estimated_cost, lambda: future
                )
                
                # Track the future
                self.active_futures[future] = None  # Will be set when batch is created
                
                # Update stats
                self._stats.submitted_batches += 1
                self._stats.active_batches = len(self.active_futures)
                self._stats.total_jobs += len(jobs)
                
                logger.info(f"Submitted batch with {len(jobs)} jobs, estimated cost: ${estimated_cost:.2f}")
                return future
                
            except Exception as e:
                # Release reserved cost on submission failure
                self.cost_tracker.track(0.0, reserved_cost_usd=estimated_cost)
                logger.error(f"Failed to submit batch: {e}")
                raise
    
    def _execute_batch_wrapper(
        self,
        provider: Provider,
        jobs: List[Job],
        reserved_cost: float,
        future_getter
    ) -> Tuple[BatchRequest, List[JobResult]]:
        """Wrapper to pass future reference to execute_batch."""
        return self._execute_batch(provider, jobs, reserved_cost, future_getter())
    
    def _execute_batch(
        self,
        provider: Provider,
        jobs: List[Job],
        reserved_cost: float,
        future: Optional[Future] = None
    ) -> Tuple[BatchRequest, List[JobResult]]:
        """Execute a batch of jobs.
        
        Internal method that runs in a thread pool worker.
        
        Args:
            provider: Provider to use
            jobs: Jobs to execute
            reserved_cost: Cost that was reserved
            future: The Future object for this execution
            
        Returns:
            Tuple of (BatchRequest, results)
        """
        batch_request = None
        
        try:
            # Create the batch
            batch_request = provider.create_batch(jobs)
            
            with self._lock:
                # Store the batch request
                if future and future in self.active_futures:
                    self.active_futures[future] = batch_request
                    self._batch_to_future[batch_request.id] = future
            
            logger.info(f"Created batch {batch_request.id} with {len(jobs)} jobs")
            
            # Poll for completion
            while not batch_request.is_complete and not self._shutdown:
                time.sleep(0.1)  # Poll interval
                batch_request.get_status()
                logger.debug(f"Batch {batch_request.id} status: {batch_request.status}")
            
            if self._shutdown:
                logger.warning(f"Executor shutting down, batch {batch_request.id} may be incomplete")
                return batch_request, []
            
            # Get results
            if batch_request.is_failed:
                logger.error(f"Batch {batch_request.id} failed")
                raise Exception(f"Batch failed: {batch_request.id}")
            
            results = batch_request.get_results()
            
            # Calculate actual cost
            actual_cost = sum(r.cost_usd for r in results)
            
            # Update cost tracker with actual cost
            self.cost_tracker.track(actual_cost, reserved_cost_usd=reserved_cost)
            
            logger.info(
                f"Batch {batch_request.id} completed: "
                f"{len(results)} results, actual cost: ${actual_cost:.2f}"
            )
            
            # Update stats
            with self._lock:
                self._stats.completed_batches += 1
                self._stats.completed_jobs += len([r for r in results if r.is_success])
                self._stats.failed_jobs += len([r for r in results if not r.is_success])
            
            return batch_request, results
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            
            # Release reserved cost on failure
            self.cost_tracker.track(0.0, reserved_cost_usd=reserved_cost)
            
            # Update stats
            with self._lock:
                self._stats.failed_batches += 1
                self._stats.failed_jobs += len(jobs)
            
            raise
    
    def get_completed(self) -> List[Tuple[BatchRequest, List[JobResult]]]:
        """Check for completed batches and return their results.
        
        This method is non-blocking and returns all newly completed batches
        since the last call.
        
        Returns:
            List of (BatchRequest, results) tuples
        """
        completed = []
        
        with self._lock:
            # Check for completed futures
            done_futures = []
            
            for future in list(self.active_futures.keys()):
                if future.done():
                    done_futures.append(future)
            
            # Process completed futures
            for future in done_futures:
                batch_request = self.active_futures.pop(future)
                
                try:
                    batch, results = future.result()
                    completed.append((batch, results))
                    
                    # Clean up mapping
                    if batch and batch.id in self._batch_to_future:
                        del self._batch_to_future[batch.id]
                        
                except Exception as e:
                    logger.error(f"Failed to get batch results: {e}")
                    # Still remove from active to avoid reprocessing
            
            # Update active count
            self._stats.active_batches = len(self.active_futures)
        
        return completed
    
    def wait_for_capacity(self, timeout: Optional[float] = None) -> bool:
        """Wait for capacity to submit more batches.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if capacity is available, False if timeout
        """
        start_time = time.time()
        
        while not self._shutdown:
            with self._lock:
                if len(self.active_futures) < self.max_concurrent:
                    return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
        
        return False
    
    def get_active_count(self) -> int:
        """Get number of active batches."""
        with self._lock:
            return len(self.active_futures)
    
    def get_stats(self) -> ExecutorStats:
        """Get current execution statistics."""
        with self._lock:
            return ExecutorStats(
                submitted_batches=self._stats.submitted_batches,
                completed_batches=self._stats.completed_batches,
                failed_batches=self._stats.failed_batches,
                active_batches=len(self.active_futures),
                total_jobs=self._stats.total_jobs,
                completed_jobs=self._stats.completed_jobs,
                failed_jobs=self._stats.failed_jobs
            )
    
    def shutdown(self, wait: bool = True, cancel_pending: bool = False):
        """Shutdown the executor.
        
        Args:
            wait: If True, wait for active batches to complete
            cancel_pending: If True, cancel pending submissions
        """
        logger.info(f"Shutting down executor (wait={wait}, cancel_pending={cancel_pending})")
        
        self._shutdown = True
        
        if cancel_pending:
            # Cancel futures that haven't started
            with self._lock:
                for future in self.active_futures:
                    if not future.running():
                        future.cancel()
        
        self.executor.shutdown(wait=wait)
        
        if wait:
            # Collect any remaining results
            self.get_completed()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)