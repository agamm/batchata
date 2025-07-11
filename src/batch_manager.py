"""
BatchManager Module (Refactored)

Manages large-scale batch processing by splitting inputs into jobs,
running them in parallel, with state persistence and cost management.
"""

import json
import os
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import threading
import logging

from pydantic import BaseModel

from .core import batch, MessageConversations, FileInputs
from .batch_job import BatchJob
from .utils import check_flat_model_for_citation_mapping, run_jobs_with_conditional_parallel


# Type definitions
BatchInputData = Union[MessageConversations, FileInputs]  # Either message conversations OR file inputs


# Custom exceptions
class BatchManagerError(Exception):
    """Base exception for BatchManager errors"""
    pass


class StateFileError(BatchManagerError):
    """Error related to state file operations"""
    pass


class InvalidStateError(StateFileError):
    """State file is invalid or corrupted"""
    pass


class CostLimitExceededError(BatchManagerError):
    """Processing stopped due to cost limit"""
    pass


class JobProcessingError(BatchManagerError):
    """Error during job processing"""
    pass


class ProgressMonitor:
    """Real-time progress monitoring for BatchManager"""
    
    def __init__(self, batch_manager: 'BatchManager'):
        self.batch_manager = batch_manager
        self.running = False
        self.thread = None
        self.start_time = time.time()
        self.is_retry = False
        
    def start(self, is_retry: bool = False):
        """Start the progress monitor"""
        self.running = True
        self.is_retry = is_retry
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the progress monitor"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        # Clear the line
        print("\r" + " " * 100 + "\r", end="", flush=True)
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self._update_progress()
                time.sleep(3)  # Update every 3 seconds
            except Exception as e:
                # Don't let progress monitoring crash the main process
                logging.debug(f"Progress monitor error: {e}")
                
    def _update_progress(self):
        """Update and display current progress"""
        stats = self.batch_manager.stats
        elapsed = time.time() - self.start_time
        
            
        # Simple progress display
        pct = (stats['completed_items'] / stats['total_items'] * 100) if stats['total_items'] > 0 else 0
        status_line = f"⚡ Processing {stats['completed_items']}/{stats['total_items']} ({pct:.1f}%) | Cost: ${stats['total_cost']:.3f} | Time: {self._format_duration(elapsed)}"
        
        if stats['failed_items'] > 0:
            status_line += f" | Failed: {stats['failed_items']}"
            
        print(f"\r{status_line}", end="", flush=True)
        
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable form"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    


class ItemStatus(str, Enum):
    """Status of individual items within a job"""
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Status of a job (group of items)"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobItem:
    """Individual item within a job"""
    original_index: int
    content: Union[List[dict], str]  # messages or file path/bytes representation
    status: ItemStatus = ItemStatus.PENDING
    error: Optional[str] = None
    cost: Optional[float] = None
    completed_at: Optional[str] = None


@dataclass
class Job:
    """A job containing multiple items to process as one batch"""
    index: int
    batch_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    items: List[JobItem] = None
    job_cost: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []


@dataclass
class ManagerState:
    """Complete state of the BatchManager"""
    manager_id: str
    created_at: str
    model: str
    items_per_job: int
    max_cost: Optional[float]
    save_results_dir: Optional[str]
    batch_kwargs: Dict[str, Any]
    jobs: List[Job]
    total_cost: float = 0.0
    last_updated: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagerState':
        """Create from dictionary"""
        # Convert strings back to enums
        jobs = []
        for job_data in data["jobs"]:
            items = [
                JobItem(**{**item, "status": ItemStatus(item["status"])})
                for item in job_data["items"]
            ]
            jobs.append(Job(**{**job_data, "status": JobStatus(job_data["status"]), "items": items}))
        
        return cls(
            **{**data, "jobs": jobs}
        )
    


class BatchManager:
    """
    Manages large-scale batch processing with automatic job splitting,
    parallel execution, state persistence, and cost management.
    """
    
    def __init__(
        self,
        messages: Optional[MessageConversations] = None,
        files: Optional[FileInputs] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        items_per_job: int = 10,
        max_parallel_jobs: int = 1,
        max_cost: Optional[float] = None,
        max_wait_time: int = 360,
        state_path: Optional[str] = None,
        save_results_dir: Optional[str] = None,
        **batch_kwargs
    ):
        """
        Initialize BatchManager.
        
        Args:
            messages: List of message conversations (XOR with files)
            files: List of file paths or bytes (XOR with messages)
            prompt: Required when using files
            model: Model name (required)
            items_per_job: Items per job (default: 10)
            max_parallel_jobs: Maximum concurrent jobs (default: 1)
            max_cost: Maximum total cost before stopping (default: None)
            max_wait_time: Maximum time to wait for batch completion in seconds (default: 360 - 6 minutes)
            state_path: Path to state file for persistence (default: None)
            save_results_dir: Directory to save results (default: None)
            **batch_kwargs: Additional arguments passed to batch()
        """
        # Validate inputs
        if model is None:
            raise BatchManagerError("model is required")
        
        if messages is not None and files is not None:
            raise BatchManagerError("Cannot provide both messages and files. Use either messages or files, not both.")
        
        if messages is None and files is None:
            raise BatchManagerError("Must provide either messages or files.")
        
        if files is not None and prompt is None:
            raise BatchManagerError("prompt is required when using files")
        
        if max_cost is not None and max_cost <= 0:
            raise BatchManagerError("max_cost must be positive")
        
        # Store configuration
        self.model = model
        self.items_per_job = items_per_job
        self.max_parallel_jobs = max_parallel_jobs
        self.max_cost = max_cost
        self.max_wait_time = max_wait_time
        self.state_path = state_path
        self.save_results_dir = save_results_dir
        
        # Store response_model separately since it can't be serialized
        self._response_model = batch_kwargs.pop('response_model', None)
        self.batch_kwargs = batch_kwargs
        
        # Validate flat model requirement for citation mapping
        check_flat_model_for_citation_mapping(self._response_model, batch_kwargs.get('enable_citations', False))
        
        # Add prompt to batch_kwargs if using files
        if files is not None:
            self.batch_kwargs["prompt"] = prompt
        
        # Initialize input data attributes
        self._original_data: Optional[BatchInputData] = None
        
        # Initialize or load state
        if state_path and os.path.exists(state_path):
            self._load_state()
            self._validate_resumed_state(messages, files)
        else:
            self._initialize_state(messages, files)
    
    def _initialize_state(self, messages: Optional[List[List[dict]]], files: Optional[List]) -> None:
        """Initialize new state from input data."""
        input_data = messages if messages is not None else files
        is_messages = messages is not None
        
        # Create jobs
        jobs = []
        for job_start in range(0, len(input_data), self.items_per_job):
            job_end = min(job_start + self.items_per_job, len(input_data))
            
            items = []
            for idx in range(job_start, job_end):
                # Store content differently for messages vs files
                if is_messages:
                    content = input_data[idx]
                else:
                    # For files, store path or bytes representation with type info
                    file_item = input_data[idx]
                    if isinstance(file_item, bytes):
                        content = f"<bytes length={len(file_item)} hash={hash(file_item)}>"
                    else:
                        content = str(file_item)
                
                items.append(JobItem(
                    original_index=idx,
                    content=content
                ))
            
            jobs.append(Job(
                index=len(jobs),
                items=items
            ))
        
        # Initialize state
        self.state = ManagerState(
            manager_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            model=self.model,
            items_per_job=self.items_per_job,
            max_cost=self.max_cost,
            save_results_dir=self.save_results_dir,
            batch_kwargs=self.batch_kwargs,
            jobs=jobs,
            total_cost=0.0,  # Explicitly initialize
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        # Store original input data for processing
        self._original_data = input_data
        
        # Initialize progress monitor
        self._progress_monitor = None
        
        # Save initial state
        if self.state_path:
            self._save_state()
    
    def _load_state(self) -> None:
        """Load existing state from file."""
        if not self.state_path:
            raise StateFileError("No state path configured")
        
        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidStateError(f"State file contains invalid JSON: {e}")
        except IOError as e:
            raise StateFileError(f"Error reading state file: {e}")
        
        # Validate required fields
        required_fields = ["manager_id", "created_at", "model", "items_per_job", "jobs"]
        for field in required_fields:
            if field not in data:
                raise InvalidStateError(f"State file missing required field: {field}")
        
        self.state = ManagerState.from_dict(data)
        
        # Update configuration from state
        self.model = self.state.model
        self.items_per_job = self.state.items_per_job
        self.max_cost = self.state.max_cost
        self.save_results_dir = self.state.save_results_dir
        self.batch_kwargs = self.state.batch_kwargs
        
        # response_model is not restored from state - it must be passed again when resuming
        self._response_model = None
        
        # Validate and fix cost consistency
        self._validate_cost_consistency()
        
        # Initialize progress monitor
        self._progress_monitor = None
    
    def _validate_resumed_state(self, messages: Optional[List[List[dict]]], files: Optional[List]) -> None:
        """Validate and reconstruct data when resuming from state."""
        # Note: messages parameter is ignored when resuming from state, only files are needed for file-based batches
        
        # Determine input type from first job's first item
        if not self.state.jobs or not self.state.jobs[0].items:
            raise InvalidStateError("State file contains no jobs or items")
        
        first_content = self.state.jobs[0].items[0].content
        is_messages = isinstance(first_content, list)
        
        
        # Reconstruct original data from state
        if is_messages:
            messages_data: List[List[dict]] = []
            for job in self.state.jobs:
                for item in job.items:
                    if isinstance(item.content, list):
                        messages_data.append(item.content)
                    else:
                        raise InvalidStateError(f"Expected message content to be list, got {type(item.content)}")
            self._original_data = messages_data
        else:
            # For files, we need the actual file data passed in
            if files is None:
                raise InvalidStateError("Files must be provided when resuming a file-based batch")
            self._original_data = files
    
    def _validate_cost_consistency(self) -> None:
        """Basic cost validation."""
        # Simple validation - just check for negative costs
        for job in self.state.jobs:
            for item in job.items:
                if item.cost is not None and item.cost < 0:
                    item.cost = 0.0
    
    # Class-level lock for state file operations
    _state_locks = {}
    _state_locks_lock = threading.Lock()
    
    @classmethod
    def _get_state_lock(cls, state_path: str) -> threading.Lock:
        """Get or create a lock for a specific state file."""
        with cls._state_locks_lock:
            if state_path not in cls._state_locks:
                cls._state_locks[state_path] = threading.Lock()
            return cls._state_locks[state_path]
    
    
    def _save_state(self) -> None:
        """Save current state to file with proper locking."""
        if not self.state_path:
            return
        
        lock = self._get_state_lock(self.state_path)
        with lock:
            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            
            # Create directory if needed
            state_dir = os.path.dirname(self.state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            
            # Write atomically
            temp_path = self.state_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            
            os.rename(temp_path, self.state_path)
    
    def _update_item_status(
        self, job_idx: int, item_idx: int, status: ItemStatus,
        error: Optional[str] = None, cost: Optional[float] = None
    ) -> None:
        """Update status of a specific item and save state."""
        item = self.state.jobs[job_idx].items[item_idx]
        item.status = status
        
        if error is not None:
            item.error = error
        
        if cost is not None:
            # Validate cost is non-negative
            if cost < 0:
                raise JobProcessingError(f"Invalid negative cost: {cost}")
            
            item.cost = cost
            self.state.jobs[job_idx].job_cost += cost
            self.state.total_cost += cost
        
        if status in [ItemStatus.SUCCEEDED, ItemStatus.FAILED]:
            item.completed_at = datetime.now(timezone.utc).isoformat()
        
        self._save_state()
    
    def _process_job_with_error_handling(self, job: Job) -> None:
        """
        Process a job with error handling for the parallel utility.
        """
        try:
            job_idx, error = self._process_job(job)
            if error:
                # Could log error here if needed
                pass
        except Exception as e:
            # Handle unexpected errors silently
            pass
    
    def _process_job(self, job: Job) -> Tuple[int, Optional[str]]:
        """
        Process a single job. Returns (job_index, error_message).
        
        This method is designed to be called in a thread.
        """
        job_idx = job.index
        
        try:
            # Update job status
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_state()
            
            # Prepare input data
            input_items = []
            pending_indices = []
            
            for i, item in enumerate(job.items):
                if item.status == ItemStatus.PENDING:
                    if self._original_data is None:
                        raise JobProcessingError("Original input data not available - cannot process job")
                    
                    try:
                        input_items.append(self._original_data[item.original_index])
                    except (IndexError, KeyError):
                        raise JobProcessingError(f"Invalid original_index {item.original_index} in job {job_idx}")
                    
                    pending_indices.append(i)
            
            if not input_items:
                # No items to process
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                self._save_state()
                return job_idx, None
            
            # Execute batch job - check if input is messages (list of dicts) or files (strings)
            if input_items and isinstance(input_items[0], list):
                # Messages format
                batch_job = batch(
                    model=self.model,
                    messages=input_items,
                    response_model=self._response_model,
                    max_tokens=self.batch_kwargs.get("max_tokens", 1024),
                    temperature=self.batch_kwargs.get("temperature", 0.0),
                    verbose=self.batch_kwargs.get("verbose", False),
                    enable_citations=self.batch_kwargs.get("enable_citations", False),
                    raw_results_dir=os.path.join(self.save_results_dir, "raw") if self.save_results_dir else None
                )
            else:
                # Files format
                batch_job = batch(
                    model=self.model,
                    files=input_items,
                    prompt=self.batch_kwargs.get("prompt", "Process this document"),
                    response_model=self._response_model,
                    max_tokens=self.batch_kwargs.get("max_tokens", 1024),
                    temperature=self.batch_kwargs.get("temperature", 0.0),
                    verbose=self.batch_kwargs.get("verbose", False),
                    enable_citations=self.batch_kwargs.get("enable_citations", False),
                    raw_results_dir=os.path.join(self.save_results_dir, "raw") if self.save_results_dir else None
                )
            
            try:
                # Store batch ID
                job.batch_id = batch_job._batch_id
                self._save_state()
                
                # Wait for completion with progress indicator
                start_time = time.time()
                poll_interval = 2    # Check every 2 seconds
                
                while not batch_job.is_complete():
                    elapsed = int(time.time() - start_time)
                    if elapsed >= self.max_wait_time:
                        raise JobProcessingError(f"Batch job {job.batch_id} did not complete within {self.max_wait_time} seconds")
                    
                    # Just wait, progress is handled by ProgressMonitor
                    time.sleep(poll_interval)
                
                # Job completion is now handled separately - no carriage return output here
                elapsed = int(time.time() - start_time)
                
                # Get results (now in unified format)
                results = batch_job.results()
                
                # Get cost information
                stats = batch_job.stats()
                total_cost = stats.get("total_cost", 0.0)
                
                
            except Exception as e:
                # Clear progress line before showing error
                print(f"\rBatch processing failed: {e}" + " " * 20)
                raise
            
            # Validate cost is reasonable
            if total_cost < 0:
                raise JobProcessingError(f"Invalid negative total cost from batch job: {total_cost}")
            
            cost_per_item = total_cost / len(pending_indices) if pending_indices else 0.0
            
            # Update item statuses
            if len(results) == 0 and len(pending_indices) > 0:
                # Batch completed but no results - mark all as failed
                for item_idx in pending_indices:
                    self._update_item_status(job_idx, item_idx, ItemStatus.FAILED, 
                                           error="Batch completed but no results returned - likely API processing error", cost=0.0)
            else:
                # Normal case - update based on results
                for i, item_idx in enumerate(pending_indices):
                    if i < len(results):
                        self._update_item_status(job_idx, item_idx, ItemStatus.SUCCEEDED, cost=cost_per_item)
                    else:
                        self._update_item_status(job_idx, item_idx, ItemStatus.FAILED, 
                                               error="No result returned", cost=0.0)
            
            # Save results if requested
            if self.save_results_dir and results:
                self._save_job_results(job_idx, results)
            
            # Mark job as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_state()
            
            return job_idx, None
            
        except Exception as e:
            # Job failed - mark as failed and let retry mechanism handle it
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_state()
            print(f"⚠️  Job {job_idx} failed: {e}")
            
            return job_idx, str(e)
    
    def _save_job_results(self, job_idx: int, results: List[Any]) -> None:
        """Save job results to disk."""
        if not self.save_results_dir:
            return
        
        processed_dir = os.path.join(self.save_results_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Convert results to serializable format
        serialized_results = []
        for entry in results:
            serialized_entry = {
                "result": entry["result"].model_dump() if hasattr(entry["result"], 'model_dump') else entry["result"],
                "citations": self._serialize_citations(entry["citations"])
            }
            serialized_results.append(serialized_entry)
        
        # Save to file
        results_file = os.path.join(processed_dir, f"job_{job_idx}_results.json")
        with open(results_file, 'w') as f:
            json.dump(serialized_results, f, indent=2)
    
    def _reset_interrupted_jobs(self) -> None:
        """Reset jobs that were left in PROCESSING state from interrupted runs."""
        reset_count = 0
        for job in self.state.jobs:
            if job.status == JobStatus.PROCESSING:
                # Check if job has been processing for a long time (likely interrupted)
                if job.started_at:
                    try:
                        started = datetime.fromisoformat(job.started_at.replace('Z', '+00:00'))
                        elapsed = datetime.now(timezone.utc) - started
                        # If processing for more than max_wait_time + 60 seconds, consider it interrupted
                        if elapsed.total_seconds() > (self.max_wait_time + 60):
                            job.status = JobStatus.PENDING
                            job.started_at = None
                            job.batch_id = None
                            reset_count += 1
                    except (ValueError, TypeError):
                        # If timestamp parsing fails, reset anyway
                        job.status = JobStatus.PENDING
                        job.started_at = None
                        job.batch_id = None
                        reset_count += 1
                else:
                    # No started_at timestamp, definitely interrupted
                    job.status = JobStatus.PENDING
                    reset_count += 1
        
        if reset_count > 0:
            print(f"🔄 Reset {reset_count} interrupted job(s) to pending state")
            self._save_state()
    
    def _serialize_citations(self, citations: Any) -> Any:
        """Helper to serialize citations to JSON-safe format."""
        if citations is None:
            return None
        elif isinstance(citations, dict):
            # Field citations - convert Citation objects to dicts
            serialized = {}
            for field_name, field_citations in citations.items():
                if isinstance(field_citations, list):
                    serialized[field_name] = [
                        cit.model_dump() if hasattr(cit, 'model_dump') else cit
                        for cit in field_citations
                    ]
                else:
                    serialized[field_name] = field_citations
            return serialized
        elif isinstance(citations, list):
            # List of Citation objects
            return [
                cit.model_dump() if hasattr(cit, 'model_dump') else cit
                for cit in citations
            ]
        else:
            # Single citation or other format
            if hasattr(citations, 'model_dump'):
                return citations.model_dump()
            else:
                return citations
    
    def run(self, print_progress: bool = True) -> Dict[str, Any]:
        """
        Run batch processing with automatic resume capability.
        
        Returns processing summary instead of all results to avoid memory issues.
        """
        # Reset any jobs that were left in PROCESSING state (e.g., from interrupted runs)
        self._reset_interrupted_jobs()
        
        # Get jobs that need processing
        pending_jobs = [job for job in self.state.jobs if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]]
        
        if not pending_jobs:
            if print_progress:
                print("✅ All jobs already completed")
            return self._get_summary()
        
        if print_progress:
            print(f"🚀 Starting batch processing: {len(pending_jobs)} jobs remaining")
            print(f"   Max parallel jobs: {self.max_parallel_jobs}")
            if self.max_cost:
                print(f"   Cost limit: ${self.max_cost:.2f}")
            print(f"   Total items: {self.stats['total_items']}")
            print()
            
            # Start progress monitor
            self._progress_monitor = ProgressMonitor(self)
            self._progress_monitor.start()
        
        try:
            # Define simple condition function for cost limit checking
            # Note: This will be called under lock by the parallel utility
            def cost_limit_exceeded() -> bool:
                return self.max_cost is not None and self.state.total_cost >= self.max_cost
            
            # Get the shared lock for atomic cost operations
            shared_lock = self._get_state_lock(self.state_path) if self.state_path else None
            
            # Process jobs with conditional parallel execution using shared lock
            run_jobs_with_conditional_parallel(
                max_parallel=self.max_parallel_jobs,
                condition_fn=cost_limit_exceeded,
                jobs=pending_jobs,
                job_processor_fn=self._process_job_with_error_handling,
                shared_lock=shared_lock
            )
        
        finally:
            # Always stop progress monitor
            if print_progress and self._progress_monitor:
                self._progress_monitor.stop()
                print()  # New line after progress
        
        if print_progress:
            print("✅ Batch processing completed")
            self._print_final_stats()
        
        return self._get_summary()
    
    def retry_failed(self) -> Dict[str, Any]:
        """Retry all failed items and pending items in failed jobs by creating new jobs."""
        # Collect all failed items and pending items in failed jobs
        retry_items = []
        failed_count = 0
        pending_count = 0
        
        for job in self.state.jobs:
            for item in job.items:
                if item.status == ItemStatus.FAILED:
                    # Reset failed item for retry
                    item.status = ItemStatus.PENDING
                    item.error = None
                    item.cost = None
                    item.completed_at = None
                    retry_items.append(item)
                    failed_count += 1
                elif item.status == ItemStatus.PENDING and job.status == JobStatus.FAILED:
                    # Include pending items from failed jobs
                    retry_items.append(item)
                    pending_count += 1
        
        if not retry_items:
            print("✅ No items need retrying")
            return {"retry_count": 0}
        
        print(f"🔄 Preparing retry for {len(retry_items)} items:")
        print(f"   {failed_count} previously failed items")
        print(f"   {pending_count} pending items from failed jobs")
        
        # Create new jobs for retry items
        new_jobs = []
        for i in range(0, len(retry_items), self.items_per_job):
            items = retry_items[i:i + self.items_per_job]
            new_jobs.append(Job(
                index=len(self.state.jobs) + len(new_jobs),
                items=items
            ))
        
        # Add new jobs to state
        self.state.jobs.extend(new_jobs)
        self._save_state()
        
        print(f"   Created {len(new_jobs)} retry jobs")
        print()
        
        # Process retry jobs with retry indicator
        if self._progress_monitor:
            summary = self.run(print_progress=True)
            # Override the progress monitor to show it's a retry
            self._progress_monitor.is_retry = True
        else:
            summary = self.run(print_progress=True)
            
        summary["retry_count"] = len(retry_items)
        return summary
    
    def get_results_from_disk(self) -> List[Any]:
        """Load results from saved files."""
        if not self.save_results_dir:
            raise BatchManagerError("No save_results_dir configured")
        
        processed_dir = os.path.join(self.save_results_dir, "processed")
        if not os.path.exists(processed_dir):
            return []
        
        # Load all result files
        all_results = {}
        for filename in os.listdir(processed_dir):
            if filename.startswith("job_") and filename.endswith("_results.json"):
                job_idx = int(filename.split("_")[1])
                with open(os.path.join(processed_dir, filename), 'r') as f:
                    data = json.load(f)
                    
                # Map results back to original indices
                job = self.state.jobs[job_idx]
                succeeded_items = [item for item in job.items if item.status == ItemStatus.SUCCEEDED]
                
                for i, item in enumerate(succeeded_items):
                    if i < len(data):
                        # Handle unified format: data is now a list of {result: ..., citations: ...}
                        all_results[item.original_index] = data[i]
        
        # Return in original order
        max_idx = max(all_results.keys()) if all_results else -1
        return [all_results.get(i) for i in range(max_idx + 1)]
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        stats = self.stats
        return {
            "total_items": stats["total_items"],
            "completed_items": stats["completed_items"],
            "failed_items": stats["failed_items"],
            "total_cost": stats["total_cost"],
            "jobs_completed": stats["jobs_completed"],
            "cost_limit_reached": stats["cost_limit_reached"]
        }
    
    
    def _print_final_stats(self) -> None:
        """Print final statistics."""
        stats = self.stats
        success_rate = (stats['completed_items'] / stats['total_items'] * 100) if stats['total_items'] > 0 else 0
        
        print(f"📊 Final Statistics:")
        print(f"   Items: {stats['completed_items']}/{stats['total_items']} completed ({success_rate:.1f}%)")
        if stats['failed_items'] > 0:
            print(f"   Failed: {stats['failed_items']} items")
        print(f"   Jobs: {stats['jobs_completed']} completed")
        print(f"   Total cost: ${stats['total_cost']:.4f}")
        
        if self.max_cost:
            cost_pct = (stats['total_cost'] / self.max_cost) * 100
            print(f"   Budget used: {cost_pct:.1f}%")
            if stats['cost_limit_reached']:
                print(f"   ⚠️  Cost limit was reached")
        
        # Show any jobs that need retry
        retry_needed = sum(1 for job in self.state.jobs 
                          for item in job.items 
                          if item.status == ItemStatus.FAILED or 
                             (item.status == ItemStatus.PENDING and job.status == JobStatus.FAILED))
        if retry_needed > 0:
            print(f"   💡 {retry_needed} items available for retry")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        total_items = sum(len(job.items) for job in self.state.jobs)
        completed_items = sum(
            1 for job in self.state.jobs 
            for item in job.items 
            if item.status == ItemStatus.SUCCEEDED
        )
        failed_items = sum(
            1 for job in self.state.jobs 
            for item in job.items 
            if item.status == ItemStatus.FAILED
        )
        jobs_completed = sum(
            1 for job in self.state.jobs 
            if job.status == JobStatus.COMPLETED
        )
        
        return {
            "total_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "total_cost": self.state.total_cost,
            "jobs_completed": jobs_completed,
            "cost_limit_reached": self.max_cost is not None and self.state.total_cost >= self.max_cost
        }