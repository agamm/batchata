"""State persistence utilities."""

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.job import Job
from ..core.job_result import JobResult
from ..exceptions import StateError


@dataclass
class BatchState:
    """Represents the state of a batch run.
    
    Attributes:
        batch_id: Unique identifier for this batch run
        created_at: When the batch was created
        pending_jobs: Jobs that haven't been submitted yet
        active_batches: Currently running batch IDs
        completed_results: Results from completed jobs
        failed_jobs: Jobs that failed with errors
        total_cost_usd: Total cost incurred so far
        config: Original batch configuration
    """
    
    batch_id: str
    created_at: str  # ISO format datetime
    pending_jobs: List[Dict[str, Any]]  # Serialized Job objects
    active_batches: List[str]  # Provider batch IDs
    completed_results: List[Dict[str, Any]]  # Serialized JobResult objects
    failed_jobs: List[Dict[str, Any]]  # Jobs with error info
    total_cost_usd: float
    config: Dict[str, Any]  # Batch configuration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchState':
        """Create from dictionary."""
        return cls(**data)


class StateManager:
    """Manages state persistence for batch runs.
    
    Provides thread-safe state management with automatic persistence
    to JSON files. Supports resuming interrupted batch runs.
    
    Example:
        >>> manager = StateManager("./batch_state.json")
        >>> manager.save_state(batch_state)
        >>> resumed_state = manager.load_state()
    """
    
    def __init__(self, state_file: str):
        """Initialize state manager.
        
        Args:
            state_file: Path to state file
        """
        self.state_file = Path(state_file)
        self._lock = threading.Lock()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the directory for state file exists."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state: BatchState) -> None:
        """Save batch state to file.
        
        Thread-safe method to persist state to JSON file.
        Creates a temporary file and atomically replaces the target.
        
        Args:
            state: BatchState to save
            
        Raises:
            StateError: If save operation fails
        """
        with self._lock:
            try:
                # Write to temporary file first
                temp_file = self.state_file.with_suffix('.tmp')
                
                with open(temp_file, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
                
                # Atomic replace
                temp_file.replace(self.state_file)
                
            except Exception as e:
                raise StateError(f"Failed to save state: {e}")
    
    def load_state(self) -> Optional[BatchState]:
        """Load batch state from file.
        
        Returns:
            BatchState if file exists, None otherwise
            
        Raises:
            StateError: If load operation fails
        """
        with self._lock:
            if not self.state_file.exists():
                return None
            
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                return BatchState.from_dict(data)
                
            except Exception as e:
                raise StateError(f"Failed to load state: {e}")
    
    def update_state(self, update_fn) -> None:
        """Atomically update state.
        
        Loads current state, applies update function, and saves.
        
        Args:
            update_fn: Function that takes BatchState and modifies it
            
        Raises:
            StateError: If update fails
        """
        with self._lock:
            # Load state without lock (we already have it)
            if not self.state_file.exists():
                raise StateError("No state found to update")
            
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                state = BatchState.from_dict(data)
            except Exception as e:
                raise StateError(f"Failed to load state: {e}")
            
            # Apply update
            update_fn(state)
            
            # Save state without lock (we already have it)
            try:
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
                temp_file.replace(self.state_file)
            except Exception as e:
                raise StateError(f"Failed to save state: {e}")
    
    def clear_state(self) -> None:
        """Delete the state file."""
        with self._lock:
            if self.state_file.exists():
                self.state_file.unlink()
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint of current state.
        
        Returns:
            Path to checkpoint file
        """
        with self._lock:
            if not self.state_file.exists():
                raise StateError("No state to checkpoint")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.state_file.with_suffix(f'.checkpoint_{timestamp}.json')
            
            with open(self.state_file, 'rb') as src:
                with open(checkpoint_file, 'wb') as dst:
                    dst.write(src.read())
            
            return str(checkpoint_file)
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints.
        
        Returns:
            List of checkpoint file paths
        """
        pattern = f"{self.state_file.stem}.checkpoint_*.json"
        checkpoints = list(self.state_file.parent.glob(pattern))
        return sorted([str(cp) for cp in checkpoints])
    
    def restore_checkpoint(self, checkpoint_path: str) -> None:
        """Restore state from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            StateError: If checkpoint doesn't exist or restore fails
        """
        with self._lock:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                raise StateError(f"Checkpoint not found: {checkpoint_path}")
            
            try:
                # Backup current state
                if self.state_file.exists():
                    backup = self.state_file.with_suffix('.backup')
                    self.state_file.rename(backup)
                
                # Copy checkpoint to state file
                with open(checkpoint, 'rb') as src:
                    with open(self.state_file, 'wb') as dst:
                        dst.write(src.read())
                        
            except Exception as e:
                # Try to restore backup
                backup = self.state_file.with_suffix('.backup')
                if backup.exists():
                    backup.rename(self.state_file)
                raise StateError(f"Failed to restore checkpoint: {e}")
    
    @staticmethod
    def serialize_job(job: Job) -> Dict[str, Any]:
        """Serialize a Job to dict.
        
        Args:
            job: Job to serialize
            
        Returns:
            Dictionary representation
        """
        return {
            "id": job.id,
            "model": job.model,
            "messages": job.messages,
            "file": str(job.file) if job.file else None,
            "prompt": job.prompt,
            "temperature": job.temperature,
            "max_tokens": job.max_tokens,
            "response_model": job.response_model.__name__ if job.response_model else None,
            "enable_citations": job.enable_citations
        }
    
    @staticmethod
    def serialize_job_result(result: JobResult) -> Dict[str, Any]:
        """Serialize a JobResult to dict.
        
        Args:
            result: JobResult to serialize
            
        Returns:
            Dictionary representation
        """
        return {
            "job_id": result.job_id,
            "response": result.response,
            "parsed_response": result.parsed_response if isinstance(result.parsed_response, dict) else None,
            "citations": [asdict(c) for c in result.citations] if result.citations else None,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "error": result.error
        }