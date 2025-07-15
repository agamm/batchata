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
        created_at: When the batch was created
        pending_jobs: Jobs that haven't been submitted yet
        active_batches: Currently running batch IDs
        completed_results: Results from completed jobs
        failed_jobs: Jobs that failed with errors
        total_cost_usd: Total cost incurred so far
        config: Original batch configuration
    """
    
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
    """Simple state persistence for batch runs.
    
    Focused class that handles save, load, and clear operations
    for batch state persistence. Thread-safe with atomic file operations.
    
    Example:
        >>> manager = StateManager("./batch_state.json")
        >>> manager.save(batch_state)
        >>> resumed_state = manager.load()
        >>> manager.clear()  # Remove state file
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
    
    def save(self, state) -> None:
        """Save batch state to file.
        
        Thread-safe method to persist state to JSON file.
        Creates a temporary file and atomically replaces the target.
        
        Args:
            state: BatchState or BatchRun object with to_json() method
            
        Raises:
            StateError: If save operation fails
        """
        with self._lock:
            try:
                # Write to temporary file first
                temp_file = self.state_file.with_suffix('.tmp')
                
                # Convert to dict - handle both BatchState and BatchRun
                if hasattr(state, 'to_json'):
                    data = state.to_json()
                elif hasattr(state, 'to_dict'):
                    data = state.to_dict()
                else:
                    raise StateError(f"State object must have to_json() or to_dict() method")
                
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Atomic replace
                temp_file.replace(self.state_file)
                
            except Exception as e:
                raise StateError(f"Failed to save state: {e}")
    
    def load(self) -> Optional[BatchState]:
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
    
    def clear(self) -> None:
        """Delete the state file."""
        with self._lock:
            if self.state_file.exists():
                self.state_file.unlink()
    
