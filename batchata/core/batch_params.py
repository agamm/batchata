"""BatchParams data model."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .job import Job
from ..providers import get_provider


@dataclass
class BatchParams:
    """Parameters for a batch job.
    
    Attributes:
        state_file: Optional path to state file for persistence
        results_dir: Directory to store results
        max_parallel_batches: Maximum parallel batch requests
        items_per_batch: Number of jobs per provider batch
        cost_limit_usd: Optional cost limit in USD
        default_params: Default parameters for all jobs
        reuse_state: Whether to resume from existing state file
        raw_files: Whether to save debug files (raw responses, JSONL files) from providers
        verbosity: Logging verbosity level ("debug", "info", "warn", "error")
        time_limit_seconds: Optional time limit in seconds for the entire batch (min 10s, max 24h)
    """
    
    state_file: Optional[str]
    results_dir: str
    max_parallel_batches: int
    items_per_batch: int = 10
    cost_limit_usd: Optional[float] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    reuse_state: bool = True
    raw_files: bool = True
    verbosity: str = "info"
    time_limit_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.max_parallel_batches <= 0:
            raise ValueError("max_parallel_batches must be greater than 0")
        
        if self.items_per_batch <= 0:
            raise ValueError("items_per_batch must be greater than 0")
        
        if self.cost_limit_usd is not None and self.cost_limit_usd < 0:
            raise ValueError("cost_limit_usd must be non-negative")
        
        if self.time_limit_seconds is not None:
            if self.time_limit_seconds < 10:
                raise ValueError("time_limit_seconds must be at least 10 seconds")
            if self.time_limit_seconds > 86400:  # 24 hours
                raise ValueError("time_limit_seconds must be at most 24 hours (86400 seconds)")
    
    def validate_default_params(self, model: str) -> None:
        """Validate default parameters for a model."""
        if not self.default_params:
            return
        
        provider = get_provider(model)
        provider.validate_params(model, **self.default_params)