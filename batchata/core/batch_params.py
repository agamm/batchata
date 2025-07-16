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
        max_concurrent: Maximum concurrent batch requests
        items_per_batch: Number of jobs per provider batch
        cost_limit_usd: Optional cost limit in USD
        default_params: Default parameters for all jobs
        reuse_state: Whether to resume from existing state file
        save_raw_responses: Whether to save raw API responses from providers
        verbosity: Logging verbosity level ("debug", "info", "warn", "error")
    """
    
    state_file: Optional[str]
    results_dir: str
    max_concurrent: int
    items_per_batch: int = 10
    cost_limit_usd: Optional[float] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    reuse_state: bool = True
    save_raw_responses: bool = True
    verbosity: str = "info"
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        
        if self.items_per_batch <= 0:
            raise ValueError("items_per_batch must be greater than 0")
        
        if self.cost_limit_usd is not None and self.cost_limit_usd < 0:
            raise ValueError("cost_limit_usd must be non-negative")
    
    def validate_default_params(self, model: str) -> None:
        """Validate default parameters for a model."""
        if not self.default_params:
            return
        
        provider = get_provider(model)
        provider.validate_params(model, **self.default_params)