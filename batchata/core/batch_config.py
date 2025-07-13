"""BatchConfig data model."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .job import Job
from ..providers import get_provider


@dataclass
class BatchConfig:
    """Configuration for a batch job.
    
    Attributes:
        state_file: Path to state file for persistence
        results_dir: Directory to store results
        max_concurrent: Maximum concurrent batch requests
        cost_limit_usd: Optional cost limit in USD
        default_params: Default parameters for all jobs
        progress_callback: Optional callback for progress updates
        progress_interval: Interval in seconds between progress updates
        jobs: List of jobs to process
    """
    
    state_file: str
    results_dir: str
    max_concurrent: int
    cost_limit_usd: Optional[float] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    progress_callback: Optional[Callable[[Dict, float], None]] = None
    progress_interval: float = 3.0
    jobs: List[Job] = field(default_factory=list)
    
    def validate_default_params(self, model: str) -> None:
        """Validate default parameters for a model."""
        if not self.default_params:
            return
        
        provider = get_provider(model)
        provider.validate_params(model, **self.default_params)