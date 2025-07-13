"""Cost tracking utilities."""

import threading
from datetime import datetime
from typing import Any, Dict, Optional

from ..exceptions import CostLimitExceededError


class CostTracker:
    """Tracks cumulative costs and enforces limits.
    
    Thread-safe implementation for tracking costs across multiple
    concurrent operations. Supports optional cost limits and provides
    detailed statistics.
    
    Example:
        >>> tracker = CostTracker(limit_usd=100.0)
        >>> if tracker.can_afford(5.0):
        ...     # Do work
        ...     tracker.track_spend(4.8)
    """
    
    def __init__(self, limit_usd: Optional[float] = None):
        """Initialize cost tracker.
        
        Args:
            limit_usd: Optional cost limit in USD
        """
        self.limit_usd = limit_usd
        self.used_usd = 0.0
        self._lock = threading.Lock()
        self._last_updated = datetime.now()
    
    def can_afford(self, cost_usd: float) -> bool:
        """Check if we can afford the given cost.
        
        Args:
            cost_usd: Cost to check
            
        Returns:
            True if we can afford it, False otherwise
        """
        with self._lock:
            if self.limit_usd is None:
                return True
            
            return (self.used_usd + cost_usd) <= self.limit_usd
    
    def track_spend(self, cost_usd: float):
        """Track actual spending.
        
        Args:
            cost_usd: Actual cost incurred
        """
        with self._lock:
            self.used_usd += cost_usd
            self._last_updated = datetime.now()
    
    
    def remaining(self) -> Optional[float]:
        """Get remaining budget.
        
        Returns:
            Remaining budget in USD, or None if no limit set
        """
        with self._lock:
            if self.limit_usd is None:
                return None
            return max(0.0, self.limit_usd - self.used_usd)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current cost statistics.
        
        Returns:
            Dictionary with current statistics
        """
        with self._lock:
            return {
                "total_cost_usd": self.used_usd,
                "limit_usd": self.limit_usd,
                "remaining_usd": self.remaining(),
                "last_updated": self._last_updated
            }
    
    def reset(self):
        """Reset the cost tracker to initial state."""
        with self._lock:
            self.used_usd = 0.0
            self._last_updated = datetime.now()
    
    def set_limit(self, limit_usd: Optional[float]):
        """Update the cost limit.
        
        Args:
            limit_usd: New limit in USD (None to remove limit)
        """
        with self._lock:
            self.limit_usd = limit_usd
    
    def __repr__(self) -> str:
        """String representation of the tracker."""
        if self.limit_usd is None:
            return f"CostTracker(used=${self.used_usd:.2f}, no limit)"
        else:
            return (
                f"CostTracker(used=${self.used_usd:.2f}, "
                f"limit=${self.limit_usd:.2f}, "
                f"remaining=${self.remaining():.2f})"
            )