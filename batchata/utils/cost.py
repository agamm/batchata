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
        >>> if tracker.reserve_cost(5.0):
        ...     # Do work
        ...     tracker.adjust_reserved_cost(5.0, 4.8)
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
    
    def reserve_cost(self, estimated_cost: float) -> bool:
        """Reserve budget for estimated cost atomically.
        
        Args:
            estimated_cost: Estimated cost to reserve
            
        Returns:
            True if reservation successful, False if would exceed limit
        """
        with self._lock:
            if self.limit_usd is None:
                self.used_usd += estimated_cost
                self._last_updated = datetime.now()
                return True
            
            if (self.used_usd + estimated_cost) <= self.limit_usd:
                self.used_usd += estimated_cost
                self._last_updated = datetime.now()
                return True
            
            return False
    
    def adjust_reserved_cost(self, estimated_cost: float, actual_cost: float):
        """Adjust previously reserved cost to actual cost.
        
        Args:
            estimated_cost: Previously reserved amount
            actual_cost: Actual cost incurred
        """
        with self._lock:
            # Remove the reserved amount and add the actual cost
            self.used_usd = self.used_usd - estimated_cost + actual_cost
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
            # Calculate remaining budget without calling self.remaining() to avoid deadlock
            remaining_usd = None
            if self.limit_usd is not None:
                remaining_usd = max(0.0, self.limit_usd - self.used_usd)
            
            return {
                "total_cost_usd": self.used_usd,
                "limit_usd": self.limit_usd,
                "remaining_usd": remaining_usd,
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