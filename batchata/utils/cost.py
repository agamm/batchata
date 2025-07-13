"""Cost tracking utilities."""

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..exceptions import CostLimitExceededError


@dataclass
class CostStats:
    """Statistics about cost usage.
    
    Attributes:
        total_cost_usd: Total cost incurred so far
        limit_usd: Cost limit (if set)
        remaining_usd: Remaining budget (if limit set)
        last_updated: When stats were last updated
    """
    
    total_cost_usd: float
    limit_usd: Optional[float]
    remaining_usd: Optional[float]
    last_updated: datetime


class CostTracker:
    """Tracks cumulative costs and enforces limits.
    
    Thread-safe implementation for tracking costs across multiple
    concurrent operations. Supports optional cost limits and provides
    detailed statistics.
    
    Example:
        >>> tracker = CostTracker(limit_usd=100.0)
        >>> if tracker.can_proceed(5.0):
        ...     # Do work
        ...     tracker.track(4.8)  # Track actual cost
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
    
    def can_proceed(self, estimated_cost_usd: float) -> bool:
        """Check if we can proceed with given cost.
        
        This method is thread-safe and checks whether the estimated cost
        would exceed the limit (if set).
        
        Args:
            estimated_cost_usd: Estimated cost for the operation
            
        Returns:
            True if operation can proceed, False otherwise
        """
        with self._lock:
            if self.limit_usd is None:
                return True
            
            return (self.used_usd + estimated_cost_usd) <= self.limit_usd
    
    def reserve(self, estimated_cost_usd: float) -> bool:
        """Reserve budget for an operation.
        
        Atomically checks if the cost can be accommodated and reserves it.
        This prevents race conditions where multiple threads might check
        can_proceed() simultaneously.
        
        Args:
            estimated_cost_usd: Estimated cost to reserve
            
        Returns:
            True if reservation successful, False otherwise
        """
        with self._lock:
            if self.limit_usd is None:
                return True
            
            if (self.used_usd + estimated_cost_usd) <= self.limit_usd:
                self.used_usd += estimated_cost_usd
                self._last_updated = datetime.now()
                return True
            
            return False
    
    def track(self, actual_cost_usd: float, reserved_cost_usd: Optional[float] = None):
        """Track actual cost after completion.
        
        If a cost was reserved, this adjusts the tracked amount to reflect
        the actual cost. If no reservation was made, it simply adds the cost.
        
        Args:
            actual_cost_usd: Actual cost incurred
            reserved_cost_usd: Amount that was reserved (if any)
        """
        with self._lock:
            if reserved_cost_usd is not None:
                # Adjust from reserved to actual
                adjustment = actual_cost_usd - reserved_cost_usd
                self.used_usd += adjustment
            else:
                # No reservation, just add
                self.used_usd += actual_cost_usd
            
            self._last_updated = datetime.now()
    
    def check_limit(self, estimated_cost_usd: float) -> None:
        """Check if cost would exceed limit and raise if so.
        
        Args:
            estimated_cost_usd: Estimated cost to check
            
        Raises:
            CostLimitExceededError: If cost would exceed limit
        """
        if not self.can_proceed(estimated_cost_usd):
            remaining = self.get_remaining()
            raise CostLimitExceededError(
                f"Cost limit would be exceeded. "
                f"Limit: ${self.limit_usd:.2f}, "
                f"Used: ${self.used_usd:.2f}, "
                f"Requested: ${estimated_cost_usd:.2f}, "
                f"Remaining: ${remaining:.2f}"
            )
    
    def get_remaining(self) -> Optional[float]:
        """Get remaining budget.
        
        Returns:
            Remaining budget in USD, or None if no limit set
        """
        with self._lock:
            if self.limit_usd is None:
                return None
            return max(0.0, self.limit_usd - self.used_usd)
    
    def get_stats(self) -> CostStats:
        """Get current cost statistics.
        
        Returns:
            CostStats object with current statistics
        """
        with self._lock:
            remaining = None
            if self.limit_usd is not None:
                remaining = max(0.0, self.limit_usd - self.used_usd)
            
            return CostStats(
                total_cost_usd=self.used_usd,
                limit_usd=self.limit_usd,
                remaining_usd=remaining,
                last_updated=self._last_updated
            )
    
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
        stats = self.get_stats()
        if stats.limit_usd is None:
            return f"CostTracker(used=${stats.total_cost_usd:.2f}, no limit)"
        else:
            return (
                f"CostTracker(used=${stats.total_cost_usd:.2f}, "
                f"limit=${stats.limit_usd:.2f}, "
                f"remaining=${stats.remaining_usd:.2f})"
            )