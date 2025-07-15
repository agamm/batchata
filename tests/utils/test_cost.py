"""Tests for cost tracking utilities.

Testing:
1. Cost tracking and limit enforcement
2. Thread-safe operations
3. Statistics and reset functionality
"""

import pytest
import threading
import time

from batchata.utils.cost import CostTracker
from batchata.exceptions import CostLimitExceededError


class TestCostTracker:
    """Test CostTracker functionality."""
    
    def test_cost_tracking_without_limit(self):
        """Test tracking costs without a limit."""
        tracker = CostTracker()
        
        # No limit means can afford anything
        assert tracker.can_afford(1000000.0) is True
        assert tracker.remaining() is None
        
        # Track some spending
        tracker.track_spend(10.0)
        tracker.track_spend(5.5)
        tracker.track_spend(2.25)
        
        assert tracker.used_usd == 17.75
        assert tracker.remaining() is None
        
        # Stats should reflect usage
        stats = tracker.get_stats()
        assert stats["total_cost_usd"] == 17.75
        assert stats["limit_usd"] is None
        assert stats["remaining_usd"] is None
    
    def test_cost_limit_enforcement(self):
        """Test enforcing cost limits."""
        tracker = CostTracker(limit_usd=50.0)
        
        # Should be able to afford within limit
        assert tracker.can_afford(30.0) is True
        assert tracker.can_afford(50.0) is True
        assert tracker.can_afford(50.01) is False
        
        # Track spending
        tracker.track_spend(20.0)
        assert tracker.used_usd == 20.0
        assert tracker.remaining() == 30.0
        
        # Check affordability with current spending
        assert tracker.can_afford(30.0) is True
        assert tracker.can_afford(30.01) is False
        
        # Track more
        tracker.track_spend(25.0)
        assert tracker.used_usd == 45.0
        assert tracker.remaining() == 5.0
        
        # Near limit
        assert tracker.can_afford(5.0) is True
        assert tracker.can_afford(6.0) is False
        
        # Stats
        stats = tracker.get_stats()
        assert stats["total_cost_usd"] == 45.0
        assert stats["limit_usd"] == 50.0
        assert stats["remaining_usd"] == 5.0
    
    def test_thread_safety(self):
        """Test thread-safe cost tracking."""
        tracker = CostTracker(limit_usd=1000.0)
        errors = []
        
        def track_costs(amount, count):
            try:
                for _ in range(count):
                    if tracker.can_afford(amount):
                        tracker.track_spend(amount)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=track_costs, args=(10.0, 20))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors
        assert len(errors) == 0
        
        # Total should be correct (5 threads * 20 iterations * $10)
        assert tracker.used_usd == 1000.0
        assert tracker.remaining() == 0.0