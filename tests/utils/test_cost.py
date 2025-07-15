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
        
        # No limit means can reserve anything
        assert tracker.reserve_cost(1000000.0) is True
        assert tracker.remaining() is None
        
        # Reserve some costs
        assert tracker.reserve_cost(10.0) is True
        assert tracker.reserve_cost(5.5) is True
        assert tracker.reserve_cost(2.25) is True
        
        assert tracker.used_usd == 1000017.75
        assert tracker.remaining() is None
        
        # Stats should reflect usage
        stats = tracker.get_stats()
        assert stats["total_cost_usd"] == 1000017.75
        assert stats["limit_usd"] is None
        assert stats["remaining_usd"] is None
    
    def test_cost_limit_enforcement(self):
        """Test enforcing cost limits."""
        tracker = CostTracker(limit_usd=50.0)
        
        # Should be able to reserve within limit
        assert tracker.reserve_cost(30.0) is True
        assert tracker.reserve_cost(20.0) is True
        assert tracker.reserve_cost(0.01) is False  # Would exceed limit
        
        # Check current state
        assert tracker.used_usd == 50.0
        assert tracker.remaining() == 0.0
        
        # Test adjustment - reduce one reservation
        tracker.adjust_reserved_cost(20.0, 15.0)  # Actual cost was less
        assert tracker.used_usd == 45.0
        assert tracker.remaining() == 5.0
        
        # Now we can reserve more
        assert tracker.reserve_cost(5.0) is True
        assert tracker.reserve_cost(0.01) is False  # Would exceed limit
        
        # Stats
        stats = tracker.get_stats()
        assert stats["total_cost_usd"] == 50.0
        assert stats["limit_usd"] == 50.0
        assert stats["remaining_usd"] == 0.0
    
    def test_thread_safety(self):
        """Test thread-safe cost reservation."""
        tracker = CostTracker(limit_usd=1000.0)
        errors = []
        
        def reserve_costs(amount, count):
            try:
                for _ in range(count):
                    # Reserve cost atomically - no race condition
                    tracker.reserve_cost(amount)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=reserve_costs, args=(10.0, 20))
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
    
    def test_concurrent_cost_reservation_prevents_overspending(self):
        """Test that concurrent cost reservation prevents budget overspending."""
        tracker = CostTracker(limit_usd=100.0)
        
        successful_reservations = []
        failed_reservations = []
        errors = []
        
        def try_reserve_cost(thread_id, amount):
            """Try to reserve cost from multiple threads."""
            try:
                if tracker.reserve_cost(amount):
                    successful_reservations.append((thread_id, amount))
                else:
                    failed_reservations.append((thread_id, amount))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Create multiple threads trying to reserve costs
        threads = []
        for i in range(10):
            # Each thread tries to reserve $15 (10 * $15 = $150 > $100 limit)
            t = threading.Thread(target=try_reserve_cost, args=(i, 15.0))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        assert len(errors) == 0, f"Got unexpected errors: {errors}"
        assert tracker.used_usd <= tracker.limit_usd, f"Budget exceeded! Used: ${tracker.used_usd:.2f}, Limit: ${tracker.limit_usd:.2f}"
        
        # Should have exactly 6 successful reservations (6 * $15 = $90 <= $100)
        # and 4 failed reservations
        expected_successful = int(tracker.limit_usd // 15.0)  # 6
        expected_failed = 10 - expected_successful  # 4
        
        assert len(successful_reservations) == expected_successful
        assert len(failed_reservations) == expected_failed
        assert tracker.used_usd == 90.0
    
    def test_cost_adjustment_after_reservation(self):
        """Test that cost adjustments work correctly after reservations."""
        tracker = CostTracker(limit_usd=100.0)
        
        # Reserve some costs
        assert tracker.reserve_cost(50.0) is True
        assert tracker.reserve_cost(30.0) is True
        assert tracker.used_usd == 80.0
        
        # Should be near limit
        assert tracker.reserve_cost(25.0) is False
        
        # Adjust the first reservation down (actual cost was less)
        tracker.adjust_reserved_cost(50.0, 40.0)
        assert tracker.used_usd == 70.0
        
        # Now we should be able to reserve more
        assert tracker.reserve_cost(10.0) is True
        assert tracker.used_usd == 80.0
        
        # Adjust the second reservation up (actual cost was more)
        tracker.adjust_reserved_cost(30.0, 35.0)
        assert tracker.used_usd == 85.0
        assert tracker.remaining() == 15.0
    
