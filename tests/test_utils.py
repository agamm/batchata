"""Tests for utility modules."""

import json
import pytest
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from batchata.core import Job, JobResult
from batchata.utils import CostTracker, StateManager
from batchata.utils.state import BatchState
from batchata.exceptions import CostLimitExceededError, StateError
from batchata.types import Citation


class TestCostTracker:
    """Tests for CostTracker."""
    
    def test_init_no_limit(self):
        """Test initialization without limit."""
        tracker = CostTracker()
        assert tracker.limit_usd is None
        assert tracker.used_usd == 0.0
        assert tracker.remaining() is None
    
    def test_init_with_limit(self):
        """Test initialization with limit."""
        tracker = CostTracker(limit_usd=100.0)
        assert tracker.limit_usd == 100.0
        assert tracker.used_usd == 0.0
        assert tracker.remaining() == 100.0
    
    def test_can_afford_no_limit(self):
        """Test can_afford without limit."""
        tracker = CostTracker()
        assert tracker.can_afford(1000.0) is True
        assert tracker.can_afford(0.0) is True
    
    def test_can_afford_with_limit(self):
        """Test can_afford with limit."""
        tracker = CostTracker(limit_usd=100.0)
        assert tracker.can_afford(50.0) is True
        assert tracker.can_afford(100.0) is True
        assert tracker.can_afford(101.0) is False
        
        # Track some usage
        tracker.track_spend(60.0)
        assert tracker.can_afford(40.0) is True
        assert tracker.can_afford(41.0) is False
    
    def test_track_costs(self):
        """Test tracking costs."""
        tracker = CostTracker(limit_usd=100.0)
        
        tracker.track_spend(25.0)
        assert tracker.used_usd == 25.0
        assert tracker.remaining() == 75.0
        
        tracker.track_spend(30.0)
        assert tracker.used_usd == 55.0
        assert tracker.remaining() == 45.0
    
    def test_simple_tracking(self):
        """Test simple cost tracking workflow."""
        tracker = CostTracker(limit_usd=100.0)
        
        # Can afford initial cost
        assert tracker.can_afford(50.0) is True
        
        # Track actual cost
        tracker.track_spend(40.0)
        assert tracker.used_usd == 40.0
        
        # Can afford more
        assert tracker.can_afford(30.0) is True
        
        # Track more actual cost
        tracker.track_spend(35.0)
        assert tracker.used_usd == 75.0
    
    def test_can_afford_limit_checks(self):
        """Test can_afford respects limits."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track_spend(80.0)
        
        assert tracker.can_afford(20.0) is True
        assert tracker.can_afford(21.0) is False  # Would exceed limit
    
    def test_simplified_cost_checks(self):
        """Test simplified cost checking."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track_spend(90.0)
        
        # Can afford within limit
        assert tracker.can_afford(10.0) is True
        
        # Cannot afford over limit
        assert tracker.can_afford(11.0) is False
    
    def test_get_stats(self):
        """Test getting statistics."""
        tracker = CostTracker(limit_usd=50.0)
        tracker.track_spend(30.0)
        
        stats = tracker.get_stats()
        assert stats["total_cost_usd"] == 30.0
        assert stats["limit_usd"] == 50.0
        assert stats["remaining_usd"] == 20.0
        assert isinstance(stats["last_updated"], datetime)
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track_spend(50.0)
        
        tracker.reset()
        assert tracker.used_usd == 0.0
        assert tracker.limit_usd == 100.0  # Limit unchanged
        assert tracker.remaining() == 100.0
    
    def test_set_limit(self):
        """Test updating limit."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track_spend(30.0)
        
        tracker.set_limit(50.0)
        assert tracker.limit_usd == 50.0
        assert tracker.can_afford(20.0) is True
        assert tracker.can_afford(21.0) is False
        
        tracker.set_limit(None)
        assert tracker.limit_usd is None
        assert tracker.can_afford(1000.0) is True
    
    def test_repr(self):
        """Test string representation."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track_spend(25.5)
        
        repr_str = repr(tracker)
        assert "CostTracker" in repr_str
        assert "used=$25.50" in repr_str
        assert "limit=$100.00" in repr_str
        assert "remaining=$74.50" in repr_str
        
        # Without limit
        tracker2 = CostTracker()
        tracker2.track_spend(10.0)
        repr_str2 = repr(tracker2)
        assert "used=$10.00" in repr_str2
        assert "no limit" in repr_str2
    
    def test_thread_safety(self):
        """Test thread safety of cost tracker."""
        tracker = CostTracker(limit_usd=1000.0)
        
        def track_costs():
            for _ in range(100):
                if tracker.can_afford(1.0):
                    time.sleep(0.0001)  # Simulate work
                    tracker.track_spend(0.9)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=track_costs)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have tracked costs correctly
        assert tracker.used_usd <= 1000.0
        assert tracker.used_usd > 0.0


class TestStateManager:
    """Tests for StateManager."""
    
    def test_init_creates_directory(self):
        """Test that initialization creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "subdir" / "state.json"
            manager = StateManager(str(state_file))
            
            assert state_file.parent.exists()
    
    def test_save_and_load(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            # Create state
            state = BatchState(
                batch_id="batch-123",
                created_at=datetime.now().isoformat(),
                pending_jobs=[{"id": "job-1", "model": "test"}],
                active_batches=["provider-batch-1"],
                completed_results=[],
                failed_jobs=[],
                total_cost_usd=10.5,
                config={"max_concurrent": 5}
            )
            
            # Save
            manager.save(state)
            assert state_file.exists()
            
            # Load
            loaded = manager.load()
            assert loaded is not None
            assert loaded.batch_id == "batch-123"
            assert loaded.total_cost_usd == 10.5
            assert loaded.pending_jobs == [{"id": "job-1", "model": "test"}]
    
    def test_load_nonexistent(self):
        """Test loading when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            loaded = manager.load()
            assert loaded is None
    
    def test_clear(self):
        """Test clearing state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            # Create state
            state = BatchState(
                batch_id="batch-123",
                created_at=datetime.now().isoformat(),
                pending_jobs=[],
                active_batches=[],
                completed_results=[],
                failed_jobs=[],
                total_cost_usd=0.0,
                config={}
            )
            manager.save(state)
            assert state_file.exists()
            
            # Clear
            manager.clear()
            assert not state_file.exists()
            
            # Clear again (should not error)
            manager.clear()
    
    def test_serialize_job(self):
        """Test job serialization."""
        job = Job(
            id="job-1",
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1000
        )
        
        serialized = job.to_dict()
        assert serialized["id"] == "job-1"
        assert serialized["model"] == "claude-sonnet-4-20250514"
        assert serialized["messages"] == [{"role": "user", "content": "Hello"}]
        assert serialized["temperature"] == 0.7
        assert serialized["file"] is None
        assert serialized["response_model"] is None
    
    def test_serialize_job_result(self):
        """Test job result serialization."""
        result = JobResult(
            job_id="job-1",
            response="Generated text",
            parsed_response={"key": "value"},
            citations=[Citation(text="Quote", source="Page 1")],
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        
        serialized = result.to_dict()
        assert serialized["job_id"] == "job-1"
        assert serialized["response"] == "Generated text"
        assert serialized["parsed_response"] == {"key": "value"}
        assert len(serialized["citations"]) == 1
        assert serialized["citations"][0]["text"] == "Quote"
        assert serialized["cost_usd"] == 0.005
        assert serialized["error"] is None