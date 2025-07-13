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
        assert tracker.get_remaining() is None
    
    def test_init_with_limit(self):
        """Test initialization with limit."""
        tracker = CostTracker(limit_usd=100.0)
        assert tracker.limit_usd == 100.0
        assert tracker.used_usd == 0.0
        assert tracker.get_remaining() == 100.0
    
    def test_can_proceed_no_limit(self):
        """Test can_proceed without limit."""
        tracker = CostTracker()
        assert tracker.can_proceed(1000.0) is True
        assert tracker.can_proceed(0.0) is True
    
    def test_can_proceed_with_limit(self):
        """Test can_proceed with limit."""
        tracker = CostTracker(limit_usd=100.0)
        assert tracker.can_proceed(50.0) is True
        assert tracker.can_proceed(100.0) is True
        assert tracker.can_proceed(101.0) is False
        
        # Track some usage
        tracker.track(60.0)
        assert tracker.can_proceed(40.0) is True
        assert tracker.can_proceed(41.0) is False
    
    def test_track_costs(self):
        """Test tracking costs."""
        tracker = CostTracker(limit_usd=100.0)
        
        tracker.track(25.0)
        assert tracker.used_usd == 25.0
        assert tracker.get_remaining() == 75.0
        
        tracker.track(30.0)
        assert tracker.used_usd == 55.0
        assert tracker.get_remaining() == 45.0
    
    def test_reserve_and_track(self):
        """Test reservation workflow."""
        tracker = CostTracker(limit_usd=100.0)
        
        # Reserve estimated cost
        assert tracker.reserve(50.0) is True
        assert tracker.used_usd == 50.0
        
        # Track actual cost (less than reserved)
        tracker.track(40.0, reserved_cost_usd=50.0)
        assert tracker.used_usd == 40.0
        
        # Reserve more
        assert tracker.reserve(30.0) is True
        assert tracker.used_usd == 70.0
        
        # Track actual cost (more than reserved)
        tracker.track(35.0, reserved_cost_usd=30.0)
        assert tracker.used_usd == 75.0
    
    def test_reserve_exceeds_limit(self):
        """Test reservation that exceeds limit."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track(80.0)
        
        assert tracker.reserve(20.0) is True
        assert tracker.reserve(1.0) is False  # Would exceed limit
    
    def test_check_limit(self):
        """Test check_limit method."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track(90.0)
        
        # Should not raise
        tracker.check_limit(10.0)
        
        # Should raise
        with pytest.raises(CostLimitExceededError) as exc_info:
            tracker.check_limit(11.0)
        
        assert "Cost limit would be exceeded" in str(exc_info.value)
        assert "$100.00" in str(exc_info.value)  # Limit
        assert "$90.00" in str(exc_info.value)   # Used
        assert "$11.00" in str(exc_info.value)   # Requested
    
    def test_get_stats(self):
        """Test getting statistics."""
        tracker = CostTracker(limit_usd=50.0)
        tracker.track(30.0)
        
        stats = tracker.get_stats()
        assert stats.total_cost_usd == 30.0
        assert stats.limit_usd == 50.0
        assert stats.remaining_usd == 20.0
        assert isinstance(stats.last_updated, datetime)
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track(50.0)
        
        tracker.reset()
        assert tracker.used_usd == 0.0
        assert tracker.limit_usd == 100.0  # Limit unchanged
        assert tracker.get_remaining() == 100.0
    
    def test_set_limit(self):
        """Test updating limit."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track(30.0)
        
        tracker.set_limit(50.0)
        assert tracker.limit_usd == 50.0
        assert tracker.can_proceed(20.0) is True
        assert tracker.can_proceed(21.0) is False
        
        tracker.set_limit(None)
        assert tracker.limit_usd is None
        assert tracker.can_proceed(1000.0) is True
    
    def test_repr(self):
        """Test string representation."""
        tracker = CostTracker(limit_usd=100.0)
        tracker.track(25.5)
        
        repr_str = repr(tracker)
        assert "CostTracker" in repr_str
        assert "used=$25.50" in repr_str
        assert "limit=$100.00" in repr_str
        assert "remaining=$74.50" in repr_str
        
        # Without limit
        tracker2 = CostTracker()
        tracker2.track(10.0)
        repr_str2 = repr(tracker2)
        assert "used=$10.00" in repr_str2
        assert "no limit" in repr_str2
    
    def test_thread_safety(self):
        """Test thread safety of cost tracker."""
        tracker = CostTracker(limit_usd=1000.0)
        
        def track_costs():
            for _ in range(100):
                if tracker.reserve(1.0):
                    time.sleep(0.0001)  # Simulate work
                    tracker.track(0.9, reserved_cost_usd=1.0)
        
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
    
    def test_save_and_load_state(self):
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
            manager.save_state(state)
            assert state_file.exists()
            
            # Load
            loaded = manager.load_state()
            assert loaded is not None
            assert loaded.batch_id == "batch-123"
            assert loaded.total_cost_usd == 10.5
            assert loaded.pending_jobs == [{"id": "job-1", "model": "test"}]
    
    def test_load_nonexistent(self):
        """Test loading when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            loaded = manager.load_state()
            assert loaded is None
    
    def test_update_state(self):
        """Test atomic state update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            # Create initial state
            state = BatchState(
                batch_id="batch-123",
                created_at=datetime.now().isoformat(),
                pending_jobs=[{"id": "job-1"}, {"id": "job-2"}],
                active_batches=[],
                completed_results=[],
                failed_jobs=[],
                total_cost_usd=0.0,
                config={}
            )
            manager.save_state(state)
            
            # Update state
            def update_fn(s):
                s.pending_jobs.pop(0)  # Remove first job
                s.active_batches.append("batch-456")
                s.total_cost_usd += 5.0
            
            manager.update_state(update_fn)
            
            # Verify update
            loaded = manager.load_state()
            assert len(loaded.pending_jobs) == 1
            assert loaded.pending_jobs[0]["id"] == "job-2"
            assert loaded.active_batches == ["batch-456"]
            assert loaded.total_cost_usd == 5.0
    
    def test_update_state_no_file(self):
        """Test updating when no state exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            with pytest.raises(StateError, match="No state found to update"):
                manager.update_state(lambda s: None)
    
    def test_clear_state(self):
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
            manager.save_state(state)
            assert state_file.exists()
            
            # Clear
            manager.clear_state()
            assert not state_file.exists()
            
            # Clear again (should not error)
            manager.clear_state()
    
    def test_checkpoint_operations(self):
        """Test checkpoint functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            # Create state
            state = BatchState(
                batch_id="batch-123",
                created_at=datetime.now().isoformat(),
                pending_jobs=[{"id": "job-1"}],
                active_batches=[],
                completed_results=[],
                failed_jobs=[],
                total_cost_usd=10.0,
                config={}
            )
            manager.save_state(state)
            
            # Create checkpoint
            checkpoint_path = manager.create_checkpoint()
            assert Path(checkpoint_path).exists()
            assert "checkpoint" in checkpoint_path
            
            # List checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoint_path in checkpoints
            
            # Modify state
            state.total_cost_usd = 20.0
            manager.save_state(state)
            
            # Restore checkpoint
            manager.restore_checkpoint(checkpoint_path)
            
            # Verify restoration
            loaded = manager.load_state()
            assert loaded.total_cost_usd == 10.0
    
    def test_checkpoint_no_state(self):
        """Test creating checkpoint when no state exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            with pytest.raises(StateError, match="No state to checkpoint"):
                manager.create_checkpoint()
    
    def test_restore_nonexistent_checkpoint(self):
        """Test restoring non-existent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(str(state_file))
            
            with pytest.raises(StateError, match="Checkpoint not found"):
                manager.restore_checkpoint("/nonexistent/checkpoint.json")
    
    def test_serialize_job(self):
        """Test job serialization."""
        job = Job(
            id="job-1",
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1000
        )
        
        serialized = StateManager.serialize_job(job)
        assert serialized["id"] == "job-1"
        assert serialized["model"] == "claude-3-sonnet"
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
        
        serialized = StateManager.serialize_job_result(result)
        assert serialized["job_id"] == "job-1"
        assert serialized["response"] == "Generated text"
        assert serialized["parsed_response"] == {"key": "value"}
        assert len(serialized["citations"]) == 1
        assert serialized["citations"][0]["text"] == "Quote"
        assert serialized["cost_usd"] == 0.005
        assert serialized["error"] is None