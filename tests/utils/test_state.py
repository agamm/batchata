"""Tests for state persistence utilities.

Testing:
1. State save and load operations
2. State file atomic operations
3. Thread safety and error handling
"""

import pytest
import json
import threading
from pathlib import Path
from datetime import datetime

from batchata.utils.state import BatchState, StateManager
from batchata.exceptions import StateError
from batchata.core.job import Job
from batchata.core.job_result import JobResult


class TestStatePersistence:
    """Test state persistence functionality."""
    
    def test_save_and_load_state(self, temp_dir):
        """Test saving and loading batch state."""
        state_file = str(temp_dir / "test_state.json")
        manager = StateManager(state_file)
        
        # Create test state
        state = BatchState(
            created_at=datetime.now().isoformat(),
            pending_jobs=[
                {"id": "job-1", "messages": [{"role": "user", "content": "Q1"}]},
                {"id": "job-2", "messages": [{"role": "user", "content": "Q2"}]}
            ],
            active_batches=["batch-123"],
            completed_results=[
                {"job_id": "job-0", "content": "A0", "cost": 0.01, 
                 "input_tokens": 10, "output_tokens": 20}
            ],
            failed_jobs=[],
            total_cost_usd=0.01,
            config={"max_parallel_batches": 10, "items_per_batch": 5}
        )
        
        # Save state
        manager.save(state)
        assert Path(state_file).exists()
        
        # Load state
        loaded = manager.load()
        
        assert loaded is not None
        assert len(loaded.pending_jobs) == 2
        assert loaded.pending_jobs[0]["id"] == "job-1"
        assert len(loaded.active_batches) == 1
        assert loaded.active_batches[0] == "batch-123"
        assert len(loaded.completed_results) == 1
        assert loaded.completed_results[0]["job_id"] == "job-0"
        assert loaded.total_cost_usd == 0.01
        assert loaded.config["max_parallel_batches"] == 10
    
    def test_atomic_file_operations(self, temp_dir):
        """Test atomic file replacement during save."""
        state_file = str(temp_dir / "atomic_test.json")
        manager = StateManager(state_file)
        
        # Create initial state
        state1 = BatchState(
            created_at=datetime.now().isoformat(),
            pending_jobs=[{"id": "job-1"}],
            active_batches=[],
            completed_results=[],
            failed_jobs=[],
            total_cost_usd=1.0,
            config={}
        )
        
        manager.save(state1)
        
        # Simulate partial write by creating temp file
        temp_file = Path(state_file).with_suffix('.tmp')
        temp_file.write_text("partial write")
        
        # Save new state - should complete atomically
        state2 = BatchState(
            created_at=datetime.now().isoformat(),
            pending_jobs=[{"id": "job-2"}],
            active_batches=[],
            completed_results=[],
            failed_jobs=[],
            total_cost_usd=2.0,
            config={}
        )
        
        manager.save(state2)
        
        # Load should get complete state, not partial
        loaded = manager.load()
        assert loaded.total_cost_usd == 2.0
        assert loaded.pending_jobs[0]["id"] == "job-2"
        
        # Temp file should be gone
        assert not temp_file.exists()
    
    def test_thread_safety(self, temp_dir):
        """Test thread-safe state operations."""
        state_file = str(temp_dir / "thread_test.json")
        manager = StateManager(state_file)
        results = []
        
        def save_and_load(thread_id):
            try:
                # Create unique state
                state = BatchState(
                    created_at=datetime.now().isoformat(),
                    pending_jobs=[{"id": f"job-{thread_id}"}],
                    active_batches=[],
                    completed_results=[],
                    failed_jobs=[],
                    total_cost_usd=float(thread_id),
                    config={"thread": thread_id}
                )
                
                # Save
                manager.save(state)
                
                # Load
                loaded = manager.load()
                results.append((thread_id, loaded.total_cost_usd))
                
            except Exception as e:
                results.append((thread_id, f"error: {e}"))
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=save_and_load, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All operations should complete without errors
        assert len(results) == 5
        assert all(not str(r[1]).startswith("error") for r in results)