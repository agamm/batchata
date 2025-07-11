"""
Unit tests for state management in BatchManager.
"""

import json
import os
import tempfile
import shutil

from batchata.batch_manager import BatchManager


class TestStateManagement:
    """Test state file operations and consistency."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "test_state.json")
        self.test_messages = [
            [{"role": "user", "content": f"Message {i}"}] 
            for i in range(1, 11)
        ]
        
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_state_file_creation(self):
        """Test that state file is created correctly."""
        manager = BatchManager(
            messages=self.test_messages[:5],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        assert os.path.exists(self.state_file)
        
        # Verify state file content
        with open(self.state_file, 'r') as f:
            state_data = json.load(f)
            
        assert "manager_id" in state_data
        assert state_data["model"] == "claude-3-haiku-20240307"
        assert state_data["items_per_job"] == 2
        assert len(state_data["jobs"]) == 3  # 5 messages / 2 per job = 3 jobs
        
    def test_state_file_loading(self):
        """Test loading existing state file."""
        # Create initial state with valid job structure
        initial_state = {
            "manager_id": "test-uuid-123",
            "created_at": "2024-01-20T10:00:00Z",
            "model": "claude-3-haiku-20240307",
            "items_per_job": 3,
            "max_cost": 10.0,
            "save_results_dir": None,
            "batch_kwargs": {},
            "jobs": [
                {
                    "index": 0,
                    "batch_id": None,
                    "status": "pending",
                    "items": [
                        {
                            "original_index": 0,
                            "content": [{"role": "user", "content": "Test message"}],
                            "status": "pending",
                            "error": None,
                            "cost": None,
                            "completed_at": None
                        }
                    ],
                    "job_cost": 0.0,
                    "started_at": None,
                    "completed_at": None
                }
            ],
            "total_cost": 1.50,
            "last_updated": "2024-01-20T10:05:00Z"
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(initial_state, f)
            
        # Load state (messages should be ignored when loading existing state)
        manager = BatchManager(
            messages=self.test_messages,  # Should be ignored
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        assert manager.state.manager_id == "test-uuid-123"
        assert manager.state.total_cost == 1.50
        assert manager.items_per_job == 3  # From loaded state
        
    def test_state_update_persistence(self):
        """Test that state updates are persisted to disk."""
        manager = BatchManager(
            messages=self.test_messages[:3],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        # Update state
        manager.state.total_cost = 5.25
        manager._save_state()
        
        # Verify persistence
        with open(self.state_file, 'r') as f:
            state_data = json.load(f)
            
        assert state_data["total_cost"] == 5.25