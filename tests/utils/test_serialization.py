"""Tests for serialization utilities.

Testing:
1. to_dict conversion for Job and JobResult
2. Batch state serialization
3. Basic type handling
"""

import pytest
import json

from batchata.utils.serialization import to_dict
from batchata.core.job import Job
from batchata.core.job_result import JobResult
from batchata.utils.state import BatchState


class TestSerialization:
    """Test serialization utilities."""
    
    def test_job_serialization(self):
        """Test serializing Job objects."""
        job = Job(
            id="test-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1000
        )
        
        serialized = to_dict(job)
        
        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert json_str  # Just verify it serializes
        
        # Check key fields
        assert serialized["id"] == "test-job"
        assert serialized["model"] == "claude-3-5-sonnet-20241022"
    
    def test_job_result_serialization(self):
        """Test serializing JobResult objects."""
        result = JobResult(
            job_id="test-job",
            response="The answer is 42",
            cost_usd=0.01,
            input_tokens=100,
            output_tokens=50
        )
        
        serialized = to_dict(result)
        
        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert json_str
        
        # Check key fields
        assert serialized["job_id"] == "test-job"
        assert serialized["response"] == "The answer is 42"
    
    def test_batch_state_serialization(self):
        """Test serializing BatchState objects."""
        state = BatchState(
            created_at="2024-01-01T00:00:00",
            pending_jobs=[{"id": "job-1"}],
            active_batches=["batch-123"],
            completed_results=[],
            failed_jobs=[],
            total_cost_usd=0.0,
            config={"max_concurrent": 10}
        )
        
        serialized = to_dict(state)
        
        # Should be JSON serializable  
        json_str = json.dumps(serialized)
        assert json_str
        
        # Check structure is preserved
        assert serialized["created_at"] == "2024-01-01T00:00:00"
        assert len(serialized["pending_jobs"]) == 1
        assert serialized["active_batches"] == ["batch-123"]