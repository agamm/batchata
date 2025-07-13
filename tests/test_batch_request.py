"""Tests for BatchRequest class."""

import pytest
import time
from datetime import datetime

from batchata.core import Job
from batchata.providers import BatchRequest
from tests.mocks import MockProvider


class TestBatchRequest:
    """Tests for BatchRequest class."""
    
    def test_batch_request_creation(self):
        """Test creating a batch request."""
        provider = MockProvider(auto_register=False)
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        ]
        
        batch = BatchRequest(
            id="batch-123",
            provider=provider,
            jobs=jobs
        )
        
        assert batch.id == "batch-123"
        assert batch.provider is provider
        assert batch.jobs == jobs
        assert batch.status == "pending"
        assert isinstance(batch.submitted_at, datetime)
    
    def test_batch_status_update(self):
        """Test updating batch status."""
        provider = MockProvider(auto_register=False)
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        ]
        
        # Create batch through provider
        batch = provider.create_batch(jobs)
        
        # Initially pending
        assert batch.status == "pending"
        assert not batch.is_complete
        assert not batch.is_failed
        
        # Update status
        status = batch.get_status()
        assert status in ["pending", "running", "complete"]
    
    def test_batch_get_results_not_complete(self):
        """Test getting results before batch is complete."""
        provider = MockProvider(auto_register=False)
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        ]
        
        batch = provider.create_batch(jobs)
        
        # Set delay to keep batch running
        provider.set_mock_delay("job-1", 10.0)
        
        with pytest.raises(ValueError) as exc_info:
            batch.get_results()
        
        assert "Cannot get results for batch in status:" in str(exc_info.value)
    
    def test_batch_get_results_success(self):
        """Test getting results from completed batch."""
        provider = MockProvider(auto_register=False)
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        ]
        
        batch = provider.create_batch(jobs)
        
        # Wait for completion (no delay set, so should be quick)
        time.sleep(0.1)
        batch.get_status()
        
        if batch.is_complete:
            results = batch.get_results()
            assert len(results) == 1
            assert results[0].job_id == "job-1"