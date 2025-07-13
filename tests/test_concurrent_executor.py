"""Tests for ConcurrentExecutor."""

import pytest
import threading
import time
from concurrent.futures import Future
from unittest.mock import Mock, patch

from batchata.core import Job, JobResult, ConcurrentExecutor, ExecutorStats
from batchata.providers import BatchRequest
from tests.mocks import MockProvider
from batchata.utils import CostTracker


class TestConcurrentExecutor:
    """Tests for ConcurrentExecutor."""
    
    def test_init(self):
        """Test executor initialization."""
        executor = ConcurrentExecutor(max_concurrent=5)
        
        assert executor.max_concurrent == 5
        assert executor.get_active_count() == 0
        assert isinstance(executor.cost_tracker, CostTracker)
        assert executor.cost_tracker.limit_usd is None
    
    def test_init_with_cost_tracker(self):
        """Test initialization with cost tracker."""
        tracker = CostTracker(limit_usd=100.0)
        executor = ConcurrentExecutor(max_concurrent=3, cost_tracker=tracker)
        
        assert executor.cost_tracker is tracker
        assert executor.cost_tracker.limit_usd == 100.0
    
    def test_submit_basic(self):
        """Test basic job submission."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=5)
        
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        ]
        
        future = executor.submit_if_allowed(provider, jobs)
        assert future is not None
        assert isinstance(future, Future)
        assert executor.get_active_count() == 1
    
    def test_submit_at_capacity(self):
        """Test submission when at capacity."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=2)
        
        # Add delays to keep batches active
        provider.set_mock_delay("job-1", 1.0)
        provider.set_mock_delay("job-2", 1.0)
        
        jobs1 = [Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])]
        jobs2 = [Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "2"}])]
        jobs3 = [Job(id="job-3", model="mock-model-basic", messages=[{"role": "user", "content": "3"}])]
        
        # Submit two batches (at capacity)
        future1 = executor.submit_if_allowed(provider, jobs1)
        future2 = executor.submit_if_allowed(provider, jobs2)
        assert future1 is not None
        assert future2 is not None
        
        # Third submission should be rejected
        future3 = executor.submit_if_allowed(provider, jobs3)
        assert future3 is None
        
        # Cleanup
        executor.shutdown(wait=False)
    
    def test_submit_cost_limit(self):
        """Test submission with cost limit."""
        provider = MockProvider(auto_register=False)
        tracker = CostTracker(limit_usd=0.02)  # $0.02 limit
        executor = ConcurrentExecutor(max_concurrent=5, cost_tracker=tracker)
        
        # MockProvider estimates $0.01 per job
        jobs1 = [Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])]
        jobs2 = [Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "2"}])]
        jobs3 = [Job(id="job-3", model="mock-model-basic", messages=[{"role": "user", "content": "3"}])]
        
        # First two should succeed
        future1 = executor.submit_if_allowed(provider, jobs1)
        future2 = executor.submit_if_allowed(provider, jobs2)
        assert future1 is not None
        assert future2 is not None
        
        # Third should be rejected due to cost limit
        future3 = executor.submit_if_allowed(provider, jobs3)
        assert future3 is None
    
    def test_get_completed(self):
        """Test getting completed results."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=5)
        
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Test"}])
        ]
        
        # Submit batch
        future = executor.submit_if_allowed(provider, jobs)
        assert future is not None
        
        # Initially no completed
        completed = executor.get_completed()
        assert len(completed) == 0
        
        # Wait for completion
        time.sleep(0.2)
        
        # Should have completed results
        completed = executor.get_completed()
        assert len(completed) == 1
        
        batch, results = completed[0]
        assert isinstance(batch, BatchRequest)
        assert len(results) == 1
        assert results[0].job_id == "job-1"
        
        # Second call should return empty (already collected)
        completed = executor.get_completed()
        assert len(completed) == 0
    
    def test_wait_for_capacity(self):
        """Test waiting for capacity."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=1)
        
        # Add delay to keep batch active
        provider.set_mock_delay("job-1", 0.5)
        
        jobs = [Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])]
        
        # Submit to fill capacity
        executor.submit_if_allowed(provider, jobs)
        assert executor.get_active_count() == 1
        
        # Should timeout waiting for capacity
        has_capacity = executor.wait_for_capacity(timeout=0.1)
        assert has_capacity is False
        
        # Wait for batch to complete
        time.sleep(0.6)
        executor.get_completed()
        
        # Now should have capacity
        has_capacity = executor.wait_for_capacity(timeout=0.1)
        assert has_capacity is True
    
    def test_get_stats(self):
        """Test getting execution statistics."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=5)
        
        # Initial stats
        stats = executor.get_stats()
        assert stats.submitted_batches == 0
        assert stats.completed_batches == 0
        assert stats.active_batches == 0
        
        # Submit some jobs
        jobs1 = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}]),
            Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "2"}])
        ]
        
        executor.submit_if_allowed(provider, jobs1)
        
        # Check updated stats
        stats = executor.get_stats()
        assert stats.submitted_batches == 1
        assert stats.active_batches == 1
        assert stats.total_jobs == 2
        
        # Wait for completion
        time.sleep(0.2)
        executor.get_completed()
        
        # Final stats
        stats = executor.get_stats()
        assert stats.completed_batches == 1
        assert stats.active_batches == 0
        assert stats.completed_jobs == 2
        assert stats.failed_jobs == 0
    
    def test_shutdown(self):
        """Test executor shutdown."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=5)
        
        # Add delay to test shutdown during execution
        provider.set_mock_delay("job-1", 2.0)
        
        jobs = [Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])]
        
        # Submit batch
        future = executor.submit_if_allowed(provider, jobs)
        assert future is not None
        
        # Shutdown without waiting
        executor.shutdown(wait=False)
        
        # Should reject new submissions
        future2 = executor.submit_if_allowed(provider, jobs)
        assert future2 is None
    
    def test_context_manager(self):
        """Test executor as context manager."""
        provider = MockProvider(auto_register=False)
        
        with ConcurrentExecutor(max_concurrent=5) as executor:
            jobs = [Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])]
            future = executor.submit_if_allowed(provider, jobs)
            assert future is not None
        
        # Executor should be shut down after context
        assert executor._shutdown is True
    
    def test_concurrent_submissions(self):
        """Test truly concurrent submissions."""
        provider = MockProvider(auto_register=False)
        executor = ConcurrentExecutor(max_concurrent=10)
        
        results = []
        lock = threading.Lock()
        
        def submit_job(job_id):
            job = Job(id=job_id, model="mock-model-basic", messages=[{"role": "user", "content": job_id}])
            future = executor.submit_if_allowed(provider, [job])
            
            with lock:
                results.append(future is not None)
        
        # Submit 10 jobs concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=submit_job, args=(f"job-{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All should have been submitted
        assert all(results)
        assert executor.get_stats().submitted_batches == 10
        
        # Cleanup
        executor.shutdown(wait=False)
    
    def test_cost_tracking_accuracy(self):
        """Test that cost tracking is accurate."""
        provider = MockProvider(auto_register=False)
        tracker = CostTracker(limit_usd=1.0)
        executor = ConcurrentExecutor(max_concurrent=5, cost_tracker=tracker)
        
        # Set custom costs
        job1 = Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "1"}])
        job2 = Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "2"}])
        
        # Set actual costs different from estimates
        provider.set_mock_response("job-1", JobResult(
            job_id="job-1",
            response="Response 1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.15  # Actual cost
        ))
        
        provider.set_mock_response("job-2", JobResult(
            job_id="job-2",
            response="Response 2",
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.25  # Actual cost
        ))
        
        # Submit jobs
        executor.submit_if_allowed(provider, [job1])
        executor.submit_if_allowed(provider, [job2])
        
        # Wait for completion
        time.sleep(0.2)
        executor.get_completed()
        
        # Check actual cost tracked
        stats = tracker.get_stats()
        assert abs(stats.total_cost_usd - 0.40) < 0.0001  # 0.15 + 0.25