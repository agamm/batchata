"""Test time limit functionality."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from batchata.core.batch import Batch
from tests.mocks.mock_provider import MockProvider


def test_time_limit_basic():
    """Test that time limit causes jobs to fail with time limit message."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            # Create a slow mock provider (2 second delay)
            mock_provider = MockProvider(delay=2.0)
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            
            batch = (Batch(results_dir=str(results_dir))
                    .set_default_params(model="mock-model-advanced")
                    .add_time_limit(seconds=1))  # 1 second time limit, job takes 2 seconds
            
            batch.add_job(messages=[{"role": "user", "content": "Test message"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should complete quickly due to time limit (not wait full 2 seconds)
            assert elapsed < 1.5, f"Expected time limit around 1s, took {elapsed:.2f}s"
            
            # Job should be marked as failed due to time limit
            results = run.results()
            job_id = batch.jobs[0].id
            
            # Check if job is in failed results
            assert job_id in results["failed"], "Job should be in failed results"
            assert "Time limit" in results["failed"][job_id].error


def test_add_time_limit_fluent_api():
    """Test the fluent API for setting time limits."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        batch = Batch(results_dir=str(results_dir))
        
        # Test seconds
        batch.add_time_limit(seconds=30)
        assert batch.config.time_limit_seconds == 30
        
        # Test minutes
        batch.add_time_limit(minutes=2)
        assert batch.config.time_limit_seconds == 120
        
        # Test hours
        batch.add_time_limit(hours=1)
        assert batch.config.time_limit_seconds == 3600
        
        # Test combination
        batch.add_time_limit(hours=1, minutes=30, seconds=15)
        assert batch.config.time_limit_seconds == 5415  # 3600 + 1800 + 15


def test_time_limit_validation():
    """Test time limit validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        batch = Batch(results_dir=str(results_dir))
        
        # Test minimum time limit
        try:
            batch.add_time_limit(seconds=5)  # Less than 10 seconds
            batch.config.__post_init__()  # Trigger validation
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "at least 10 seconds" in str(e)
        
        # Test maximum time limit
        try:
            batch.add_time_limit(hours=25)  # More than 24 hours
            batch.config.__post_init__()  # Trigger validation
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "at most 24 hours" in str(e)
        
        # Test empty time limit
        try:
            batch.add_time_limit()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Must specify at least one" in str(e)


def test_time_limit_watchdog_timing():
    """Test that time limit watchdog checks every second and stops promptly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            # Create mock provider with 4 second delay
            mock_provider = MockProvider(delay=4.0)
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            
            # Single job batch with 2 second time limit
            batch = (Batch(results_dir=str(results_dir))
                    .set_default_params(model="mock-model-advanced")
                    .add_time_limit(seconds=2))  # 2 second time limit
            
            # Add a single job that takes 4 seconds
            batch.add_job(messages=[{"role": "user", "content": "Test message"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should hit time limit around 2 seconds (not wait full 4 seconds)
            assert 1.5 < elapsed < 2.5, f"Expected time limit around 2s, took {elapsed:.2f}s"
            
            # Check that job was marked as failed/cancelled due to time limit
            results = run.results()
            
            # Job should not complete (takes 4s, time limit at 2s)
            assert len(results["completed"]) == 0, "Job should not have completed"
            
            # Job should be failed or cancelled due to time limit
            total_failed_cancelled = len(results["failed"]) + len(results["cancelled"])
            assert total_failed_cancelled == 1, f"Expected 1 failed/cancelled job, got {total_failed_cancelled}"
            
            # Check error messages
            all_errors = []
            for job_result in results["failed"].values():
                all_errors.append(job_result.error or "")
            for job_result in results["cancelled"].values():
                all_errors.append(job_result.error or "")
            
            assert any("time limit" in err.lower() for err in all_errors), f"Expected time limit error, got: {all_errors}"


def test_time_limit_during_polling():
    """Test time limit during polling phase."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            # Mock provider with long polling time
            mock_provider = MockProvider(delay=5.0)  # 5 second batch processing
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            
            batch = (Batch(results_dir=str(results_dir))
                    .set_default_params(model="mock-model-advanced")
                    .add_time_limit(seconds=2))  # Time limit during polling
            
            batch.add_job(messages=[{"role": "user", "content": "Test"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should hit time limit around 2 seconds (not wait full 5 seconds)
            assert 1.5 < elapsed < 3.0, f"Expected time limit around 2s, took {elapsed:.2f}s"
            
            # Job should be cancelled/failed due to time limit
            results = run.results()
            
            # Either failed or cancelled
            total_failed_cancelled = len(results["failed"]) + len(results["cancelled"])
            assert total_failed_cancelled == 1
            
            # Check error messages
            all_errors = []
            for job_result in results["failed"].values():
                all_errors.append(job_result.error or "")
            for job_result in results["cancelled"].values():
                all_errors.append(job_result.error or "")
            
            assert any("time limit" in err.lower() for err in all_errors)

def test_multi_batch_time_limit_cancellation():
    """Test that all batches are cancelled when time limit is reached."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            # Mock provider with 5 second delay
            mock_provider = MockProvider(delay=5.0)
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            
            # Track cancellation calls
            cancel_calls = []
            original_cancel = mock_provider.cancel_batch
            def track_cancel_batch(batch_id):
                cancel_calls.append(batch_id)
                return original_cancel(batch_id)
            mock_provider.cancel_batch = track_cancel_batch
            
            # Create batch with multiple concurrent batches
            batch = (Batch(results_dir=str(results_dir), max_parallel_batches=3, items_per_batch=1)
                    .set_default_params(model="mock-model-advanced")
                    .add_time_limit(seconds=2))  # Time limit before batches complete
            
            # Add 3 jobs (will create 3 concurrent batches)
            for i in range(3):
                batch.add_job(messages=[{"role": "user", "content": f"Test {i}"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should hit time limit around 2 seconds
            assert 1.5 < elapsed < 3.0, f"Expected time limit around 2s, took {elapsed:.2f}s"
            
            # All batches should have been cancelled
            assert len(cancel_calls) == 3, f"Expected 3 batch cancellations, got {len(cancel_calls)}"
            
            # All jobs should be marked as failed due to time limit
            results = run.results()
            assert len(results["completed"]) == 0, "No jobs should have completed"
            assert len(results["failed"]) == 3, "All 3 jobs should be failed due to time limit"
            
            # All errors should be time limit-related
            for job_result in results["failed"].values():
                assert "time limit" in job_result.error.lower(), f"Expected time limit error, got: {job_result.error}"