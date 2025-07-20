"""Test timeout functionality."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from batchata.core.batch import Batch
from tests.mocks.mock_provider import MockProvider


def test_timeout_basic():
    """Test that timeout causes jobs to fail with timeout message."""
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
                    .set_timeout(seconds=1))  # 1 second timeout, job takes 2 seconds
            
            batch.add_job(messages=[{"role": "user", "content": "Test message"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should complete quickly due to timeout (not wait full 2 seconds)
            assert elapsed < 1.5, f"Expected timeout around 1s, took {elapsed:.2f}s"
            
            # Job should be marked as failed due to timeout
            failed_jobs = run.get_failed_jobs()
            job_id = batch.jobs[0].id
            assert job_id in failed_jobs
            assert "Timeout" in failed_jobs[job_id]


def test_set_timeout_fluent_api():
    """Test the fluent API for setting timeouts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        batch = Batch(results_dir=str(results_dir))
        
        # Test seconds
        batch.set_timeout(seconds=30)
        assert batch.config.timeout_seconds == 30
        
        # Test minutes
        batch.set_timeout(minutes=2)
        assert batch.config.timeout_seconds == 120
        
        # Test hours
        batch.set_timeout(hours=1)
        assert batch.config.timeout_seconds == 3600
        
        # Test combination
        batch.set_timeout(hours=1, minutes=30, seconds=15)
        assert batch.config.timeout_seconds == 5415  # 3600 + 1800 + 15


def test_timeout_validation():
    """Test timeout validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        batch = Batch(results_dir=str(results_dir))
        
        # Test minimum timeout
        try:
            batch.set_timeout(seconds=5)  # Less than 10 seconds
            batch.config.__post_init__()  # Trigger validation
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "at least 10 seconds" in str(e)
        
        # Test maximum timeout
        try:
            batch.set_timeout(hours=25)  # More than 24 hours
            batch.config.__post_init__()  # Trigger validation
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "at most 24 hours" in str(e)
        
        # Test empty timeout
        try:
            batch.set_timeout()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Must specify at least one" in str(e)


def test_timeout_watchdog_timing():
    """Test that timeout watchdog checks every second and stops promptly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            # Create mock provider with 4 second delay
            mock_provider = MockProvider(delay=4.0)
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            
            # Single job batch with 2 second timeout
            batch = (Batch(results_dir=str(results_dir))
                    .set_default_params(model="mock-model-advanced")
                    .set_timeout(seconds=2))  # 2 second timeout
            
            # Add a single job that takes 4 seconds
            batch.add_job(messages=[{"role": "user", "content": "Test message"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should timeout around 2 seconds (not wait full 4 seconds)
            assert 1.5 < elapsed < 2.5, f"Expected timeout around 2s, took {elapsed:.2f}s"
            
            # Check that job was marked as failed/cancelled due to timeout
            results = run.results()
            failed_jobs = run.get_failed_jobs()
            cancelled_jobs = getattr(run, 'cancelled_jobs', {})
            
            # Job should not complete (takes 4s, timeout at 2s)
            assert len(results) == 0, "Job should not have completed"
            
            # Job should be failed or cancelled due to timeout
            assert len(failed_jobs) + len(cancelled_jobs) == 1
            all_errors = list(failed_jobs.values()) + list(cancelled_jobs.values())
            assert any("timeout" in err.lower() for err in all_errors), f"Expected timeout error, got: {all_errors}"


def test_timeout_during_polling():
    """Test timeout during polling phase."""
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
                    .set_timeout(seconds=2))  # Timeout during polling
            
            batch.add_job(messages=[{"role": "user", "content": "Test"}])
            
            # Run the batch
            start_time = time.time()
            run = batch.run()
            elapsed = time.time() - start_time
            
            # Should timeout around 2 seconds (not wait full 5 seconds)
            assert 1.5 < elapsed < 3.0, f"Expected timeout around 2s, took {elapsed:.2f}s"
            
            # Job should be cancelled/failed due to timeout
            failed_jobs = run.get_failed_jobs()
            cancelled_jobs = getattr(run, 'cancelled_jobs', {})
            
            # Either failed or cancelled
            assert len(failed_jobs) + len(cancelled_jobs) == 1
            all_errors = list(failed_jobs.values()) + list(cancelled_jobs.values())
            assert any("timeout" in err.lower() for err in all_errors)