"""Tests for BatchRun."""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from batchata.core import Batch, BatchRun, Job, JobResult
from batchata.providers import ProviderRegistry
from tests.mocks import MockProvider
from batchata.utils.state import BatchState


class TestBatchRun:
    """Tests for BatchRun class."""
    
    def setup_method(self):
        """Setup for each test."""
        ProviderRegistry.clear()
        # Register mock provider
        MockProvider(auto_register=True)
    
    def test_init(self):
        """Test BatchRun initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results", max_concurrent=5)
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Test"}])
            )
            
            run = BatchRun(batch.config)
            
            assert run.batch_id.startswith("batch-run-")
            assert run.cost_tracker.limit_usd is None
            assert len(run.pending_jobs) == 1
            assert len(run.completed_results) == 0
            assert run.results_dir.exists()
    
    def test_init_with_cost_limit(self):
        """Test initialization with cost limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .add_cost_limit(usd=50.0)
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Test"}])
            )
            
            run = BatchRun(batch.config)
            assert run.cost_tracker.limit_usd == 50.0
    
    def test_start_and_wait(self):
        """Test starting and waiting for completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results", max_concurrent=5)
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
                .add_job(messages=[{"role": "user", "content": "Job 2"}])
            )
            
            run = BatchRun(batch.config)
            
            # Should not be complete initially
            assert not run.is_complete
            
            # Start execution
            run.start()
            
            # Should not be able to start again
            with pytest.raises(RuntimeError, match="already started"):
                run.start()
            
            # Wait for completion
            run.wait(timeout=2.0)
            
            # Should be complete
            assert run.is_complete
            
            # Check results
            results = run.results()
            assert len(results) == 2
            assert all(r.is_success for r in results.values())
    
    def test_status(self):
        """Test status reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
                .add_job(messages=[{"role": "user", "content": "Job 2"}])
                .add_job(messages=[{"role": "user", "content": "Job 3"}])
            )
            
            run = BatchRun(batch.config)
            
            # Initial status
            status = run.status()
            assert status["total"] == 3
            assert status["pending"] == 3
            assert status["active"] == 0
            assert status["completed"] == 0
            assert status["failed"] == 0
            assert status["is_complete"] is False
            
            # Start and wait a bit
            run.start()
            time.sleep(0.5)
            
            # Final status after completion
            run.wait(timeout=2.0)
            status = run.status()
            assert status["completed"] == 3
            assert status["pending"] == 0
            assert status["is_complete"] is True
            assert status["cost_usd"] > 0
    
    def test_results_saved_to_files(self):
        """Test that results are saved to individual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            
            batch = (
                Batch(f"{tmpdir}/state.json", str(results_dir))
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Test"}])
            )
            
            run = BatchRun(batch.config)
            run.start()
            run.wait(timeout=2.0)
            
            # Check result files
            result_files = list(results_dir.glob("*.json"))
            assert len(result_files) == 1
            
            # Load and verify result
            with open(result_files[0]) as f:
                saved_result = json.load(f)
            
            assert saved_result["response"] == "Mock response for job job-1"
            assert saved_result["cost_usd"] == 0.005
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        callback_stats = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .defaults(model="mock-model-basic")
                .on_progress(lambda stats: callback_stats.append(stats))
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
                .add_job(messages=[{"role": "user", "content": "Job 2"}])
            )
            
            run = BatchRun(batch.config)
            run.start()
            run.wait(timeout=2.0)
            
            # Should have received progress updates
            assert len(callback_stats) > 0
            
            # Final update should show completion
            final_stats = callback_stats[-1]
            assert final_stats["completed"] == 2
            assert final_stats["is_complete"] is True
    
    def test_failed_jobs(self):
        """Test handling of failed jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up a job to fail
            provider = ProviderRegistry.get_provider("mock-model-basic")
            provider.set_mock_failure(
                "job-1", 
                Exception("Simulated failure")
            )
            
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "This will fail"}])
                .add_job(messages=[{"role": "user", "content": "This will succeed"}])
            )
            
            run = BatchRun(batch.config)
            run.start()
            run.wait(timeout=2.0)
            
            # Check results
            results = run.results()
            failed = run.get_failed_jobs()
            
            # One should succeed, one should fail
            assert len(results) == 1
            assert len(failed) == 1
            assert "Simulated failure" in list(failed.values())[0]
    
    def test_state_persistence(self):
        """Test saving and resuming from state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = f"{tmpdir}/state.json"
            
            # Create and partially execute a batch
            batch = (
                Batch(state_file, f"{tmpdir}/results")
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
                .add_job(messages=[{"role": "user", "content": "Job 2"}])
                .add_job(messages=[{"role": "user", "content": "Job 3"}])
            )
            
            # Add delay to second job so we can interrupt
            provider = ProviderRegistry.get_provider("mock-model-basic")
            provider.set_mock_delay("job-2", 5.0)  # Long delay
            
            run1 = BatchRun(batch.config)
            batch_id = run1.batch_id
            run1.start()
            
            # Wait for first job to complete
            time.sleep(0.5)
            
            # Shutdown before completion
            run1.shutdown(wait_for_active=False)
            
            # State should be saved
            assert Path(state_file).exists()
            
            # Create new run that should resume
            batch2 = Batch(state_file, f"{tmpdir}/results")
            run2 = BatchRun(batch2.config)
            
            # Should have same batch ID
            assert run2.batch_id == batch_id
            
            # Should have some completed results
            assert len(run2.completed_results) >= 1
            
            # Should have remaining pending jobs
            assert len(run2.pending_jobs) >= 1
    
    def test_cost_limit_enforcement(self):
        """Test that cost limits are enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set very low cost limit
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .add_cost_limit(usd=0.01)  # Only allows 1 job
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
                .add_job(messages=[{"role": "user", "content": "Job 2"}])
                .add_job(messages=[{"role": "user", "content": "Job 3"}])
            )
            
            run = BatchRun(batch.config)
            run.start()
            run.wait(timeout=2.0)
            
            # Should only complete some jobs due to cost limit
            results = run.results()
            assert len(results) <= 2  # At most 2 jobs at $0.01 each
            
            # Cost should not exceed limit
            status = run.status()
            assert status["cost_usd"] <= 0.01
    
    def test_shutdown_signal_handler(self):
        """Test graceful shutdown via signal handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results")
                .defaults(model="mock-model-basic")
                .add_job(messages=[{"role": "user", "content": "Job 1"}])
            )
            
            run = BatchRun(batch.config)
            
            # Mock signal handler to test it's set up
            with patch('signal.signal') as mock_signal:
                run2 = BatchRun(batch.config)
                
                # Should have registered handlers
                assert mock_signal.call_count >= 2  # SIGINT and SIGTERM
    
    def test_concurrent_execution(self):
        """Test that jobs are executed concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create batch with multiple jobs
            batch = (
                Batch(f"{tmpdir}/state.json", f"{tmpdir}/results", max_concurrent=3)
                .defaults(model="mock-model-basic")
            )
            
            # Add jobs with delays
            provider = ProviderRegistry.get_provider("mock-model-basic")
            for i in range(6):
                job_id = f"job-{i+1}"
                batch.add_job(messages=[{"role": "user", "content": f"Job {i+1}"}])
                provider.set_mock_delay(job_id, 0.5)  # Each job takes 0.5s
            
            start_time = time.time()
            run = BatchRun(batch.config)
            run.start()
            run.wait(timeout=10.0)
            elapsed = time.time() - start_time
            
            # With max_concurrent=3, 6 jobs at 0.5s each should take ~1s
            # (2 batches of 3 jobs each)
            # Without concurrency it would take 3s (6 * 0.5s)
            assert elapsed < 2.0  # Allow some overhead
            
            # All jobs should complete
            assert len(run.results()) == 6