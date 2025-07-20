"""Test on_error callback functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from batchata.core.batch import Batch
from batchata.core.job_result import JobResult
from tests.mocks.mock_provider import MockProvider


def test_on_error_callback_invoked_on_failure():
    """Test that on_error callback is called with correct parameters when a job fails."""
    # Create a mock callback
    error_callback = Mock()
    
    # Use temporary directory to avoid state file conflicts
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        # Create batch with mock provider
        with patch('batchata.core.batch_run.get_provider') as mock_get, \
             patch('batchata.core.batch.get_provider') as mock_get2:
            mock_provider = MockProvider(delay=0.0)
            mock_get.return_value = mock_provider
            mock_get2.return_value = mock_provider
            batch = (Batch(results_dir=str(results_dir))
                    .set_default_params(model="mock-model-advanced"))
            
            batch.add_job(
                messages=[{"role": "user", "content": "Test message"}],
                on_error=error_callback
            )
            
            job = batch.jobs[0]
            
            # Modify the mock provider to generate a failed result
            def generate_failed_results(batch_id: str) -> None:
                """Generate a failed result for testing."""
                batch_info = mock_provider.batches[batch_id]
                jobs = batch_info["jobs"]
                
                # Create a failed result
                failed_result = JobResult(
                    job_id=jobs[0].id,
                    raw_response=None,
                    parsed_response=None,
                    error="Mock API failure for testing",
                    cost_usd=0.0,
                    input_tokens=0,
                    output_tokens=0
                )
                
                batch_info["results"] = [failed_result]
            
            # Replace the result generation method
            mock_provider._generate_results = generate_failed_results
            
            # Run the batch
            run = batch.run()
            
            # Verify the callback was called exactly once
            error_callback.assert_called_once()
            
            # Verify the callback was called with correct parameters: (job, error_message)
            call_args = error_callback.call_args[0]
            assert call_args[0].id == job.id  # job object
            assert call_args[1] == "Mock API failure for testing"  # error message
            
            # Verify the job failed
            failed_jobs = run.get_failed_jobs()
            assert job.id in failed_jobs
            assert failed_jobs[job.id] == "Mock API failure for testing"