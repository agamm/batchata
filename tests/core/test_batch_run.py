"""BatchRun tests - simplified version focusing on parameter validation."""

import pytest
from unittest.mock import patch, MagicMock

from batchata.core.batch_run import BatchRun
from batchata.core.batch_params import BatchParams
from batchata.core.job import Job


@pytest.fixture
def mock_all_file_operations():
    """Mock all file operations that could cause hanging."""
    with patch('batchata.utils.StateManager') as mock_state_manager_class, \
         patch('pathlib.Path.mkdir'), \
         patch('pathlib.Path.exists', return_value=False), \
         patch('shutil.rmtree'), \
         patch('time.sleep'):
        
        # Mock StateManager to avoid file I/O
        mock_state_manager = MagicMock()
        mock_state_manager.load.return_value = None
        mock_state_manager.save.return_value = None
        mock_state_manager.clear.return_value = None
        mock_state_manager_class.return_value = mock_state_manager
        
        yield mock_state_manager


class TestBatchRun:
    """BatchRun tests focusing on parameter validation."""
    
    @pytest.mark.parametrize("max_parallel_batches,items_per_batch,cost_limit,jobs,expected_error", [
        # Valid parameters
        (2, 1, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], None),
        
        # Invalid max_parallel_batches
        (0, 1, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], ValueError),
        (-1, 1, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], ValueError),
        
        # Invalid items_per_batch
        (2, 0, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], ValueError),
        (2, -1, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], ValueError),
        
        # Invalid cost_limit
        (2, 1, -1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], ValueError),
        
        # Empty jobs list
        (2, 1, 1.0, [], None),
    ])
    def test_parameter_validation(self, temp_dir, max_parallel_batches, items_per_batch, cost_limit, jobs, expected_error, mock_all_file_operations):
        """Test that BatchRun validates parameters correctly."""
        results_dir = str(temp_dir / "results")
        
        if expected_error:
            # Should raise an error during BatchParams creation
            with pytest.raises(expected_error):
                params = BatchParams(
                    state_file=str(temp_dir / "state.json"),
                    results_dir=results_dir,
                    max_parallel_batches=max_parallel_batches,
                    items_per_batch=items_per_batch,
                    cost_limit_usd=cost_limit
                )
        else:
            # Should create successfully
            params = BatchParams(
                state_file=str(temp_dir / "state.json"),
                results_dir=results_dir,
                max_parallel_batches=max_parallel_batches,
                items_per_batch=items_per_batch,
                cost_limit_usd=cost_limit
            )
            
            run = BatchRun(params, jobs)
            assert run.config == params
            assert len(run.jobs) == len(jobs)
            assert run.pending_jobs == jobs
    
    def test_batch_run_initialization(self, temp_dir, mock_all_file_operations):
        """Test that BatchRun can be initialized properly."""
        results_dir = str(temp_dir / "results")
        
        params = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=2,
            items_per_batch=1,
            cost_limit_usd=1.0
        )
        
        jobs = [
            Job(id="job-1", 
                model="claude-3-5-haiku-latest",
                messages=[{"role": "user", "content": "Test"}])
        ]
        
        run = BatchRun(params, jobs)
        
        # Test basic properties
        assert run.config == params
        assert len(run.jobs) == 1
        assert len(run.pending_jobs) == 1
        assert run.pending_jobs[0].id == "job-1"
        assert run._started is False
        assert run._start_time is None
        assert run.cost_tracker.limit_usd == 1.0
        
        # Test status method
        status = run.status()
        assert status['total'] == 1
        assert status['pending'] == 1
        assert status['completed'] == 0
        assert status['failed'] == 0
        assert status['is_complete'] is False
    
    def test_progress_callback_setup(self, temp_dir, mock_all_file_operations):
        """Test that progress callbacks can be set up."""
        results_dir = str(temp_dir / "results")
        
        params = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=1,
            items_per_batch=1,
            cost_limit_usd=1.0
        )
        
        jobs = [
            Job(id="job-1", 
                model="claude-3-5-haiku-latest",
                messages=[{"role": "user", "content": "Test"}])
        ]
        
        def progress_callback(stats, elapsed_time):
            pass
        
        run = BatchRun(params, jobs)
        run.set_on_progress(progress_callback, interval=2.0)
        
        # Verify callback is set
        assert run._progress_callback == progress_callback
        assert run._progress_interval == 2.0
    
    @pytest.mark.skip(reason="Complex integration test - requires full provider mocking")
    def test_job_execution_flow(self):
        """Test that all jobs are processed correctly."""
        # This test would require complex mocking of the entire execution flow
        # including provider registry, file operations, and state management
        pass
    
    @pytest.mark.skip(reason="Complex integration test - requires full provider mocking")
    def test_progress_callbacks(self):
        """Test that progress callbacks are invoked correctly."""
        pass
    
    def test_cost_reservation_blocks_then_frees_budget(self, temp_dir, mock_all_file_operations):
        """Test that cost reservation and adjustment works correctly across sequential batches."""
        from unittest.mock import Mock, patch
        from batchata.core.job_result import JobResult
        
        results_dir = str(temp_dir / "results")
        
        # Create three jobs - first reserves most budget, others proceed after adjustment
        jobs = [
            Job(id="job-1", model="claude-3-5-haiku-20241022", messages=[{"role": "user", "content": "Test 1"}]),
            Job(id="job-2", model="claude-3-5-haiku-20241022", messages=[{"role": "user", "content": "Test 2"}]),
            Job(id="job-3", model="claude-3-5-haiku-20241022", messages=[{"role": "user", "content": "Test 3"}])
        ]
        
        params = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=1,  # Sequential execution to test reserve->adjust->reserve flow
            items_per_batch=1,  # One job per batch
            cost_limit_usd=100.0
        )
        
        run = BatchRun(params, jobs)
        
        # Track what happens with each batch
        execution_results = []
        
        # Mock provider that simulates different cost patterns
        mock_provider = Mock()
        
        def mock_estimate_cost(jobs):
            job_id = jobs[0].id
            if job_id == "job-1":
                # First batch: high estimate that reserves most budget
                return 70.0
            elif job_id == "job-2":
                # Second batch: moderate estimate
                return 40.0
            else:
                # Third batch: small estimate
                return 20.0
        
        def mock_create_batch(jobs, raw_files_dir=None):
            job_id = jobs[0].id
            execution_results.append(f"created_{job_id}")
            # Return tuple: (batch_id, job_mapping)
            job_mapping = {job.id: job for job in jobs}
            return f"batch_{job_id}", job_mapping
        
        def mock_get_batch_status(batch_id):
            return "complete", None
        
        def mock_get_batch_results(batch_id, job_mapping, raw_files_dir=None):
            if "job-1" in batch_id:
                # First batch: actual cost much lower than estimate
                execution_results.append(f"completed_job-1_actual_30")
                return [JobResult(
                    job_id="job-1",
                    raw_response="response1",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=30.0,  # Much lower than $70 estimate
                    error=None
                )]
            elif "job-2" in batch_id:
                # Second batch: should not reach here if properly blocked
                execution_results.append(f"completed_job-2_actual_40")
                return [JobResult(
                    job_id="job-2", 
                    raw_response="response2",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=40.0,
                    error=None
                )]
            else:
                # Third batch: should be able to proceed after job-1 adjustment
                execution_results.append(f"completed_job-3_actual_20")
                return [JobResult(
                    job_id="job-3", 
                    raw_response="response3",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=20.0,
                    error=None
                )]
        
        mock_provider.estimate_cost = mock_estimate_cost
        mock_provider.create_batch = mock_create_batch
        mock_provider.get_batch_status = mock_get_batch_status
        mock_provider.get_batch_results = mock_get_batch_results
        mock_provider.validate_job = Mock()
        
        # Mock the provider registry to return our mock
        with patch('batchata.core.batch_run.get_provider', return_value=mock_provider):
            # Start the batch run
            run.start()
            
            # Verify execution patterns
            results = run.results()
            failed = run.get_failed_jobs()
            
            # Job-1 should have executed (reserved 70, used 30)
            assert "job-1" in results
            assert results["job-1"].cost_usd == 30.0
            
            # Job-2 should have executed after job-1 freed budget
            assert "job-2" in results
            assert results["job-2"].cost_usd == 40.0
            
            # Job-3 should have executed after job-2
            assert "job-3" in results
            assert results["job-3"].cost_usd == 20.0
            
            # Final cost should be 30 + 40 + 20 = 90
            assert run.cost_tracker.used_usd == 90.0
            assert run.cost_tracker.remaining() == 10.0
            
            # Verify the execution flow
            assert "created_job-1" in execution_results
            assert "created_job-2" in execution_results
            assert "created_job-3" in execution_results
            assert "completed_job-1_actual_30" in execution_results 
            assert "completed_job-2_actual_40" in execution_results
            assert "completed_job-3_actual_20" in execution_results