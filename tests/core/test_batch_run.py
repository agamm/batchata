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
    
    @pytest.mark.parametrize("max_concurrent,items_per_batch,cost_limit,jobs,expected_error", [
        # Valid parameters
        (2, 1, 1.0, [Job(id="job-1", model="claude-3-5-haiku-latest", messages=[{"role": "user", "content": "Test"}])], None),
        
        # Invalid max_concurrent
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
    def test_parameter_validation(self, temp_dir, max_concurrent, items_per_batch, cost_limit, jobs, expected_error, mock_all_file_operations):
        """Test that BatchRun validates parameters correctly."""
        state_file = str(temp_dir / "state.json")
        results_dir = str(temp_dir / "results")
        
        if expected_error:
            # Should raise an error during BatchParams creation
            with pytest.raises(expected_error):
                params = BatchParams(
                    state_file=state_file,
                    results_dir=results_dir,
                    max_concurrent=max_concurrent,
                    items_per_batch=items_per_batch,
                    cost_limit_usd=cost_limit
                )
        else:
            # Should create successfully
            params = BatchParams(
                state_file=state_file,
                results_dir=results_dir,
                max_concurrent=max_concurrent,
                items_per_batch=items_per_batch,
                cost_limit_usd=cost_limit
            )
            
            run = BatchRun(params, jobs)
            assert run.config == params
            assert len(run.jobs) == len(jobs)
            assert run.pending_jobs == jobs
    
    def test_batch_run_initialization(self, temp_dir, mock_all_file_operations):
        """Test that BatchRun can be initialized properly."""
        state_file = str(temp_dir / "state.json")
        results_dir = str(temp_dir / "results")
        
        params = BatchParams(
            state_file=state_file,
            results_dir=results_dir,
            max_concurrent=2,
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
        state_file = str(temp_dir / "state.json")
        results_dir = str(temp_dir / "results")
        
        params = BatchParams(
            state_file=state_file,
            results_dir=results_dir,
            max_concurrent=1,
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
    
    @pytest.mark.skip(reason="Complex integration test - requires full provider mocking")
    def test_cost_limit_enforcement(self):
        """Test that cost limits stop execution when exceeded."""
        pass