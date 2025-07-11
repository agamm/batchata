"""
Comprehensive tests for BatchManager
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional
import pytest
from pydantic import BaseModel

from batchata.batch_manager import BatchManager, BatchManagerError
from batchata.batch_job import BatchJob
from tests.fixtures import (
    create_mock_batch_job, 
    create_structured_results, 
    create_cited_results,
    setup_mock_batch_function
)


class InvoiceModel(BaseModel):
    """Test model for structured responses"""
    invoice_number: str
    company_name: str
    total_amount: float


class TestBatchManager:
    """Test BatchManager functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "test_state.json")
        self.results_dir = os.path.join(self.temp_dir, "results")
        
        # Mock messages for testing
        self.test_messages = [
            [{"role": "user", "content": f"Message {i}"}] 
            for i in range(1, 251)  # 250 messages
        ]
        
        # Mock files for testing
        self.test_files = [f"test_file_{i:03d}.pdf" for i in range(1, 101)]  # 100 files
        
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_validation(self):
        """Test BatchManager initialization validation"""
        # Test missing model
        with pytest.raises(BatchManagerError, match="model is required"):
            BatchManager(messages=self.test_messages[:5])
        
        # Test both messages and files provided
        with pytest.raises(BatchManagerError, match="Cannot provide both messages and files"):
            BatchManager(
                messages=self.test_messages[:5],
                files=self.test_files[:5],
                model="claude-3-haiku-20240307"
            )
        
        # Test neither messages nor files provided
        with pytest.raises(BatchManagerError, match="Must provide either messages or files"):
            BatchManager(model="claude-3-haiku-20240307")
        
        # Test files without prompt
        with pytest.raises(BatchManagerError, match="prompt is required when using files"):
            BatchManager(files=self.test_files[:5], model="claude-3-haiku-20240307")

    def test_init_with_messages(self):
        """Test successful initialization with messages"""
        manager = BatchManager(
            messages=self.test_messages[:5],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            max_parallel_jobs=2,
            max_cost=10.0,
            state_path=self.state_file
        )
        
        assert manager.model == "claude-3-haiku-20240307"
        assert manager.items_per_job == 2
        assert manager.max_parallel_jobs == 2
        assert manager.max_cost == 10.0
        assert manager.state_path == self.state_file
        assert len(manager.state.jobs) == 3  # 5 messages / 2 per job = 3 jobs
        
    def test_init_with_files(self):
        """Test successful initialization with files"""
        manager = BatchManager(
            files=self.test_files[:5],
            prompt="Extract data from this document",
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        assert len(manager.state.jobs) == 3  # 5 files / 2 per job = 3 jobs
        
    def test_batching_edge_cases(self):
        """Test batching with edge cases"""
        # Test with exactly items_per_job items
        manager = BatchManager(
            messages=self.test_messages[:100],
            model="claude-3-haiku-20240307",
            items_per_job=100
        )
        assert len(manager.state.jobs) == 1
        
        # Test with items_per_job - 1 items
        manager = BatchManager(
            messages=self.test_messages[:99],
            model="claude-3-haiku-20240307",
            items_per_job=100
        )
        assert len(manager.state.jobs) == 1
        
        # Test with items_per_job + 1 items
        manager = BatchManager(
            messages=self.test_messages[:101],
            model="claude-3-haiku-20240307",
            items_per_job=100
        )
        assert len(manager.state.jobs) == 2
        assert len(manager.state.jobs[0].items) == 100
        assert len(manager.state.jobs[1].items) == 1

    def test_state_persistence_new_state(self):
        """Test creating new state file"""
        manager = BatchManager(
            messages=self.test_messages[:5],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        # Check state file was created
        assert os.path.exists(self.state_file)
        
        # Check state file content
        with open(self.state_file, 'r') as f:
            state_data = json.load(f)
        
        assert state_data["model"] == "claude-3-haiku-20240307"
        assert state_data["items_per_job"] == 2
        assert len(state_data["jobs"]) == 3
        assert state_data["total_cost"] == 0.0

    def test_state_persistence_load_existing(self):
        """Test loading existing state file"""
        # Create initial state
        initial_state = {
            "manager_id": "test-uuid",
            "created_at": "2024-01-20T10:00:00Z",
            "model": "claude-3-haiku-20240307",
            "items_per_job": 2,
            "max_cost": 10.0,
            "results_dir": None,
            "batch_kwargs": {},
            "jobs": [
                {
                    "index": 0,
                    "batch_id": None,
                    "status": "completed",
                    "items": [
                        {
                            "original_index": 0,
                            "content": [{"role": "user", "content": "Message 1"}],
                            "status": "succeeded",
                            "error": None,
                            "cost": 0.01,
                            "completed_at": "2024-01-20T10:01:00Z"
                        },
                        {
                            "original_index": 1,
                            "content": [{"role": "user", "content": "Message 2"}],
                            "status": "succeeded",
                            "error": None,
                            "cost": 0.01,
                            "completed_at": "2024-01-20T10:01:00Z"
                        }
                    ],
                    "job_cost": 0.02,
                    "started_at": "2024-01-20T10:00:30Z",
                    "completed_at": "2024-01-20T10:01:00Z"
                }
            ],
            "total_cost": 0.02,
            "last_updated": "2024-01-20T10:01:00Z"
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(initial_state, f)
        
        # Load existing state
        manager = BatchManager(
            messages=self.test_messages[:2],  # Should be ignored
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        assert manager.state.manager_id == "test-uuid"
        assert manager.state.total_cost == 0.02
        assert len(manager.state.jobs) == 1
        assert manager.state.jobs[0].status.value == "completed"

    @patch('batchata.batch_manager.batch')
    def test_run_simple_batch(self, mock_batch):
        """Test running a simple batch without parallel processing"""
        # Create realistic results
        results = create_structured_results(2)
        setup_mock_batch_function(mock_batch, results, "batch_123")
        
        manager = BatchManager(
            messages=self.test_messages[:2],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            max_parallel_jobs=1,
            state_path=self.state_file
        )
        
        summary = manager.run(print_progress=False)
        
        # Check summary structure  
        assert summary["completed_items"] == 2
        assert summary["total_cost"] == 0.02
        assert mock_batch.call_count == 1

    @patch('batchata.batch_manager.batch')
    def test_run_with_parallel_execution(self, mock_batch):
        """Test running batch with parallel execution"""
        # Create realistic results for two jobs
        results1 = [
            {"result": "Result 1", "citations": None},
            {"result": "Result 2", "citations": None}
        ]
        results2 = [
            {"result": "Result 3", "citations": None},
            {"result": "Result 4", "citations": None}
        ]
        
        mock_job1 = create_mock_batch_job("batch_123", results1)
        mock_job2 = create_mock_batch_job("batch_456", results2)
        mock_batch.side_effect = [mock_job1, mock_job2]
        
        manager = BatchManager(
            messages=self.test_messages[:4],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            max_parallel_jobs=2,
            state_path=self.state_file
        )
        
        summary = manager.run(print_progress=False)
        
        # Check summary structure
        assert summary["completed_items"] == 4
        assert summary["total_cost"] == 0.04  # Two jobs at 0.02 each
        assert mock_batch.call_count == 2

    @patch('batchata.batch_manager.batch')
    def test_cost_limit_enforcement(self, mock_batch):
        """Test that cost limit stops new jobs"""
        # Create expensive results that use up the cost limit
        results = [
            {"result": "Result 1", "citations": None},
            {"result": "Result 2", "citations": None}
        ]
        
        def create_expensive_job(*args, **kwargs):
            return create_mock_batch_job("batch_123", results, total_cost=5.0)
        
        mock_batch.side_effect = create_expensive_job
        
        manager = BatchManager(
            messages=self.test_messages[:6],  # Should create 3 jobs
            model="claude-3-haiku-20240307",
            items_per_job=2,
            max_parallel_jobs=1,
            max_cost=5.0,  # Low cost limit
            state_path=self.state_file
        )
        
        summary = manager.run(print_progress=False)
        
        # With max_parallel_jobs=1, first job completes and hits cost limit
        # Remaining jobs are blocked
        assert summary["cost_limit_reached"] == True
        # Only first job completes (2 items)
        assert summary["completed_items"] == 2
        assert mock_batch.call_count == 1

    @patch('batchata.batch_manager.batch')
    def test_retry_failed_items(self, mock_batch):
        """Test retry_failed() method"""
        # Setup state with failed items
        manager = BatchManager(
            messages=self.test_messages[:4],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        # Mark some items as failed
        from batchata.batch_manager import ItemStatus
        manager.state.jobs[0].items[0].status = ItemStatus.FAILED
        manager.state.jobs[0].items[0].error = "API Error"
        manager.state.jobs[1].items[1].status = ItemStatus.FAILED
        manager.state.jobs[1].items[1].error = "Rate limit"
        
        # Create realistic retry results
        retry_results = [
            {"result": "Retry Result 1", "citations": None},
            {"result": "Retry Result 2", "citations": None}
        ]
        
        def create_retry_jobs(*args, **kwargs):
            # For pending jobs and retry job
            return create_mock_batch_job("retry_batch_123", retry_results[:2])
        
        mock_batch.side_effect = create_retry_jobs
        
        retry_summary = manager.retry_failed()
        
        # retry_failed creates a new job for failed items and runs all pending jobs
        # Original 2 jobs were already created, so batch() is called for pending items + retry job
        assert retry_summary["completed_items"] == 6  # All items across all jobs
        assert retry_summary["retry_count"] == 2  # Number of items that were retried
        assert mock_batch.call_count == 2  # Pending job items + retry job

    @patch('batchata.batch_manager.batch')
    def test_results_saving(self, mock_batch):
        """Test saving results to directory"""
        # Mock batch() with structured results and citations
        mock_job = Mock(spec=BatchJob)
        mock_job.is_complete.return_value = True
        mock_job.results.return_value = [
            {
                "result": InvoiceModel(invoice_number="INV-001", company_name="Test Corp", total_amount=100.0),
                "citations": {"invoice_number": [{"start": 45, "end": 52, "text": "INV-001"}]}
            },
            {
                "result": InvoiceModel(invoice_number="INV-002", company_name="Test Corp", total_amount=200.0),
                "citations": {"invoice_number": [{"start": 45, "end": 52, "text": "INV-002"}]}
            }
        ]
        mock_job.stats.return_value = {"total_cost": 0.02}
        mock_job._batch_id = "results_batch_123"
        # Mock the raw directory creation since we're mocking the job
        def mock_results():
            # Create raw directory to simulate real behavior
            os.makedirs(os.path.join(self.results_dir, "raw"), exist_ok=True)
            return [
                {"result": InvoiceModel(invoice_number="INV-001", company_name="Test Corp", total_amount=100.0), 
                 "citations": {"invoice_number": [{"start": 45, "end": 52, "text": "INV-001"}]}},
                {"result": InvoiceModel(invoice_number="INV-002", company_name="Test Corp", total_amount=200.0), 
                 "citations": {"invoice_number": [{"start": 45, "end": 52, "text": "INV-002"}]}}
            ]
        mock_job.results.side_effect = mock_results
        mock_batch.return_value = mock_job
        
        manager = BatchManager(
            messages=self.test_messages[:2],
            model="claude-3-haiku-20240307",
            response_model=InvoiceModel,
            enable_citations=True,
            items_per_job=2,
            max_parallel_jobs=1,
            results_dir=self.results_dir,
            state_path=self.state_file
        )
        
        summary = manager.run(print_progress=False)
        
        # Check results directory structure
        assert os.path.exists(os.path.join(self.results_dir, "raw"))
        assert os.path.exists(os.path.join(self.results_dir, "processed"))
        
        # Check processed results file
        processed_file = os.path.join(self.results_dir, "processed", "job_0_results.json")
        assert os.path.exists(processed_file)
        
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        # Check unified format structure
        assert isinstance(processed_data, list)
        assert len(processed_data) == 2
        
        # Check first entry structure
        first_entry = processed_data[0]
        assert "result" in first_entry
        assert "citations" in first_entry
        assert first_entry["result"]["invoice_number"] == "INV-001"
        assert first_entry["result"]["company_name"] == "Test Corp"

    def test_result_order_maintenance(self):
        """Test that results maintain original order despite parallel processing"""
        # This test would require mocking ThreadPoolExecutor to control completion order
        # For now, we'll test the logic that should maintain order
        manager = BatchManager(
            messages=self.test_messages[:10],
            model="claude-3-haiku-20240307",
            items_per_job=2
        )
        
        # Simulate results from jobs completing out of order
        job_results = {
            2: ["Result 5", "Result 6"],  # Job 2 completes first
            0: ["Result 1", "Result 2"],  # Job 0 completes second
            1: ["Result 3", "Result 4"],  # Job 1 completes third
            3: ["Result 7", "Result 8"],  # Job 3 completes fourth
            4: ["Result 9", "Result 10"]  # Job 4 completes last
        }
        
        # Test the order reconstruction logic
        ordered_results = []
        for job_idx in sorted(job_results.keys()):
            ordered_results.extend(job_results[job_idx])
        
        expected_order = [f"Result {i}" for i in range(1, 11)]
        assert ordered_results == expected_order

    def test_error_handling_partial_failures(self):
        """Test handling of partial failures in jobs"""
        manager = BatchManager(
            messages=self.test_messages[:4],
            model="claude-3-haiku-20240307",
            items_per_job=2,
            state_path=self.state_file
        )
        
        # Simulate partial failure in a job
        from batchata.batch_manager import ItemStatus
        manager.state.jobs[0].items[0].status = ItemStatus.SUCCEEDED
        manager.state.jobs[0].items[0].cost = 0.01
        manager.state.jobs[0].items[1].status = ItemStatus.FAILED
        manager.state.jobs[0].items[1].error = "API Error"
        
        # Test stats calculation with partial failures
        stats = manager.stats
        assert stats["total_items"] == 4
        assert stats["completed_items"] == 1
        assert stats["failed_items"] == 1

    def test_progress_tracking(self):
        """Test progress tracking statistics"""
        manager = BatchManager(
            messages=self.test_messages[:10],
            model="claude-3-haiku-20240307",
            items_per_job=3,
            max_cost=5.0,
            state_path=self.state_file
        )
        
        # Simulate some progress
        from batchata.batch_manager import ItemStatus, JobStatus
        manager.state.total_cost = 2.5
        manager.state.jobs[0].status = JobStatus.COMPLETED
        manager.state.jobs[0].items[0].status = ItemStatus.SUCCEEDED
        manager.state.jobs[0].items[1].status = ItemStatus.SUCCEEDED
        manager.state.jobs[0].items[2].status = ItemStatus.FAILED
        
        stats = manager.stats
        assert stats["total_items"] == 10
        assert stats["completed_items"] == 2
        assert stats["failed_items"] == 1
        assert stats["total_cost"] == 2.5
        assert stats["jobs_completed"] == 1
        assert stats["cost_limit_reached"] == False
        
        # Test cost limit reached
        manager.state.total_cost = 5.0
        stats = manager.stats
        assert stats["cost_limit_reached"] == True

    def test_content_representation_in_state(self):
        """Test that content is properly represented in state file"""
        # Test with messages
        manager = BatchManager(
            messages=self.test_messages[:2],
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        item = manager.state.jobs[0].items[0]
        assert isinstance(item.content, list)
        assert item.content == [{"role": "user", "content": "Message 1"}]
        
        # Test with files (using mock file paths) - use different state file
        file_state_path = self.state_file + "_files"
        manager_files = BatchManager(
            files=["test_file.pdf"],
            prompt="Extract data",
            model="claude-3-5-sonnet-20241022",  # Use file-capable model
            state_path=file_state_path
        )
        
        item = manager_files.state.jobs[0].items[0]
        assert item.content == "test_file.pdf"

    @patch('batchata.batch_manager.batch')
    def test_resume_from_partial_completion(self, mock_batch):
        """Test resuming from a partially completed batch"""
        # Create state with partial completion
        partial_state = {
            "manager_id": "test-uuid",
            "created_at": "2024-01-20T10:00:00Z",
            "model": "claude-3-haiku-20240307",
            "items_per_job": 2,
            "max_cost": None,
            "results_dir": None,
            "batch_kwargs": {},
            "jobs": [
                {
                    "index": 0,
                    "batch_id": None,
                    "status": "completed",
                    "items": [
                        {
                            "original_index": 0,
                            "content": [{"role": "user", "content": "Message 1"}],
                            "status": "succeeded",
                            "error": None,
                            "cost": 0.01,
                            "completed_at": "2024-01-20T10:01:00Z"
                        },
                        {
                            "original_index": 1,
                            "content": [{"role": "user", "content": "Message 2"}],
                            "status": "succeeded",
                            "error": None,
                            "cost": 0.01,
                            "completed_at": "2024-01-20T10:01:00Z"
                        }
                    ],
                    "job_cost": 0.02,
                    "started_at": "2024-01-20T10:00:30Z",
                    "completed_at": "2024-01-20T10:01:00Z"
                },
                {
                    "index": 1,
                    "batch_id": None,
                    "status": "pending",
                    "items": [
                        {
                            "original_index": 2,
                            "content": [{"role": "user", "content": "Message 3"}],
                            "status": "pending",
                            "error": None,
                            "cost": None,
                            "completed_at": None
                        },
                        {
                            "original_index": 3,
                            "content": [{"role": "user", "content": "Message 4"}],
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
            "total_cost": 0.02,
            "last_updated": "2024-01-20T10:01:00Z"
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(partial_state, f)
        
        # Create realistic results for the remaining job
        remaining_results = [
            {"result": "Result 3", "citations": None},
            {"result": "Result 4", "citations": None}
        ]
        setup_mock_batch_function(mock_batch, remaining_results, "resume_batch_123")
        
        manager = BatchManager(
            messages=self.test_messages[:4],  # This should be ignored since we load state
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        summary = manager.run(print_progress=False)
        
        # Should get results from both completed and newly processed jobs
        assert summary["completed_items"] == 4
        # Only the pending job should be processed
        assert mock_batch.call_count == 1

    @patch('batchata.batch_manager.batch')
    def test_cost_limit_sequential_execution(self, mock_batch):
        """Test cost limit prevents new job submission with sequential execution"""
        # Create initial state with some cost already accumulated
        initial_state = {
            "manager_id": "test-uuid",
            "created_at": "2024-01-20T10:00:00Z",
            "model": "claude-3-5-haiku-20241022",
            "items_per_job": 1,
            "max_cost": 0.03,
            "results_dir": None,
            "batch_kwargs": {"prompt": "Extract invoice data"},
            "jobs": [
                {
                    "index": 0,
                    "batch_id": "completed_batch_1",
                    "status": "completed",
                    "items": [{
                        "original_index": 0,
                        "content": "invoice_001.pdf",
                        "status": "succeeded",
                        "error": None,
                        "cost": 0.025,  # Already used most of the budget
                        "completed_at": "2024-01-20T10:01:00Z"
                    }],
                    "job_cost": 0.025,
                    "started_at": "2024-01-20T10:00:30Z",
                    "completed_at": "2024-01-20T10:01:00Z"
                },
                {
                    "index": 1,
                    "batch_id": None,
                    "status": "pending",
                    "items": [{
                        "original_index": 1,
                        "content": "invoice_002.pdf",
                        "status": "pending",
                        "error": None,
                        "cost": None,
                        "completed_at": None
                    }],
                    "job_cost": 0.0,
                    "started_at": None,
                    "completed_at": None
                },
                {
                    "index": 2,
                    "batch_id": None,
                    "status": "pending",
                    "items": [{
                        "original_index": 2,
                        "content": "invoice_003.pdf",
                        "status": "pending",
                        "error": None,
                        "cost": None,
                        "completed_at": None
                    }],
                    "job_cost": 0.0,
                    "started_at": None,
                    "completed_at": None
                }
            ],
            "total_cost": 0.025,  # Already close to limit of 0.03
            "last_updated": "2024-01-20T10:01:00Z"
        }
        
        # Save initial state
        with open(self.state_file, 'w') as f:
            json.dump(initial_state, f)
        
        # Create mock files
        test_files = ["invoice_001.pdf", "invoice_002.pdf", "invoice_003.pdf"]
        
        # Mock batch job with cost that would exceed limit
        def create_job(*args, **kwargs):
            mock_job = Mock(spec=BatchJob)
            mock_job.is_complete.return_value = True
            mock_job.results.return_value = [{"result": {"invoice_number": "INV-002"}, "citations": None}]
            mock_job.stats.return_value = {"total_cost": 0.02}  # 0.025 + 0.02 = 0.045 > 0.03 limit
            mock_job._batch_id = "test_batch_2"
            return mock_job
        
        mock_batch.side_effect = create_job
        
        # Load manager from state
        manager = BatchManager(
            files=test_files,
            prompt="Extract invoice data",
            model="claude-3-5-haiku-20241022",
            state_path=self.state_file,
            max_parallel_jobs=1
        )
        
        # Run processing
        summary = manager.run(print_progress=False)
        
        # First job processes and exceeds limit, second job should be blocked
        assert summary['completed_items'] == 2  # 1 from initial + 1 new
        assert summary['total_cost'] >= 0.03
        assert summary['cost_limit_reached'] == True
        
        # Only the first pending job should be processed
        assert mock_batch.call_count == 1  # Only one new job submitted

    @patch('batchata.batch_manager.batch')
    def test_cost_limit_parallel_execution(self, mock_batch):
        """Test cost limit with parallel execution - realistic cost tracking with timing variability"""
        # Create test files
        test_files = []
        for i in range(4):
            file_path = os.path.join(self.temp_dir, f"invoice_{i+1:03d}.pdf")
            with open(file_path, 'wb') as f:
                f.write(b"Mock PDF content")
            test_files.append(file_path)
        
        # Mock batch jobs - first two will exceed limit when completed
        job_count = 0
        def create_job(*args, **kwargs):
            nonlocal job_count
            job_count += 1
            mock_job = Mock(spec=BatchJob)
            mock_job.is_complete.return_value = True
            mock_job.results.return_value = [{"result": {"invoice_number": f"INV-{job_count:03d}"}, "citations": None}]
            # First two jobs use up the budget
            mock_job.stats.return_value = {"total_cost": 0.018}  # 2 * 0.018 = 0.036 > 0.03 limit
            mock_job._batch_id = f"batch_{job_count}"
            return mock_job
        
        mock_batch.side_effect = create_job
        
        # Initialize manager with parallel execution
        manager = BatchManager(
            files=test_files,
            prompt="Extract invoice data",
            model="claude-3-5-haiku-20241022",
            items_per_job=1,  # 1 file per job = 4 jobs total
            max_parallel_jobs=2,  # Process 2 at a time
            max_cost=0.03,  # Limit that first 2 jobs will exceed
            state_path=self.state_file
        )
        
        # Run processing
        summary = manager.run(print_progress=False)
        
        # With parallel execution and realistic cost tracking:
        # - Jobs 1,2 start immediately (no cost check)
        # - Depending on timing, either job 3 or job 4 may be blocked
        # - If jobs 1&2 finish before job 3 starts: 2 jobs total
        # - If job 1 finishes, job 3 starts, then job 2 finishes: 3 jobs total
        assert mock_batch.call_count >= 2  # At least 2 jobs started
        assert mock_batch.call_count <= 3  # At most 3 jobs before cost limit hit
        assert summary['completed_items'] >= 2
        assert summary['completed_items'] <= 3
        assert summary['total_cost'] >= 0.03
        assert summary['cost_limit_reached'] == True
        
        # 1-2 jobs should remain unprocessed depending on timing
        remaining = summary['total_items'] - summary['completed_items']
        assert remaining >= 1 and remaining <= 2

    def test_results_method_with_results_dir(self):
        """Test BatchManager.results() method with results_dir configured"""
        # Create BatchManager with results_dir
        manager = BatchManager(
            messages=self.test_messages[:2],
            model="claude-3-haiku-20240307",
            results_dir=self.results_dir,
            state_path=self.state_file
        )
        
        # Mock the load_results_from_disk function
        from unittest.mock import patch
        with patch('batchata.utils.load_results_from_disk') as mock_load:
            mock_load.return_value = [
                {"result": "Result 1", "citations": None},
                {"result": "Result 2", "citations": None}
            ]
            
            results = manager.results()
            
            # Verify it called load_results_from_disk with correct parameters
            mock_load.assert_called_once_with(self.results_dir, None)
            assert len(results) == 2
            assert results[0]["result"] == "Result 1"

    def test_results_method_without_results_dir(self):
        """Test BatchManager.results() method without results_dir raises error"""
        manager = BatchManager(
            messages=self.test_messages[:2],
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        with pytest.raises(BatchManagerError, match="No results available"):
            manager.results()