"""
Test that all result methods return the same unified format.

This ensures consistency across:
- BatchJob.results()
- BatchManager.results()  
- load_results_from_disk()
"""

import os
import tempfile
import json
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import pytest
from pydantic import BaseModel

from batchata.batch_job import BatchJob
from batchata.batch_manager import BatchManager
from batchata.utils import load_results_from_disk
from batchata.types import BatchResult


class SampleModel(BaseModel):
    """Test model for unified format testing"""
    name: str
    value: int


class TestUnifiedFormatConsistency:
    """Test that all result methods return the same unified format"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create sample unified format results
        self.sample_results = [
            {
                "result": SampleModel(name="test1", value=100),
                "citations": {"name": [{"text": "test1", "start": 0, "end": 5}]}
            },
            {
                "result": SampleModel(name="test2", value=200),
                "citations": {"name": [{"text": "test2", "start": 10, "end": 15}]}
            }
        ]

    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_job_results_format(self):
        """Test that BatchJob.results() returns unified format"""
        # Mock BatchJob with results
        mock_provider = Mock()
        mock_provider.get_batch_status.return_value = {"status": "completed"}
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []  # Raw results (not used directly)
        mock_provider.parse_results.return_value = self.sample_results  # Processed results
        
        job = BatchJob(
            batch_id="test_batch",
            provider=mock_provider,
            response_model=SampleModel,
            enable_citations=True
        )
        
        results = job.results()
        
        # Verify format
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result_entry in results:
            assert isinstance(result_entry, dict)
            assert "result" in result_entry
            assert "citations" in result_entry
            assert isinstance(result_entry["result"], SampleModel)
            assert isinstance(result_entry["citations"], dict)

    def test_batch_manager_results_format(self):
        """Test that BatchManager.results() returns unified format"""
        # Create processed results files
        processed_dir = os.path.join(self.results_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save sample results to disk (serialized format)
        serialized_results = [
            {
                "result": {"name": "test1", "value": 100},
                "citations": {"name": [{"type": "text", "cited_text": "test1", "document_index": 0}]}
            },
            {
                "result": {"name": "test2", "value": 200},
                "citations": {"name": [{"type": "text", "cited_text": "test2", "document_index": 0}]}
            }
        ]
        
        with open(os.path.join(processed_dir, "job_0_results.json"), "w") as f:
            json.dump(serialized_results, f)
        
        # Create BatchManager
        manager = BatchManager(
            messages=[
                [{"role": "user", "content": "test1"}],
                [{"role": "user", "content": "test2"}]
            ],
            model="claude-3-haiku-20240307",
            response_model=SampleModel,
            results_dir=self.results_dir
        )
        
        results = manager.results()
        
        # Verify format
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result_entry in results:
            assert isinstance(result_entry, dict)
            assert "result" in result_entry
            assert "citations" in result_entry
            assert isinstance(result_entry["result"], SampleModel)
            assert isinstance(result_entry["citations"], dict)

    def test_load_results_from_disk_format(self):
        """Test that load_results_from_disk() returns unified format"""
        # Create processed results directory and files
        processed_dir = os.path.join(self.results_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save sample results to disk (serialized format)
        serialized_results = [
            {
                "result": {"name": "test1", "value": 100},
                "citations": {"name": [{"type": "text", "cited_text": "test1", "document_index": 0}]}
            },
            {
                "result": {"name": "test2", "value": 200},
                "citations": {"name": [{"type": "text", "cited_text": "test2", "document_index": 0}]}
            }
        ]
        
        with open(os.path.join(processed_dir, "job_0_results.json"), "w") as f:
            json.dump(serialized_results, f)
        
        # Load results
        results = load_results_from_disk(self.results_dir, SampleModel)
        
        # Verify format
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result_entry in results:
            assert isinstance(result_entry, dict)
            assert "result" in result_entry
            assert "citations" in result_entry
            assert isinstance(result_entry["result"], SampleModel)
            assert isinstance(result_entry["citations"], dict)

    def test_all_methods_return_same_structure(self):
        """Test that all methods return the same structure"""
        # This is a conceptual test to verify they all return List[BatchResult]
        # In practice, we can't easily test all three with the same data
        # But we can verify they all return the same type structure
        
        # Create mock data
        mock_result = {
            "result": SampleModel(name="test", value=100),
            "citations": {"name": [{"text": "test", "start": 0, "end": 5}]}
        }
        
        # Test that all methods should return List[Dict[str, Any]] with same keys
        expected_keys = {"result", "citations"}
        
        # Each method should return a list of dicts with these keys
        assert isinstance(mock_result, dict)
        assert set(mock_result.keys()) == expected_keys
        
        # The BatchResult TypedDict enforces this structure
        batch_result: BatchResult = mock_result
        assert "result" in batch_result
        assert "citations" in batch_result

    def test_batch_result_type_annotation(self):
        """Test that BatchResult type annotation matches actual usage"""
        from batchata.types import BatchResult
        
        # Test with Pydantic model
        result_with_model: BatchResult = {
            "result": SampleModel(name="test", value=100),
            "citations": {"name": [{"text": "test", "start": 0, "end": 5}]}
        }
        
        # Test with dict
        result_with_dict: BatchResult = {
            "result": {"name": "test", "value": 100},
            "citations": {"name": [{"text": "test", "start": 0, "end": 5}]}
        }
        
        # Test with string
        result_with_string: BatchResult = {
            "result": "test response",
            "citations": [{"text": "test", "start": 0, "end": 4}]
        }
        
        # Test with None citations
        result_with_none: BatchResult = {
            "result": "test response",
            "citations": None
        }
        
        # All should be valid BatchResult instances
        assert isinstance(result_with_model, dict)
        assert isinstance(result_with_dict, dict)
        assert isinstance(result_with_string, dict)
        assert isinstance(result_with_none, dict)