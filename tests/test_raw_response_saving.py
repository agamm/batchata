"""Tests for raw response saving functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from src.batch_job import BatchJob
from src.core import batch
from src.file_processing import batch_files


class MockProvider:
    """Mock provider for testing."""
    
    def get_results(self, batch_id: str):
        """Mock raw results from API."""
        return [
            {
                "batch_id": batch_id,
                "request_id": "req_001",
                "result": {
                    "type": "message",
                    "content": [{"type": "text", "text": "Response 1"}]
                },
                "metadata": {"timestamp": "2024-01-01T00:00:00Z"}
            },
            {
                "batch_id": batch_id,
                "request_id": "req_002", 
                "result": {
                    "type": "message",
                    "content": [{"type": "text", "text": "Response 2"}]
                },
                "metadata": {"timestamp": "2024-01-01T00:00:01Z"}
            }
        ]
    
    def parse_results(self, raw_results, response_model, enable_citations):
        """Mock parsed results."""
        return ["Response 1", "Response 2"], []
    
    def get_batch_status(self, batch_id):
        """Mock batch status."""
        return "completed"
    
    def _is_batch_completed(self, status):
        """Mock completion check."""
        return status == "completed"


class ExampleModel(BaseModel):
    """Example model for structured responses."""
    message: str


class TestRawResponseSaving:
    """Test suite for raw response saving functionality."""
    
    def test_raw_response_saving_with_valid_directory(self):
        """Test raw response files are saved with valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create BatchJob with raw_results_dir
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="test_batch_123",
                response_model=None,
                verbose=False,
                enable_citations=False,
                raw_results_dir=temp_dir
            )
            
            # Get results (should trigger raw response saving)
            results = batch_job.results()
            
            # Check that raw response files were created
            expected_files = [
                Path(temp_dir) / "test_batch_123_0.json",
                Path(temp_dir) / "test_batch_123_1.json"
            ]
            
            for file_path in expected_files:
                assert file_path.exists(), f"Expected file {file_path} was not created"
                
                # Check file contents
                with open(file_path, 'r') as f:
                    saved_data = json.load(f)
                    assert "batch_id" in saved_data
                    assert "request_id" in saved_data
                    assert "result" in saved_data
                    assert saved_data["batch_id"] == "test_batch_123"
    
    def test_raw_response_saving_directory_creation(self):
        """Test directory is created when it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-existent subdirectory path
            non_existent_dir = Path(temp_dir) / "raw_responses" / "subdir"
            
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="test_batch_456",
                response_model=None,
                verbose=False,
                enable_citations=False,
                raw_results_dir=str(non_existent_dir)
            )
            
            # Get results (should create directory and save files)
            results = batch_job.results()
            
            # Check that directory was created
            assert non_existent_dir.exists(), "Directory should have been created"
            
            # Check that files were saved
            expected_files = [
                non_existent_dir / "test_batch_456_0.json",
                non_existent_dir / "test_batch_456_1.json"
            ]
            
            for file_path in expected_files:
                assert file_path.exists(), f"Expected file {file_path} was not created"
    
    def test_raw_response_saving_none_parameter(self):
        """Test backward compatibility when raw_results_dir is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="test_batch_789",
                response_model=None,
                verbose=False,
                enable_citations=False,
                raw_results_dir=None
            )
            
            # Get results
            results = batch_job.results()
            
            # Check that no files were created in temp directory
            files_in_temp = list(Path(temp_dir).glob("*"))
            assert len(files_in_temp) == 0, "No files should be created when raw_results_dir is None"
    
    def test_raw_response_saving_with_structured_model(self):
        """Test raw response saving works with structured response models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="test_batch_struct",
                response_model=ExampleModel,
                verbose=False,
                enable_citations=False,
                raw_results_dir=temp_dir
            )
            
            # Get results
            results = batch_job.results()
            
            # Check that raw response files were created
            expected_files = [
                Path(temp_dir) / "test_batch_struct_0.json",
                Path(temp_dir) / "test_batch_struct_1.json"
            ]
            
            for file_path in expected_files:
                assert file_path.exists(), f"Expected file {file_path} was not created"
    
    def test_raw_response_saving_with_citations(self):
        """Test raw response saving works with citations enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="test_batch_cit",
                response_model=None,
                verbose=False,
                enable_citations=True,
                raw_results_dir=temp_dir
            )
            
            # Get results
            results = batch_job.results()
            
            # Check that raw response files were created
            expected_files = [
                Path(temp_dir) / "test_batch_cit_0.json",
                Path(temp_dir) / "test_batch_cit_1.json"
            ]
            
            for file_path in expected_files:
                assert file_path.exists(), f"Expected file {file_path} was not created"
    
    def test_raw_response_saving_empty_batch(self):
        """Test raw response saving handles empty batch correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="empty_batch",
                response_model=None,
                verbose=False,
                enable_citations=False,
                raw_results_dir=temp_dir
            )
            
            # Get results
            results = batch_job.results()
            
            # Check that no files were created for empty batch
            files_in_temp = list(Path(temp_dir).glob("*"))
            assert len(files_in_temp) == 0, "No files should be created for empty batch"
    
    def test_file_naming_pattern(self):
        """Test that files are named with correct pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = MockProvider()
            batch_job = BatchJob(
                provider=provider,
                batch_id="my_batch_123",
                response_model=None,
                verbose=False,
                enable_citations=False,
                raw_results_dir=temp_dir
            )
            
            # Get results
            results = batch_job.results()
            
            # Check file names follow pattern: {batch_id}_{index}.json
            expected_files = [
                "my_batch_123_0.json",
                "my_batch_123_1.json"
            ]
            
            actual_files = [f.name for f in Path(temp_dir).glob("*.json")]
            actual_files.sort()
            
            assert actual_files == expected_files, f"Expected {expected_files}, got {actual_files}"
    
    @patch('src.core.AnthropicBatchProvider')
    def test_batch_api_with_raw_results_dir(self, mock_provider_class):
        """Test batch() API accepts raw_results_dir parameter."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.validate_batch.return_value = None
        mock_provider.prepare_batch_requests.return_value = []
        mock_provider.create_batch.return_value = "test_batch"
        mock_provider.has_citations_enabled.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should not raise an error
            job = batch(
                messages=[
                    [{"role": "user", "content": "Test message"}]
                ],
                model="claude-3-haiku-20240307",
                raw_results_dir=temp_dir
            )
            
            # Check that BatchJob was created with raw_results_dir
            assert hasattr(job, '_raw_results_dir')
            assert job._raw_results_dir == temp_dir
    
    def test_batch_files_api_with_raw_results_dir(self):
        """Test batch_files() API accepts raw_results_dir parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_bytes(b"mock pdf content")
            
            with patch('src.file_processing.batch') as mock_batch:
                mock_job = Mock()
                mock_batch.return_value = mock_job
                
                # This should not raise an error
                job = batch_files(
                    files=[str(pdf_path)],
                    prompt="Test prompt",
                    model="claude-3-haiku-20240307",
                    raw_results_dir=temp_dir
                )
                
                # Check that batch() was called with raw_results_dir
                mock_batch.assert_called_once()
                call_kwargs = mock_batch.call_args[1]
                assert 'raw_results_dir' in call_kwargs
                assert call_kwargs['raw_results_dir'] == temp_dir