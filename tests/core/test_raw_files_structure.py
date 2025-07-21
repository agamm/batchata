"""Unit tests for raw_files directory structure."""

import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from batchata import Batch
from batchata.core.job_result import JobResult
from pydantic import BaseModel, Field


class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    answer: str = Field(description="A simple answer")


@patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
def test_raw_files_directory_structure():
    """Test that raw_files=True creates the correct directory structure."""
    
    with tempfile.TemporaryDirectory(prefix="batchata_test_") as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        # Create batch with raw_files=True
        batch = (
            Batch(results_dir=str(results_dir), raw_files=True)
            .set_default_params(model="claude-sonnet-4-20250514", temperature=0.7)
        )
        
        # Add a simple job
        batch.add_job(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_model=SimpleResponse
        )
        
        # Mock the provider and batch operations
        with patch('batchata.core.batch_run.get_provider') as mock_get_provider:
            # Create mock provider
            mock_provider = Mock()
            mock_provider.create_batch.return_value = ("test_batch_id", {"job-123": batch.jobs[0]})
            mock_provider.get_batch_status.return_value = ("complete", None)
            mock_provider.get_batch_results.return_value = [
                JobResult(
                    job_id="job-123",
                    raw_response='{"answer": "4"}',
                    parsed_response=SimpleResponse(answer="4"),
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                    batch_id="test_batch_id"
                )
            ]
            mock_provider.estimate_cost.return_value = 0.001
            mock_get_provider.return_value = mock_provider
            
            # Run the batch
            run = batch.run(print_status=False)
            
            # Verify provider methods were called with raw_files_dir
            mock_provider.create_batch.assert_called_once()
            create_args = mock_provider.create_batch.call_args
            assert len(create_args[0]) == 2  # jobs list and raw_files_dir
            jobs_arg, raw_files_arg = create_args[0]
            assert len(jobs_arg) == 1  # one job
            assert raw_files_arg == str(results_dir / "raw_files")
            
            mock_provider.get_batch_results.assert_called_once()
            results_args = mock_provider.get_batch_results.call_args
            assert len(results_args[0]) == 3  # batch_id, job_mapping, raw_files_dir
            batch_id, job_mapping, raw_files_dir = results_args[0]
            assert raw_files_dir == str(results_dir / "raw_files")


@patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
def test_raw_files_false_no_directory():
    """Test that raw_files=False doesn't pass raw_files_dir to providers."""
    
    with tempfile.TemporaryDirectory(prefix="batchata_test_") as temp_dir:
        results_dir = Path(temp_dir) / "results"
        
        # Create batch with raw_files=False
        batch = (
            Batch(results_dir=str(results_dir), raw_files=False)
            .set_default_params(model="claude-sonnet-4-20250514", temperature=0.7)
        )
        
        # Add a simple job
        batch.add_job(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_model=SimpleResponse
        )
        
        # Mock the provider and batch operations
        with patch('batchata.core.batch_run.get_provider') as mock_get_provider:
            # Create mock provider
            mock_provider = Mock()
            mock_provider.create_batch.return_value = ("test_batch_id", {"job-123": batch.jobs[0]})
            mock_provider.get_batch_status.return_value = ("complete", None)
            mock_provider.get_batch_results.return_value = [
                JobResult(
                    job_id="job-123",
                    raw_response='{"answer": "4"}',
                    parsed_response=SimpleResponse(answer="4"),
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                    batch_id="test_batch_id"
                )
            ]
            mock_provider.estimate_cost.return_value = 0.001
            mock_get_provider.return_value = mock_provider
            
            # Run the batch
            run = batch.run(print_status=False)
            
            # Verify provider methods were called WITHOUT raw_files_dir
            mock_provider.create_batch.assert_called_once()
            create_args = mock_provider.create_batch.call_args
            assert len(create_args[0]) == 2  # jobs list and raw_files_dir
            jobs_arg, raw_files_arg = create_args[0]
            assert len(jobs_arg) == 1  # one job
            assert raw_files_arg is None
            
            mock_provider.get_batch_results.assert_called_once()
            results_args = mock_provider.get_batch_results.call_args
            assert len(results_args[0]) == 3  # batch_id, job_mapping, raw_files_dir
            batch_id, job_mapping, raw_files_dir = results_args[0]
            assert raw_files_dir is None


def test_provider_save_raw_requests_anthropic():
    """Test that AnthropicProvider._save_raw_requests creates correct structure."""
    
    from batchata.providers.anthropic.anthropic import AnthropicProvider
    
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_files_dir = Path(temp_dir)
        
        # Create provider instance
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider()
        
        # Test data (JSON structure for Anthropic)
        batch_requests = [
            {
                "custom_id": "job-123",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100
                }
            }
        ]
        
        # Call the method
        provider._save_raw_requests("test_batch_id", batch_requests, str(raw_files_dir), "anthropic")
        
        # Check directory structure
        requests_dir = raw_files_dir / "requests"
        assert requests_dir.exists()
        
        # Check file exists with correct name and extension
        expected_file = requests_dir / "anthropic_batch_test_batch_id.json"
        assert expected_file.exists()
        
        # Check file contents
        with open(expected_file) as f:
            saved_data = json.load(f)
        
        assert saved_data == batch_requests
        assert len(saved_data) == 1
        assert saved_data[0]["custom_id"] == "job-123"


def test_provider_save_raw_requests_openai():
    """Test that OpenAI provider._save_raw_requests creates correct structure for JSONL."""
    
    from batchata.providers.openai.openai_provider import OpenAIProvider
    
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_files_dir = Path(temp_dir)
        
        # Create provider instance
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider()
        
        # Test JSONL content (string for OpenAI)
        jsonl_content = '{"custom_id": "job-123", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}}'
        
        # Call the method
        provider._save_raw_requests("test_batch_id", jsonl_content, str(raw_files_dir), "openai")
        
        # Check directory structure
        requests_dir = raw_files_dir / "requests"
        assert requests_dir.exists()
        
        # Check file exists with correct name and extension
        expected_file = requests_dir / "openai_batch_test_batch_id.jsonl"
        assert expected_file.exists()
        
        # Check file contents
        with open(expected_file) as f:
            saved_content = f.read().strip()
        
        assert saved_content == jsonl_content


def test_provider_save_raw_responses_formats():
    """Test that _save_raw_responses handles both JSON and JSONL formats correctly."""
    
    from batchata.providers.provider import Provider
    
    class TestProvider(Provider):
        """Test provider implementation."""
        def validate_job(self, job): pass
        def create_batch(self, jobs, raw_files_dir=None): pass
        def get_batch_status(self, batch_id): pass
        def get_batch_results(self, batch_id, job_mapping, raw_files_dir=None): pass
        def cancel_batch(self, batch_id): pass
        def estimate_cost(self, jobs): pass
    
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_files_dir = Path(temp_dir)
        provider = TestProvider()
        
        # Test JSON format (dict/list)
        json_data = [{"job_id": "job-123", "result": "success"}]
        provider._save_raw_responses("test_batch_id", json_data, str(raw_files_dir), "test")
        
        json_file = raw_files_dir / "responses" / "test_batch_test_batch_id.json"
        assert json_file.exists()
        
        with open(json_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == json_data
        
        # Test JSONL format (string)
        jsonl_content = '{"job_id": "job-456", "result": "success"}\n{"job_id": "job-789", "result": "error"}'
        provider._save_raw_responses("test_batch_id_2", jsonl_content, str(raw_files_dir), "test")
        
        jsonl_file = raw_files_dir / "responses" / "test_batch_test_batch_id_2.jsonl"
        assert jsonl_file.exists()
        
        with open(jsonl_file) as f:
            loaded_content = f.read().strip()
        assert loaded_content == jsonl_content
