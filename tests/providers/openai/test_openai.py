"""Tests for OpenAIProvider.

Testing:
1. Job validation for OpenAI model constraints
2. Batch creation with JSONL format 
3. Batch status checking with different states
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json

from batchata.providers.openai.openai_provider import OpenAIProvider
from batchata.core.job import Job
from batchata.exceptions import ValidationError, BatchSubmissionError


class TestOpenAIProvider:
    """Test OpenAIProvider functionality with mocked OpenAI SDK."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('batchata.providers.openai.openai_provider.OpenAI') as mock_openai_class:
                mock_client = MagicMock()
                mock_openai_class.return_value = mock_client
                
                mock_client.batches = MagicMock()
                mock_client.files = MagicMock()
                
                yield mock_client
    
    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create OpenAIProvider with mocked client."""
        return OpenAIProvider(auto_register=False)
    
    def test_job_validation_and_model_support(self, provider):
        """Test OpenAI-specific job validation and model support."""
        # Valid job with supported model
        valid_job = Job(
            id="valid-job",
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Should not raise
        provider.validate_job(valid_job)
        
        # Test polling interval is OpenAI-specific (5 seconds)
        assert provider.get_polling_interval() == 5.0
        
        # Invalid - unsupported model
        with pytest.raises(ValidationError, match="Unsupported model: claude-3-5-sonnet"):
            invalid_job = Job(
                id="invalid-job", 
                model="claude-3-5-sonnet",  # Not an OpenAI model
                messages=[{"role": "user", "content": "Test"}]
            )
            provider.validate_job(invalid_job)
        
        # Invalid - missing API key should fail at init
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                OpenAIProvider(auto_register=False)
    
    def test_batch_creation_jsonl_format(self, provider, mock_openai_client):
        """Test batch creation with proper JSONL format and file upload."""
        jobs = [
            Job(
                id="job-1",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=100,
                temperature=0.5
            ),
            Job(
                id="job-2", 
                model="gpt-4.1-nano-2025-04-14",
                messages=[{"role": "user", "content": "Explain Python"}],
                max_tokens=200
            )
        ]
        
        # Mock file upload
        mock_file_response = MagicMock()
        mock_file_response.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file_response
        
        # Mock batch creation
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-xyz789"
        mock_batch_response.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_id, job_mapping = provider.create_batch(jobs, raw_files_dir=temp_dir)
            
            # Verify batch creation results
            assert batch_id == "batch-xyz789"
            assert len(job_mapping) == 2
            assert "job-1" in job_mapping
            assert "job-2" in job_mapping
            
            # Verify file was uploaded with correct format
            mock_openai_client.files.create.assert_called_once()
            file_call = mock_openai_client.files.create.call_args
            assert file_call.kwargs['purpose'] == 'batch'
            
            # Verify batch was created with file reference
            mock_openai_client.batches.create.assert_called_once()
            batch_call = mock_openai_client.batches.create.call_args
            assert batch_call.kwargs['input_file_id'] == "file-abc123"
            assert batch_call.kwargs['endpoint'] == "/v1/chat/completions"
            assert batch_call.kwargs['completion_window'] == "24h"
    
    def test_batch_status_checking_states(self, provider, mock_openai_client):
        """Test checking batch status for different OpenAI batch states."""
        mock_batch = MagicMock()
        mock_openai_client.batches.retrieve.return_value = mock_batch
        
        # Test validating/in_progress states -> "running"
        mock_batch.status = "in_progress"
        mock_batch.output_file_id = None
        
        status, error = provider.get_batch_status("batch-123")
        assert status == "running"
        assert error is None
        
        # Test completed status
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-123"
        mock_batch.request_counts = MagicMock()
        mock_batch.request_counts.completed = 10
        mock_batch.request_counts.failed = 0
        mock_batch.request_counts.total = 10
        
        status, error = provider.get_batch_status("batch-123")
        assert status == "complete"
        assert error is None
        
        # Test failed/expired states
        mock_batch.status = "failed"
        mock_batch.errors = MagicMock()
        mock_batch.errors.data = [{"message": "Test batch error"}]
        
        status, error = provider.get_batch_status("batch-123")
        assert status == "failed"
        assert error is not None
        assert error["status"] == "failed"
        
        # Test expired status
        mock_batch.status = "expired"
        mock_batch.errors = None
        
        status, error = provider.get_batch_status("batch-123")
        assert status == "failed"
        assert error["status"] == "expired"