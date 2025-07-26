"""Tests for Gemini provider main functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from batchata.providers.gemini import GeminiProvider
from batchata.core.job import Job


class TestGeminiProvider:
    """Test cases for Gemini provider."""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock the API key environment variable."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            yield
    
    @pytest.fixture
    def provider(self, mock_api_key):
        """Create a Gemini provider instance."""
        with patch('google.genai.Client') as mock_client_class:
            # Create a mock client instance
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock batch operations
            mock_batch_job = MagicMock()
            mock_batch_job.name = "test_batch_123456"
            mock_batch_job.state = MagicMock()
            mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
            
            mock_client.batches.create.return_value = mock_batch_job
            mock_client.batches.get.return_value = mock_batch_job
            
            return GeminiProvider()
    
    def test_provider_initialization(self, mock_api_key):
        """Test provider can be initialized with API key."""
        with patch('google.genai.Client') as mock_client:
            provider = GeminiProvider()
            mock_client.assert_called_once_with(api_key='test-key')
            assert len(provider.models) == 5  # Number of Gemini models
    
    def test_provider_initialization_no_api_key(self):
        """Test provider raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is required"):
                GeminiProvider()
    
    def test_supports_model(self, provider):
        """Test model support checking."""
        assert provider.supports_model("gemini-2.5-flash")
        assert provider.supports_model("gemini-2.5-pro")
        assert not provider.supports_model("gpt-4")
        assert not provider.supports_model("claude-3-opus")
    
    def test_validate_job_success(self, provider):
        """Test successful job validation."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Test prompt"}]
        )
        provider.validate_job(job)  # Should not raise
    
    def test_validate_job_none(self, provider):
        """Test validation fails for None job."""
        with pytest.raises(Exception, match="Job cannot be None"):
            provider.validate_job(None)
    
    def test_validate_job_unsupported_model(self, provider):
        """Test validation fails for unsupported model."""
        job = Job(
            id="test-1",
            model="unsupported-model",
            messages=[{"role": "user", "content": "Test prompt"}]
        )
        with pytest.raises(Exception, match="Unsupported model"):
            provider.validate_job(job)
    
    def test_polling_interval(self, provider):
        """Test polling interval is reasonable."""
        interval = provider.get_polling_interval()
        assert isinstance(interval, float)
        assert interval > 0
        assert interval <= 10
    
    def test_create_batch(self, provider):
        """Test batch creation."""
        jobs = [
            Job(id="test-1", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Test prompt 1"}]),
            Job(id="test-2", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Test prompt 2"}]),
        ]
        
        batch_id, job_mapping = provider.create_batch(jobs)
        
        assert isinstance(batch_id, str)
        assert len(job_mapping) == 2
        assert "test-1" in job_mapping
        assert "test-2" in job_mapping
    
    def test_create_empty_batch(self, provider):
        """Test creating empty batch raises error."""
        with pytest.raises(Exception, match="Cannot create empty batch"):
            provider.create_batch([])
    
    def test_create_too_large_batch(self, provider):
        """Test creating batch with too many jobs raises error."""
        jobs = [
            Job(id=f"test-{i}", model="gemini-2.5-flash", messages=[{"role": "user", "content": f"Test prompt {i}"}])
            for i in range(provider.MAX_REQUESTS + 1)
        ]
        
        with pytest.raises(Exception, match="Too many jobs"):
            provider.create_batch(jobs)
    
    def test_cancel_batch(self, provider):
        """Test batch cancellation."""
        # Test cancelling non-existent batch
        result = provider.cancel_batch("non-existent")
        assert result is False
    
    def test_get_batch_status_not_found(self, provider):
        """Test batch status for non-existent batch."""
        status, error_details = provider.get_batch_status("non-existent")
        assert status == "failed"
        assert "Batch not found" in error_details["error"]
    
    def test_get_batch_results_empty_mapping(self, provider):
        """Test get_batch_results with empty job mapping."""
        # First create a batch so it exists
        jobs = [Job(id="test-1", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Test"}])]
        batch_id, _ = provider.create_batch(jobs)
        
        # Mock batch as completed
        provider._batches[batch_id]["batch_job"].state.name = "JOB_STATE_SUCCEEDED"
        
        # Now test with empty mapping - should return empty list due to early exit
        result = provider.get_batch_results(batch_id, {})
        assert result == []