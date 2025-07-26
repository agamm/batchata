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
    
    def test_google_specific_validation(self, provider):
        """Test Google-specific validation constraints."""
        # Test model validation
        with pytest.raises(Exception, match="Unsupported model"):
            job = Job(
                id="bad-model",
                model="gpt-4",  # Not a Gemini model
                messages=[{"role": "user", "content": "Hello"}]
            )
            provider.validate_job(job)
    
    def test_batch_size_limits(self, provider):
        """Test Google batch size limitations."""
        # Test maximum batch size
        large_jobs = [
            Job(id=f"job-{i}", model="gemini-2.5-flash", messages=[{"role": "user", "content": f"Test {i}"}])
            for i in range(provider.MAX_REQUESTS + 1)
        ]
        
        with pytest.raises(Exception, match="Too many jobs"):
            provider.create_batch(large_jobs)
    
    def test_token_counting_integration(self, provider):
        """Test Google's token counting API integration."""
        job = Job(
            id="token-test",
            model="gemini-2.5-flash", 
            messages=[{"role": "user", "content": "Count my tokens"}]
        )
        
        # Mock the token counting response
        mock_response = MagicMock()
        mock_response.total_tokens = 42
        provider.client.models.count_tokens.return_value = mock_response
        
        # Test token counting
        token_count = provider._count_tokens(job)
        assert token_count == 42
        
        # Verify the API was called correctly
        provider.client.models.count_tokens.assert_called_once()
        call_args = provider.client.models.count_tokens.call_args
        assert call_args[1]["model"] == "gemini-2.5-flash"
    
    def test_cost_estimation_with_real_api(self, provider):
        """Test cost estimation using actual Google token counting."""
        jobs = [
            Job(id="cost-1", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Short"}]),
            Job(id="cost-2", model="gemini-2.5-pro", messages=[{"role": "user", "content": "Longer message"}])
        ]
        
        # Mock token counting responses
        mock_response_1 = MagicMock()
        mock_response_1.total_tokens = 10
        mock_response_2 = MagicMock()
        mock_response_2.total_tokens = 20
        
        provider.client.models.count_tokens.side_effect = [mock_response_1, mock_response_2]
        
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.001):
            cost = provider.estimate_cost(jobs)
            
            # Should apply 50% batch discount
            expected_cost = 0.001 * 2 * 0.5  # 2 jobs * cost * discount
            assert cost == expected_cost
    
    def test_batch_state_transitions(self, provider):
        """Test different Google batch job state transitions."""
        jobs = [Job(id="state-test", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Test"}])]
        batch_id, _ = provider.create_batch(jobs)
        
        # Test different state transitions
        states_to_test = [
            ("JOB_STATE_PENDING", "running"),
            ("JOB_STATE_QUEUED", "running"), 
            ("JOB_STATE_RUNNING", "running"),
            ("BATCH_STATE_RUNNING", "running"),  # Google's actual state
            ("JOB_STATE_SUCCEEDED", "complete"),
            ("JOB_STATE_FAILED", "failed"),
            ("JOB_STATE_CANCELLED", "cancelled")
        ]
        
        for google_state, expected_status in states_to_test:
            # Mock the batch job state
            mock_batch_job = provider._batches[batch_id]["batch_job"]
            mock_batch_job.state.name = google_state
            
            if google_state == "JOB_STATE_FAILED":
                mock_batch_job.error = MagicMock()
                mock_batch_job.error.message = "Test error message"
            
            status, error_details = provider.get_batch_status(batch_id)
            assert status == expected_status
            
            if expected_status == "failed":
                assert error_details is not None
                assert "error" in error_details
    
    def test_batch_results_with_real_google_format(self, provider):
        """Test getting batch results with real Google inline response format."""
        jobs = [Job(id="format-test", model="gemini-2.5-flash", messages=[{"role": "user", "content": "Test"}])]
        batch_id, job_mapping = provider.create_batch(jobs)
        
        # Mock a completed batch with Google's real response format
        mock_batch_job = provider._batches[batch_id]["batch_job"]
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        
        # Create mock destination with inlined responses (real Google format)
        mock_dest = MagicMock()
        mock_inline_response = MagicMock()
        mock_inline_response.response = MagicMock()
        mock_inline_response.response.text = "Test response from Google"
        mock_inline_response.response.usage_metadata = MagicMock()
        mock_inline_response.response.usage_metadata.prompt_token_count = 10
        mock_inline_response.response.usage_metadata.candidates_token_count = 5
        mock_inline_response.error = None
        
        mock_dest.inlined_responses = [mock_inline_response]
        mock_batch_job.dest = mock_dest
        
        # Get results
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.001):
            results = provider.get_batch_results(batch_id, job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "format-test"
        assert result.raw_response == "Test response from Google"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.error is None
        
        # Verify batch was cleaned up
        assert batch_id not in provider._batches