"""Tests for AnthropicProvider.

Testing:
1. Job validation for Anthropic constraints
2. Batch creation and management
3. Cost estimation with token counting
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import os

from batchata.providers.anthropic import AnthropicProvider
from batchata.core.job import Job
from batchata.core.job_result import JobResult
from batchata.exceptions import ValidationError, ProviderError, BatchSubmissionError


class TestAnthropicProvider:
    """Test AnthropicProvider functionality with mocked Anthropic SDK."""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('batchata.providers.anthropic.anthropic.Anthropic') as mock_anthropic_class:
                # Create mock client
                mock_client = MagicMock()
                mock_anthropic_class.return_value = mock_client
                
                # Setup nested attributes for batch operations
                mock_client.messages = MagicMock()
                mock_client.messages.batches = MagicMock()
                
                yield mock_client
    
    @pytest.fixture
    def provider(self, mock_anthropic_client):
        """Create AnthropicProvider with mocked client."""
        # Provider will use the mocked client
        provider = AnthropicProvider(auto_register=False)
        return provider
    
    def test_job_validation_constraints(self, provider):
        """Test Anthropic-specific job validation."""
        # Valid job with alternating roles
        valid_job = Job(
            id="valid-job",
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Should not raise
        provider.validate_job(valid_job)
        
        # Invalid - consecutive same roles  
        with pytest.raises(ValidationError, match="consecutive messages from same role"):
            invalid_job = Job(
                id="invalid-job",
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Are you there?"}
                ]
            )
            provider.validate_job(invalid_job)
        
        # Invalid - unsupported model
        with pytest.raises(ValidationError, match="Unsupported model"):
            invalid_model_job = Job(
                id="bad-model",
                model="gpt-4",  # Not an Anthropic model
                messages=[{"role": "user", "content": "Hello"}]
            )
            provider.validate_job(invalid_model_job)
    
    def test_batch_creation_and_submission(self, provider, mock_anthropic_client):
        """Test creating and submitting batches to Anthropic API."""
        # Create test jobs
        jobs = [
            Job(
                id=f"job-{i}",
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": f"Question {i}"}],
                max_tokens=100,
                temperature=0.5
            )
            for i in range(3)
        ]
        
        # Mock the batch creation response
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_abc123"
        mock_batch_response.processing_status = "in_progress"
        mock_anthropic_client.messages.batches.create.return_value = mock_batch_response
        
        # Create batch
        batch_id = provider.create_batch(jobs)
        
        # Verify batch ID
        assert batch_id == "batch_abc123"
        
        # Verify API was called with correct format
        mock_anthropic_client.messages.batches.create.assert_called_once()
        call_args = mock_anthropic_client.messages.batches.create.call_args
        
        # Check request format
        requests = call_args.kwargs['requests']
        assert len(requests) == 3
        
        # Verify each request has correct structure
        for i, req in enumerate(requests):
            assert req['custom_id'] == f"job-{i}"
            assert req['params']['model'] == "claude-3-5-sonnet-20241022"
            assert req['params']['messages'] == [{"role": "user", "content": f"Question {i}"}]
            assert req['params']['max_tokens'] == 100
            assert req['params']['temperature'] == 0.5
    
    def test_batch_status_checking(self, provider, mock_anthropic_client):
        """Test checking batch status."""
        # Mock different status responses
        mock_batch = MagicMock()
        
        # Test in-progress status
        mock_batch.processing_status = "in_progress"
        mock_batch.ended_at = None
        mock_anthropic_client.messages.batches.retrieve.return_value = mock_batch
        
        status, error_details = provider.get_batch_status("batch_123")
        assert status == "running"
        assert error_details is None
        
        # Test completed status
        mock_batch.processing_status = "ended"
        mock_batch.ended_at = "2024-01-01T00:00:00Z"
        mock_batch.request_counts.succeeded = 10
        mock_batch.request_counts.errored = 0
        mock_batch.request_counts.total = 10
        
        status, error_details = provider.get_batch_status("batch_123")
        assert status == "complete"
        assert error_details is None
        
        # Test failed status
        mock_batch.request_counts.succeeded = 5
        mock_batch.request_counts.errored = 5
        
        status, error_details = provider.get_batch_status("batch_123")
        assert status == "failed"
        assert error_details is not None
        assert error_details["errored_count"] == 5
        assert error_details["total_count"] == 10
    
    def test_batch_results_retrieval(self, provider, mock_anthropic_client):
        """Test retrieving results from completed batch."""
        # Setup job mapping
        job1 = Job(id="job-1", model="claude-3-5-sonnet-20241022", 
                  messages=[{"role": "user", "content": "What is 2+2?"}])
        job2 = Job(id="job-2", model="claude-3-5-sonnet-20241022",
                  messages=[{"role": "user", "content": "What is Python?"}])
        
        provider._job_mapping = {"job-1": job1, "job-2": job2}
        
        # Mock batch status as complete
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        mock_batch.ended_at = "2024-01-01T00:00:00Z"
        mock_batch.request_counts.succeeded = 2
        mock_batch.request_counts.errored = 0
        mock_anthropic_client.messages.batches.retrieve.return_value = mock_batch
        
        # Mock results
        mock_result1 = MagicMock()
        mock_result1.custom_id = "job-1"
        mock_result1.result.type = "succeeded"
        mock_result1.result.message.content = [MagicMock(type="text", text="4")]
        mock_result1.result.message.usage = MagicMock(input_tokens=10, output_tokens=5)
        
        mock_result2 = MagicMock()
        mock_result2.custom_id = "job-2"
        mock_result2.result.type = "succeeded"
        mock_result2.result.message.content = [MagicMock(type="text", text="Python is a programming language")]
        mock_result2.result.message.usage = MagicMock(input_tokens=15, output_tokens=20)
        
        
        mock_anthropic_client.messages.batches.results.return_value = [mock_result1, mock_result2]
        
        # Get results
        results = provider.get_batch_results("batch_123")
        
        # Verify results
        assert len(results) == 2
        assert results[0].job_id == "job-1"
        assert results[0].response == "4"
        assert results[0].input_tokens == 10
        assert results[0].output_tokens == 5
        
        assert results[1].job_id == "job-2"
        assert results[1].response == "Python is a programming language"
        assert results[1].input_tokens == 15
        assert results[1].output_tokens == 20
    
    def test_error_handling(self, provider, mock_anthropic_client):
        """Test error handling in batch operations."""
        # Test batch creation failure
        mock_anthropic_client.messages.batches.create.side_effect = Exception("API Error")
        
        job = Job(id="test", model="claude-3-5-sonnet-20241022",
                 messages=[{"role": "user", "content": "Test"}])
        
        with pytest.raises(BatchSubmissionError, match="Failed to create batch"):
            provider.create_batch([job])
        
       # TODO: test real error result.
    
    @patch('tokencost.calculate_cost_by_tokens')
    @patch('batchata.utils.llm.token_count_simple')  
    def test_cost_estimation(self, mock_token_count, mock_calc_cost, provider):
        """Test cost estimation for Anthropic jobs."""
        # Mock token counting - return different values for each call
        mock_token_count.side_effect = [100, 50]  # Different token counts for each job
        
        # Mock cost calculation - return costs for input and output tokens
        mock_calc_cost.side_effect = [
            0.003,  # Input cost for job 1
            0.015,  # Output cost for job 1
            0.001,  # Input cost for job 2
            0.005   # Output cost for job 2
        ]
        
        jobs = [
            Job(
                id="job-1",
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language"},
                    {"role": "user", "content": "Tell me more"}
                ],
                max_tokens=500
            ),
            Job(
                id="job-2",
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "Explain AI"}],
                max_tokens=200
            )
        ]
        
        total_cost = provider.estimate_cost(jobs)
        
        # Should have called token counting for each job
        assert mock_token_count.call_count == 2  # One call per job
        
        # Should have calculated costs for input and output tokens for each job
        assert mock_calc_cost.call_count == 4  # 2 calls per job (input + output)
        
        # Verify cost calculation includes batch discount
        # Expected: ((0.003 + 0.015) + (0.001 + 0.005)) * 0.5 (batch discount) = 0.012
        assert total_cost == pytest.approx(0.012, rel=0.01)