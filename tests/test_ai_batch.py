import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from src import batch


class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str


def test_batch_empty_messages():
    """Test batch function with empty messages list."""
    job = batch(
        messages=[],
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert job.is_complete()
    assert job.results() == []


def test_batch_invalid_model():
    """Test batch function with invalid model name - now handled by Anthropic API."""
    messages = [[{"role": "user", "content": "Test message"}]]
    
    # Since we removed model validation, batch creation should succeed
    # The error will come from the API when trying to get results
    job = batch(
        messages=messages,
        model="invalid-model",
        response_model=SpamResult
    )
    
    # Job should be created successfully
    assert job is not None
    from src.batch_job import BatchJob
    assert isinstance(job, BatchJob)


def test_batch_missing_required_params():
    """Test batch function with missing required parameters."""
    with pytest.raises(TypeError):
        batch()
    
    with pytest.raises(TypeError):
        batch(messages=[])


def test_batch_with_empty_messages():
    """Test that batch function works with empty messages."""
    job = batch(
        messages=[],
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert job.is_complete()
    assert job.results() == []


@patch.dict(os.environ, {}, clear=True)
def test_missing_api_key():
    """Test that missing API key raises appropriate error."""
    messages = [[{"role": "user", "content": "Test message"}]]
    
    with pytest.raises(TypeError, match="Could not resolve authentication method"):
        job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=SpamResult
        )
        # Try to get results to trigger API error
        job.results()


@patch('src.core.AnthropicBatchProvider')
def test_batch_creates_batch_job(mock_provider_class):
    """Test that batch function creates a batch job."""
    # Mock provider instance
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider.has_citations_enabled.return_value = False
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = ([SpamResult(is_spam=True, confidence=0.9, reason="Test")], None)
    
    messages = [[{"role": "user", "content": "Test message"}]]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    # Verify the provider methods were called during batch creation
    mock_provider.validate_batch.assert_called_once()
    mock_provider.prepare_batch_requests.assert_called_once()
    mock_provider.create_batch.assert_called_once()
    mock_provider.has_citations_enabled.assert_called_once()
    
    # Test that BatchJob is returned
    from src.batch_job import BatchJob
    assert isinstance(job, BatchJob)
    assert job._batch_id == "batch_123"
    
    # Test getting results
    results = job.results()
    assert len(results) == 1
    assert results[0].is_spam == True


@patch('src.core.AnthropicBatchProvider')
def test_batch_multiple_messages(mock_provider_class):
    """Test that batch processes multiple messages correctly."""
    # Mock provider instance
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [
        {'custom_id': 'request_0', 'params': {}},
        {'custom_id': 'request_1', 'params': {}}
    ]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider.has_citations_enabled.return_value = False
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = ([
        SpamResult(is_spam=True, confidence=0.9, reason="Spam"),
        SpamResult(is_spam=False, confidence=0.1, reason="Not spam")
    ], None)
    
    messages = [
        [{"role": "user", "content": "Message 1"}],
        [{"role": "user", "content": "Message 2"}]
    ]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    results = job.results()
    assert len(results) == 2
    assert results[0].is_spam == True
    assert results[1].is_spam == False


@patch('src.core.AnthropicBatchProvider')
def test_batch_without_response_model(mock_provider_class):
    """Test that batch returns raw text when no response_model is provided."""
    # Mock provider instance
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider.has_citations_enabled.return_value = False
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = (["This is a raw text response"], None)
    
    messages = [[{"role": "user", "content": "Test message"}]]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307"
    )
    
    results = job.results()
    assert len(results) == 1
    assert results[0] == "This is a raw text response"
    assert isinstance(results[0], str)