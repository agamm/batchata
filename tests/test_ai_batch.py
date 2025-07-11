import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from batchata import batch


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
    """Test batch function with invalid model name - handled by provider registry."""
    messages = [[{"role": "user", "content": "Test message"}]]
    
    # Should raise ValueError immediately when trying to get provider for invalid model
    with pytest.raises(ValueError, match="No provider supports model 'invalid-model'"):
        batch(
            messages=messages,
            model="invalid-model",
            response_model=SpamResult
        )


def test_batch_missing_required_params():
    """Test batch function with missing required parameters."""
    # Missing model parameter
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'model'"):
        batch()
    
    # Missing both messages and files
    with pytest.raises(ValueError, match="Must provide either messages or files"):
        batch(model="claude-3-haiku-20240307")


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
    
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
        job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=SpamResult
        )


@patch('batchata.core.get_provider_for_model')
def test_batch_creates_batch_job(mock_provider_func):
    """Test that batch function creates a batch job and calls provider correctly."""
    # Create a mock provider instance with realistic behavior
    mock_provider = MagicMock()
    mock_provider_func.return_value = mock_provider
    
    # Set up provider responses
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = [{"result": SpamResult(is_spam=True, confidence=0.9, reason="Test"), "citations": None}]
    
    messages = [[{"role": "user", "content": "Test message"}]]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    # Verify the provider function was called and methods called
    mock_provider_func.assert_called_once_with("claude-3-haiku-20240307")
    mock_provider.validate_batch.assert_called_once_with(messages, SpamResult)
    mock_provider.prepare_batch_requests.assert_called_once()
    mock_provider.create_batch.assert_called_once()
    
    # Test that BatchJob is returned with correct properties
    from batchata.batch_job import BatchJob
    assert isinstance(job, BatchJob)
    assert job._batch_id == "batch_123"
    assert job._response_model == SpamResult
    
    # Test getting results
    results = job.results()
    assert len(results) == 1
    assert results[0]["result"].is_spam == True


@patch('batchata.core.get_provider_for_model')
def test_batch_multiple_messages(mock_provider_func):
    """Test that batch processes multiple messages correctly."""
    # Create mock provider with realistic multi-message handling
    mock_provider = MagicMock()
    mock_provider_func.return_value = mock_provider
    
    messages = [
        [{"role": "user", "content": "Message 1"}],
        [{"role": "user", "content": "Message 2"}]
    ]
    
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [
        {'custom_id': 'request_0', 'params': {}},
        {'custom_id': 'request_1', 'params': {}}
    ]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = [
        {"result": SpamResult(is_spam=True, confidence=0.9, reason="Spam"), "citations": None},
        {"result": SpamResult(is_spam=False, confidence=0.1, reason="Not spam"), "citations": None}
    ]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    # Verify correct number of requests were prepared
    mock_provider.prepare_batch_requests.assert_called_once()
    call_args = mock_provider.prepare_batch_requests.call_args[0]
    assert len(call_args[0]) == 2  # Two messages
    
    results = job.results()
    assert len(results) == 2
    assert results[0]["result"].is_spam == True
    assert results[1]["result"].is_spam == False


@patch('batchata.core.get_provider_for_model')
def test_batch_without_response_model(mock_provider_func):
    """Test that batch returns raw text when no response_model is provided."""
    # Mock provider for raw text responses
    mock_provider = MagicMock()
    mock_provider_func.return_value = mock_provider
    
    mock_provider.validate_batch.return_value = None
    mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
    mock_provider.create_batch.return_value = "batch_123"
    mock_provider._is_batch_completed.return_value = True
    mock_provider.get_results.return_value = []
    mock_provider.parse_results.return_value = [{"result": "This is a raw text response", "citations": None}]
    
    messages = [[{"role": "user", "content": "Test message"}]]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307"
    )
    
    # Verify that None was passed as response_model
    mock_provider.validate_batch.assert_called_once_with(messages, None)
    
    results = job.results()
    
    # parse_results should be called when getting results
    mock_provider.parse_results.assert_called_once()
    
    assert len(results) == 1
    assert results[0]["result"] == "This is a raw text response"
    assert isinstance(results[0]["result"], str)