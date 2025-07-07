import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from ai_batch import batch


class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str


def test_batch_empty_messages():
    """Test batch function with empty messages list."""
    results = batch(
        messages=[],
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert results == []


def test_batch_invalid_model():
    """Test batch function with invalid model name - now handled by Anthropic API."""
    messages = [[{"role": "user", "content": "Test message"}]]
    
    # Since we removed model validation, this will now be handled by the API
    # The test should pass but may raise a different error from the API
    with pytest.raises((ValueError, RuntimeError)):
        batch(
            messages=messages,
            model="invalid-model",
            response_model=SpamResult
        )


def test_batch_missing_required_params():
    """Test batch function with missing required parameters."""
    with pytest.raises(TypeError):
        batch()
    
    with pytest.raises(TypeError):
        batch(messages=[])
    
    with pytest.raises(TypeError):
        batch(messages=[], model="claude-3-haiku-20240307")


def test_batch_with_empty_messages():
    """Test that batch function works with empty messages."""
    results = batch(
        messages=[],
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert results == []


@patch.dict(os.environ, {}, clear=True)
def test_missing_api_key():
    """Test that missing API key raises appropriate error."""
    messages = [[{"role": "user", "content": "Test message"}]]
    
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
        batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=SpamResult
        )


@patch('instructor.handle_response_model')
@patch('ai_batch.Anthropic')
@patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
def test_batch_creates_batch_job(mock_anthropic, mock_handle_response):
    """Test that batch function creates a batch job."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.batches.create.return_value.id = "batch_123"
    mock_client.messages.batches.retrieve.return_value.processing_status = "ended"
    
    # Mock batch results
    mock_result = MagicMock()
    mock_result.result.type = "succeeded"
    mock_result.result.message.content = [MagicMock()]
    mock_result.result.message.content[0].text = '{"is_spam": true, "confidence": 0.9, "reason": "Test"}'
    mock_client.messages.batches.results.return_value = [mock_result]
    
    # Mock instructor response handling
    mock_handle_response.return_value = (None, {'system': [{'type': 'text', 'text': 'JSON schema'}]})
    
    messages = [[{"role": "user", "content": "Test message"}]]
    
    result = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    # Verify the batch was created
    mock_client.messages.batches.create.assert_called_once()
    
    assert len(result) == 1
    assert result[0].is_spam == True


@patch('instructor.handle_response_model')
@patch('ai_batch.Anthropic')
@patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
def test_batch_multiple_messages(mock_anthropic, mock_handle_response):
    """Test that batch processes multiple messages correctly."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.batches.create.return_value.id = "batch_123"
    mock_client.messages.batches.retrieve.return_value.processing_status = "ended"
    
    # Mock batch results
    mock_result1 = MagicMock()
    mock_result1.result.type = "succeeded"
    mock_result1.result.message.content = [MagicMock()]
    mock_result1.result.message.content[0].text = '{"is_spam": true, "confidence": 0.9, "reason": "Spam"}'
    
    mock_result2 = MagicMock()
    mock_result2.result.type = "succeeded"
    mock_result2.result.message.content = [MagicMock()]
    mock_result2.result.message.content[0].text = '{"is_spam": false, "confidence": 0.1, "reason": "Not spam"}'
    
    mock_client.messages.batches.results.return_value = [mock_result1, mock_result2]
    
    # Mock instructor response handling
    mock_handle_response.return_value = (None, {'system': [{'type': 'text', 'text': 'JSON schema'}]})
    
    messages = [
        [{"role": "user", "content": "Message 1"}],
        [{"role": "user", "content": "Message 2"}]
    ]
    
    results = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert len(results) == 2
    assert results[0].is_spam == True
    assert results[1].is_spam == False