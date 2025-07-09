"""
Test batch validation limits for Anthropic provider.

Tests the validation of batch size and request count limits.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.providers.anthropic import AnthropicBatchProvider
from src import batch


class TestBatchValidation:
    
    def test_max_requests_limit(self):
        """Test that batches with too many requests are rejected."""
        provider = AnthropicBatchProvider()
        
        # Create messages list that exceeds MAX_REQUESTS (100,000)
        large_message_count = 100_001
        messages = [
            [{"role": "user", "content": "test"}] 
            for _ in range(large_message_count)
        ]
        
        with pytest.raises(ValueError, match="Too many requests: 100001 > 100000"):
            provider.validate_batch(messages, None)
    
    def test_max_size_limit(self):
        """Test that batches that are too large are rejected."""
        provider = AnthropicBatchProvider()
        
        # Create a message that when multiplied will exceed 256MB
        # 256MB = 268,435,456 bytes
        large_content = "x" * (10 * 1024 * 1024)  # 10MB per message
        messages = [
            [{"role": "user", "content": large_content}]
            for _ in range(30)  # 30 * 10MB = 300MB > 256MB
        ]
        
        with pytest.raises(ValueError, match="Batch too large.*MB > 256MB"):
            provider.validate_batch(messages, None)
    
    def test_valid_batch_passes(self):
        """Test that valid batches pass validation."""
        provider = AnthropicBatchProvider()
        
        # Small batch should pass
        messages = [
            [{"role": "user", "content": "test message"}]
            for _ in range(10)
        ]
        
        # Should not raise any exception
        provider.validate_batch(messages, None)
    
    def test_empty_batch_skips_validation(self):
        """Test that empty batches skip validation (early return)."""
        provider = AnthropicBatchProvider()
        
        # Empty batch should return early without validation
        provider.validate_batch([], None)
    
    def test_batch_size_calculation(self):
        """Test that batch size is calculated correctly."""
        provider = AnthropicBatchProvider()
        
        # Create messages with known sizes
        messages = [
            [{"role": "user", "content": "a" * 1000}],  # ~1KB
            [{"role": "user", "content": "b" * 2000}],  # ~2KB  
            [{"role": "user", "content": "c" * 3000}],  # ~3KB
        ]
        
        # This should pass (total ~6KB << 256MB)
        provider.validate_batch(messages, None)
        
        # Test the edge case near the limit
        # Create messages that total just under 256MB
        large_content = "x" * (250 * 1024 * 1024 // 1000)  # ~250KB per message
        near_limit_messages = [
            [{"role": "user", "content": large_content}]
            for _ in range(1000)  # 1000 * 250KB = ~250MB
        ]
        
        # Should pass (under limit)
        provider.validate_batch(near_limit_messages, None)
    
    @patch('src.core.AnthropicBatchProvider')
    def test_batch_function_validation_integration(self, mock_provider_class):
        """Test that batch() function calls validation."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        
        # Mock validation to raise an error
        mock_provider.validate_batch.side_effect = ValueError("Test validation error")
        mock_provider.has_citations_enabled.return_value = False
        
        messages = [
            [{"role": "user", "content": "test"}]
        ]
        
        with pytest.raises(ValueError, match="Test validation error"):
            batch(
                messages=messages,
                model="claude-3-haiku-20240307"
            )
        
        # Verify validation was called
        mock_provider.validate_batch.assert_called_once()
    
    def test_invalid_message_format(self):
        """Test that invalid message formats are rejected."""
        provider = AnthropicBatchProvider()
        
        # Test missing role field
        with pytest.raises(ValueError, match="missing required 'role' field"):
            provider.validate_batch([[{"content": "test"}]], None)
        
        # Test missing content field
        with pytest.raises(ValueError, match="missing required 'content' field"):
            provider.validate_batch([[{"role": "user"}]], None)
        
        # Test invalid role
        with pytest.raises(ValueError, match="has invalid role 'invalid_role'"):
            provider.validate_batch([[{"role": "invalid_role", "content": "test"}]], None)
        
        # Test non-dict message (cast to bypass type checking)
        with pytest.raises(ValueError, match="must be a dictionary"):
            # type: ignore - intentionally invalid format for testing
            provider.validate_batch([["not_a_dict"]], None)  # type: ignore
        
        # Test non-list conversation (cast to bypass type checking)
        with pytest.raises(ValueError, match="must be a list of message objects"):
            # type: ignore - intentionally invalid format for testing
            provider.validate_batch([{"role": "user", "content": "test"}], None)  # type: ignore