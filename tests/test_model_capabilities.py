"""Test model capability validation"""

import pytest
from batchata.core import batch
from batchata.providers.anthropic import AnthropicBatchProvider


class TestModelCapabilities:
    """Test model capability validation"""
    
    def test_model_supports_files(self):
        """Test checking if models support files"""
        provider = AnthropicBatchProvider()
        
        # File-capable models
        assert "claude-3-5-sonnet-20241022" in provider.FILE_CAPABLE_MODELS
        assert "claude-3-opus-20240229" in provider.FILE_CAPABLE_MODELS
        assert "claude-3-sonnet-20240229" in provider.FILE_CAPABLE_MODELS
        
        # Message-only models
        assert "claude-3-haiku-20240307" not in provider.FILE_CAPABLE_MODELS
    
    def test_haiku_with_files_raises_error(self):
        """Test that using Haiku with files raises a clear error"""
        with pytest.raises(ValueError, match="does not support file/document input"):
            batch(
                model="claude-3-haiku-20240307",
                files=[b"fake pdf content"],
                prompt="Process this document"
            )
    
    def test_file_capable_model_with_files_works(self):
        """Test that file-capable models accept files without error during validation"""
        from tests.utils.pdf_utils import create_pdf
        
        # This should not raise an error during validation
        # (it might fail later due to API issues, but validation should pass)
        pdf_bytes = create_pdf(["Test content"])
        
        # This should pass validation (though we won't wait for completion)
        try:
            batch_job = batch(
                model="claude-3-5-sonnet-20241022",
                files=[pdf_bytes],
                prompt="Process this document"
            )
            # If we get here, validation passed
            assert batch_job is not None
        except ValueError as e:
            # Should not get validation errors about model capabilities
            assert "does not support file" not in str(e)
    
    def test_unsupported_model_raises_error(self):
        """Test that unsupported models raise an error"""
        with pytest.raises(ValueError, match="No provider supports model"):
            batch(
                model="gpt-4",  # Not an Anthropic model
                messages=[[{"role": "user", "content": "Hello"}]]
            )
    
    def test_both_messages_and_files_raises_error(self):
        """Test that providing both messages and files raises an error"""
        with pytest.raises(ValueError, match="Cannot provide both messages and files"):
            batch(
                model="claude-3-5-sonnet-20241022",
                messages=[[{"role": "user", "content": "Hello"}]],
                files=[b"fake content"],
                prompt="Process this"
            )
    
    def test_neither_messages_nor_files_raises_error(self):
        """Test that providing neither messages nor files raises an error"""
        with pytest.raises(ValueError, match="Must provide either messages or files"):
            batch(
                model="claude-3-5-sonnet-20241022"
            )