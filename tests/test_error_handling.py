"""
Tests for improved error handling scenarios
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from batchata.core import batch, _detect_file_type
from batchata.batch_manager import BatchManager
from batchata.exceptions import (
    FileTooLargeError,
    UnsupportedContentError,
    BatchManagerError
)


class TestFileTypeDetection:
    """Test file type detection functionality."""
    
    def test_detect_pdf(self):
        pdf_bytes = b'%PDF-1.4\n'
        assert _detect_file_type(pdf_bytes) == 'application/pdf'
    
    def test_detect_png(self):
        png_bytes = b'\x89PNG\r\n\x1a\n'
        assert _detect_file_type(png_bytes) == 'image/png'
    
    def test_detect_jpeg(self):
        jpeg_bytes = b'\xff\xd8\xff\xe0'
        assert _detect_file_type(jpeg_bytes) == 'image/jpeg'
    
    def test_detect_gif(self):
        gif_bytes = b'GIF89a'
        assert _detect_file_type(gif_bytes) == 'image/gif'
    
    def test_detect_webp(self):
        webp_bytes = b'RIFF\x00\x00\x00\x00WEBP'
        assert _detect_file_type(webp_bytes) == 'image/webp'
    
    def test_detect_unknown(self):
        unknown_bytes = b'unknown file content'
        assert _detect_file_type(unknown_bytes) == 'application/octet-stream'


class TestEmptyFileHandling:
    """Test empty file error handling."""
    
    def test_empty_file_core_function(self):
        """Test empty file handling in core batch function."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            empty_file = f.name
        
        try:
            with pytest.raises(ValueError, match="File is empty"):
                batch(
                    model="claude-3-5-sonnet-20241022",
                    files=[empty_file],
                    prompt="Test prompt"
                )
        finally:
            Path(empty_file).unlink()
    
    def test_empty_bytes_core_function(self):
        """Test empty bytes handling in core batch function."""
        with pytest.raises(ValueError, match="File is empty: bytes input"):
            batch(
                model="claude-3-5-sonnet-20241022",
                files=[b''],
                prompt="Test prompt"
            )
    
    def test_empty_file_batch_manager(self):
        """Test empty file handling in BatchManager."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            empty_file = f.name
        
        try:
            with pytest.raises(ValueError, match="File is empty"):
                BatchManager(
                    model="claude-3-5-sonnet-20241022",
                    files=[empty_file],
                    prompt="Test prompt"
                )
        finally:
            Path(empty_file).unlink()
    
    def test_empty_bytes_batch_manager(self):
        """Test empty bytes handling in BatchManager."""
        with pytest.raises(ValueError, match="File is empty: bytes input"):
            BatchManager(
                model="claude-3-5-sonnet-20241022",
                files=[b''],
                prompt="Test prompt"
            )


class TestFileSizeValidation:
    """Test file size validation against provider limits."""
    
    @patch('batchata.providers.anthropic.AnthropicBatchProvider.validate_file_size')
    def test_file_size_validation_called(self, mock_validate):
        """Test that file size validation is called during processing."""
        pdf_content = b'%PDF-1.4\n' + b'x' * 1000
        
        batch(
            model="claude-3-5-sonnet-20241022",
            files=[pdf_content],
            prompt="Test prompt"
        )
        
        mock_validate.assert_called()
    
    def test_file_too_large_error(self):
        """Test FileTooLargeError when file exceeds provider limits."""
        # Create a mock provider that always raises FileTooLargeError
        with patch('batchata.core.get_provider_for_model') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.validate_file_size.side_effect = FileTooLargeError(
                "File 'test.pdf' is 35.0MB, exceeds 32MB limit"
            )
            mock_provider.validate_model_capabilities = Mock()  # Mock this too
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(FileTooLargeError, match="exceeds 32MB limit"):
                batch(
                    model="claude-3-5-sonnet-20241022",
                    files=[b'%PDF-1.4\n' + b'x' * 1000],
                    prompt="Test prompt"
                )


class TestImageCitationError:
    """Test image citation error handling."""
    
    def test_citation_on_png_file(self):
        """Test UnsupportedContentError when requesting citations on PNG."""
        png_content = b'\x89PNG\r\n\x1a\n' + b'x' * 100
        
        with pytest.raises(UnsupportedContentError, match="Citations are not supported for image files"):
            batch(
                model="claude-3-5-sonnet-20241022",
                files=[png_content],
                prompt="Test prompt",
                enable_citations=True
            )
    
    def test_citation_on_jpeg_file(self):
        """Test UnsupportedContentError when requesting citations on JPEG."""
        jpeg_content = b'\xff\xd8\xff\xe0' + b'x' * 100
        
        with pytest.raises(UnsupportedContentError, match="Citations are not supported for image files"):
            batch(
                model="claude-3-5-sonnet-20241022",
                files=[jpeg_content],
                prompt="Test prompt",
                enable_citations=True
            )
    
    def test_citation_on_pdf_allowed(self):
        """Test that citations work normally on PDF files."""
        pdf_content = b'%PDF-1.4\n' + b'x' * 100
        
        # This should not raise an error
        batch(
            model="claude-3-5-sonnet-20241022",
            files=[pdf_content],
            prompt="Test prompt",
            enable_citations=True
        )
    
    def test_no_citation_on_image_allowed(self):
        """Test that images work when citations are not requested."""
        png_content = b'\x89PNG\r\n\x1a\n' + b'x' * 100
        
        # This should not raise an error
        batch(
            model="claude-3-5-sonnet-20241022",
            files=[png_content],
            prompt="Test prompt",
            enable_citations=False
        )


class TestProviderFileSizeLimits:
    """Test provider-specific file size limits."""
    
    def test_anthropic_provider_file_size_limit(self):
        """Test that Anthropic provider has correct file size limit."""
        from batchata.providers.anthropic import AnthropicBatchProvider
        
        provider = AnthropicBatchProvider()
        assert provider.get_max_file_size_mb() == 32.0
    
    def test_validate_file_size_within_limit(self):
        """Test file size validation passes for files within limit."""
        from batchata.providers.anthropic import AnthropicBatchProvider
        
        provider = AnthropicBatchProvider()
        small_file = b'x' * (10 * 1024 * 1024)  # 10MB
        
        # Should not raise an error
        provider.validate_file_size(small_file, "test.pdf")
    
    def test_validate_file_size_exceeds_limit(self):
        """Test file size validation fails for files exceeding limit."""
        from batchata.providers.anthropic import AnthropicBatchProvider
        
        provider = AnthropicBatchProvider()
        large_file = b'x' * (35 * 1024 * 1024)  # 35MB
        
        with pytest.raises(FileTooLargeError, match="exceeds 32MB limit"):
            provider.validate_file_size(large_file, "large_test.pdf")