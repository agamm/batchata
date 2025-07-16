"""Tests for PDF utilities."""

import pytest
from pathlib import Path
import tempfile

from batchata.utils.pdf import create_pdf, is_textual_pdf


class TestCreatePdf:
    """Test PDF creation functionality."""
    
    def test_create_single_page(self):
        """Test creating a single page PDF."""
        pages = ["Hello World"]
        pdf_bytes = create_pdf(pages)
        
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b'%PDF-1.3')
        assert pdf_bytes.endswith(b'%%EOF\n')
        assert b'Hello World' in pdf_bytes
    
    def test_create_multi_page(self):
        """Test creating a multi-page PDF."""
        pages = ["Page 1", "Page 2", "Page 3"]
        pdf_bytes = create_pdf(pages)
        
        assert isinstance(pdf_bytes, bytes)
        assert b'Page 1' in pdf_bytes
        assert b'Page 2' in pdf_bytes
        assert b'Page 3' in pdf_bytes
    
    def test_empty_pages_raises_error(self):
        """Test that empty pages list raises ValueError."""
        with pytest.raises(ValueError, match="At least one page is required"):
            create_pdf([])


class TestIsTextualPdf:
    """Test PDF textual score detection functionality."""
    
    def test_textual_pdf_high_score(self):
        """Test that textual PDFs get high textual scores."""
        # Create a text-heavy PDF
        pages = [
            "This is a text-based PDF with lots of content",
            "Page 2 has even more text content",
            "Page 3 continues with readable text"
        ]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Should get high textual score
            score = is_textual_pdf(tmp.name)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert score > 0.5  # Should be reasonably textual
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_empty_pdf_zero_score(self):
        """Test that empty/malformed PDFs get zero score."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"not a real pdf")
            tmp.flush()
            
            # Should get zero score
            score = is_textual_pdf(tmp.name)
            assert score == 0.0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_threshold_parameters(self):
        """Test that threshold parameters affect scoring."""
        pages = ["Some text content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Different thresholds
            score_default = is_textual_pdf(tmp.name)
            score_strict = is_textual_pdf(
                tmp.name, 
                text_page_thresh=0.01,  # Very strict
            )
            
            # Both should be valid scores
            assert isinstance(score_default, float)
            assert isinstance(score_strict, float)
            assert 0.0 <= score_default <= 1.0
            assert 0.0 <= score_strict <= 1.0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_path_types(self):
        """Test that function accepts both string and Path objects."""
        pages = ["Test content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with string path
            score_str = is_textual_pdf(tmp.name)
            
            # Test with Path object
            score_path = is_textual_pdf(Path(tmp.name))
            
            # Results should be identical
            assert score_str == score_path
            assert isinstance(score_str, float)
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_score_ranges(self):
        """Test score interpretation ranges."""
        pages = ["Some text content for testing"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            score = is_textual_pdf(tmp.name)
            
            # Verify score is in valid range
            assert 0.0 <= score <= 1.0
            
            # For our simple text PDF, should be reasonably high
            assert score > 0.3  # Should detect some text content
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_nonexistent_file_returns_zero(self):
        """Test that nonexistent files return 0.0 score."""
        score = is_textual_pdf("/nonexistent/path/file.pdf")
        assert score == 0.0