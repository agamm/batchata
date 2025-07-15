"""Tests for PDF validation in batch jobs."""

import pytest
import tempfile
from pathlib import Path

from batchata.core.batch import Batch
from batchata.utils.pdf import create_pdf
from batchata.exceptions import ValidationError


class TestPdfValidation:
    """Test PDF validation when citations are enabled."""
    
    def test_image_pdf_with_citations_fails(self):
        """Test image PDF with citations should fail."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"not a real pdf")  # Score 0.0
            tmp.flush()
            
            batch = Batch("/tmp/state.json", "/tmp/results")
            batch.defaults(model="claude-3-5-sonnet-latest")
            
            with pytest.raises(ValidationError, match="appears to be image-only"):
                batch.add_job(file=tmp.name, prompt="Test", enable_citations=True)
            
            Path(tmp.name).unlink()

    def test_image_pdf_without_citations_works(self):
        """Test image PDF without citations should work."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"not a real pdf")  # Score 0.0
            tmp.flush()
            
            batch = Batch("/tmp/state.json", "/tmp/results")
            batch.defaults(model="claude-3-5-sonnet-latest")
            batch.add_job(file=tmp.name, prompt="Test", enable_citations=False)
            
            assert len(batch.jobs) == 1
            Path(tmp.name).unlink()
    
    def test_textual_pdf_with_citations_works(self):
        """Test that textual PDFs work with citations."""
        pages = ["This is a text document with plenty of content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            batch = Batch("/tmp/state.json", "/tmp/results")
            batch.defaults(model="claude-3-5-sonnet-latest")
            batch.add_job(file=tmp.name, prompt="Test", enable_citations=True)
            
            assert len(batch.jobs) == 1
            Path(tmp.name).unlink()
    
    def test_non_pdf_bypasses_validation(self):
        """Test that non-PDF files bypass PDF validation."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Text content")
            tmp.flush()
            
            batch = Batch("/tmp/state.json", "/tmp/results")
            batch.defaults(model="claude-3-5-sonnet-latest")
            batch.add_job(file=tmp.name, prompt="Test", enable_citations=True)
            
            assert len(batch.jobs) == 1
            Path(tmp.name).unlink()