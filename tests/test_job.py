"""Tests for Job data model."""

import pytest
from pathlib import Path
from pydantic import BaseModel

from batchata.core import Job


class SampleModel(BaseModel):
    """Sample model for testing structured output."""
    name: str
    value: int


class TestJob:
    """Tests for Job dataclass."""
    
    def test_job_with_messages(self):
        """Test creating a job with messages."""
        job = Job(
            id="test-1",
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert job.id == "test-1"
        assert job.model == "claude-sonnet-4-20250514"
        assert job.messages == [{"role": "user", "content": "Hello"}]
        assert job.temperature == 0.7
        assert job.max_tokens == 1000
    
    def test_job_with_file(self):
        """Test creating a job with file input."""
        job = Job(
            id="test-2",
            model="gpt-4",
            file=Path("test.pdf"),
            prompt="Summarize this document"
        )
        assert job.file == Path("test.pdf")
        assert job.prompt == "Summarize this document"
    
    def test_job_with_structured_output(self):
        """Test job with response model."""
        job = Job(
            id="test-3",
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Extract data"}],
            response_model=SampleModel,
            enable_citations=True
        )
        assert job.response_model == SampleModel
        assert job.enable_citations is True
    
    def test_job_validation_errors(self):
        """Test job validation errors."""
        # Both messages and file
        with pytest.raises(ValueError, match="Provide either messages OR file\\+prompt"):
            Job(
                id="test-4",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                file=Path("test.pdf"),
                prompt="Summarize"
            )
        
        # Neither messages nor file
        with pytest.raises(ValueError, match="Must provide either messages or file\\+prompt"):
            Job(id="test-5", model="claude-sonnet-4-20250514")
        
        # File without prompt
        with pytest.raises(ValueError, match="File input requires a prompt"):
            Job(
                id="test-6",
                model="claude-sonnet-4-20250514",
                file=Path("test.pdf")
            )