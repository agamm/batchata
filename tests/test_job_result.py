"""Tests for JobResult data model."""

import pytest
from pydantic import BaseModel

from batchata.core import JobResult
from batchata.types import Citation


class SampleModel(BaseModel):
    """Sample model for testing structured output."""
    name: str
    value: int


class TestJobResult:
    """Tests for JobResult dataclass."""
    
    def test_successful_result(self):
        """Test successful job result."""
        result = JobResult(
            job_id="test-1",
            response="Generated text",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        assert result.job_id == "test-1"
        assert result.response == "Generated text"
        assert result.is_success is True
        assert result.total_tokens == 150
        assert result.error is None
    
    def test_result_with_structured_output(self):
        """Test result with parsed response."""
        parsed = SampleModel(name="Test", value=42)
        result = JobResult(
            job_id="test-2",
            response="Generated text",
            parsed_response=parsed,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        assert result.parsed_response == parsed
        assert isinstance(result.parsed_response, SampleModel)
    
    def test_result_with_citations(self):
        """Test result with citations."""
        citations = [
            Citation(text="Quote 1", source="Page 1"),
            Citation(text="Quote 2", source="Page 2", metadata={"section": "A"})
        ]
        result = JobResult(
            job_id="test-3",
            response="Generated text",
            citations=citations,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        assert result.citations == citations
        assert len(result.citations) == 2
    
    def test_failed_result(self):
        """Test failed job result."""
        result = JobResult(
            job_id="test-4",
            response="",
            error="API rate limit exceeded"
        )
        assert result.is_success is False
        assert result.error == "API rate limit exceeded"
        assert result.total_tokens == 0
        assert result.cost_usd == 0.0
    
    def test_result_with_parse_error(self):
        """Test result with parsing error."""
        result = JobResult(
            job_id="test-5",
            response="Invalid JSON",
            parsed_response={"error": "Failed to parse response"},
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        assert result.is_success is True  # Job succeeded, only parsing failed
        assert isinstance(result.parsed_response, dict)
        assert "error" in result.parsed_response