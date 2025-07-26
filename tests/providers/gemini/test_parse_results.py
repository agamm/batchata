"""Tests for Gemini result parsing."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from batchata.providers.gemini.parse_results import parse_results
from batchata.core.job import Job


class SampleResponse(BaseModel):
    """Sample response model for structured output."""
    name: str
    age: int
    score: float


class TestParseResults:
    """Test Gemini result parsing functionality."""
    
    def test_successful_result_parsing(self):
        """Test parsing successful Gemini batch results."""
        # Mock Gemini batch result format
        results = [
            {
                "job_id": "job-1",
                "response": MagicMock(
                    text="The answer is 42",
                    usage_metadata=MagicMock(
                        prompt_token_count=50,
                        candidates_token_count=25,
                        total_token_count=75
                    )
                ),
                "error": None
            }
        ]
        
        # Create job mapping
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "What is the answer?"}]
            )
        }
        
        # Parse results with mocked cost calculation
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.002):
            job_results = parse_results(results, job_mapping, batch_discount=0.5, batch_id="batch-123")
        
        # Verify parsed results
        assert len(job_results) == 1
        result = job_results[0]
        assert result.job_id == "job-1"
        assert result.raw_response == "The answer is 42"
        assert result.input_tokens == 50
        assert result.output_tokens == 25
        assert result.cost_usd == 0.001  # 0.002 * (1 - 0.5) batch discount
        assert result.error is None
        assert result.batch_id == "batch-123"
        assert result.citations is None  # Gemini doesn't support citations yet
    
    def test_error_result_parsing(self):
        """Test parsing failed Gemini batch results."""
        # Test different error scenarios
        error_results = [
            # Direct error in result
            {
                "job_id": "job-1",
                "response": None,
                "error": "Invalid authentication"
            },
            # Missing response
            {
                "job_id": "job-2",
                "response": None,
                "error": None
            }
        ]
        
        # Create job mapping
        job_mapping = {
            f"job-{i}": Job(
                id=f"job-{i}",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": f"Test {i}"}]
            )
            for i in range(1, 3)
        }
        
        # Parse error results
        job_results = parse_results(error_results, job_mapping)
        
        # Verify error handling
        assert len(job_results) == 2
        
        # Check specific error messages
        assert job_results[0].error == "Invalid authentication"
        assert job_results[1].error == "No response in result"
        
        # All should have empty responses
        for result in job_results:
            assert result.raw_response == ""
            assert result.input_tokens == 0
            assert result.output_tokens == 0
    
    def test_structured_output_parsing(self):
        """Test parsing structured output with Pydantic models."""
        
        # Mock results with structured JSON output
        results = [
            {
                "job_id": "job-1",
                "response": MagicMock(
                    text='{"name": "Alice", "age": 30, "score": 95.5}',
                    usage_metadata=MagicMock(
                        prompt_token_count=40,
                        candidates_token_count=20
                    )
                ),
                "error": None
            },
            {
                "job_id": "job-2", 
                "response": MagicMock(
                    text='Here is the data:\n{"name": "Bob", "age": 25, "score": 88.0}',
                    usage_metadata=MagicMock(
                        prompt_token_count=40,
                        candidates_token_count=30
                    )
                ),
                "error": None
            }
        ]
        
        # Create jobs with response models
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Give me data"}],
                response_model=SampleResponse
            ),
            "job-2": Job(
                id="job-2",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Give me more data"}],
                response_model=SampleResponse
            )
        }
        
        # Parse results
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.0):
            job_results = parse_results(results, job_mapping)
        
        # Verify structured output parsing
        assert len(job_results) == 2
        
        # Check first result - direct JSON
        assert job_results[0].parsed_response is not None
        assert isinstance(job_results[0].parsed_response, SampleResponse)
        assert job_results[0].parsed_response.name == "Alice"
        assert job_results[0].parsed_response.age == 30
        assert job_results[0].parsed_response.score == 95.5
        
        # Check second result - JSON in text
        assert job_results[1].parsed_response is not None
        assert isinstance(job_results[1].parsed_response, SampleResponse)
        assert job_results[1].parsed_response.name == "Bob"
        assert job_results[1].parsed_response.age == 25
        assert job_results[1].parsed_response.score == 88.0
    
    def test_complex_response_format(self):
        """Test parsing complex response format with candidates."""
        # Mock Gemini response with candidates structure
        mock_part = MagicMock()
        mock_part.text = "Response from candidate"
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.text = None  # Force using candidates path
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=15
        )
        
        results = [
            {
                "job_id": "job-1",
                "response": mock_response,
                "error": None
            }
        ]
        
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Test"}]
            )
        }
        
        job_results = parse_results(results, job_mapping)
        
        assert len(job_results) == 1
        assert job_results[0].raw_response == "Response from candidate"
        assert job_results[0].input_tokens == 10
        assert job_results[0].output_tokens == 15
    
    def test_invalid_structured_output(self):
        """Test handling invalid JSON in structured output."""
        results = [
            {
                "job_id": "job-1",
                "response": MagicMock(
                    text='Invalid JSON content that cannot be parsed',
                    usage_metadata=MagicMock(
                        prompt_token_count=10,
                        candidates_token_count=20
                    )
                ),
                "error": None
            }
        ]
        
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Give me data"}],
                response_model=SampleResponse
            )
        }
        
        job_results = parse_results(results, job_mapping)
        
        # Should fall back to text response when JSON parsing fails
        assert len(job_results) == 1
        assert job_results[0].parsed_response is None
        assert job_results[0].raw_response == "Invalid JSON content that cannot be parsed"