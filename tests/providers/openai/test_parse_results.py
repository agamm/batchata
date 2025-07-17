"""Tests for OpenAI result parsing.

Testing:
1. Successful result parsing with all fields
2. Error handling for failed requests
3. Structured output parsing with Pydantic models
"""

import pytest
from pydantic import BaseModel
from unittest.mock import patch

from batchata.providers.openai.parse_results import parse_results
from batchata.core.job import Job


class TestParseResults:
    """Test OpenAI result parsing functionality."""
    
    def test_successful_result_parsing(self):
        """Test parsing successful OpenAI batch results."""
        # Mock OpenAI batch result format
        results = [
            {
                "custom_id": "job-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The answer is 42"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 25,
                            "total_tokens": 75
                        }
                    }
                }
            }
        ]
        
        # Create job mapping
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": "What is the answer?"}]
            )
        }
        
        # Parse results with mocked cost calculation
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.001):
            job_results = parse_results(results, job_mapping, batch_discount=0.5, batch_id="batch-123")
        
        # Verify parsed results
        assert len(job_results) == 1
        result = job_results[0]
        assert result.job_id == "job-1"
        assert result.raw_response == "The answer is 42"
        assert result.input_tokens == 50
        assert result.output_tokens == 25
        assert result.cost_usd == 0.001  # (0.001 + 0.001) * 0.5 batch discount
        assert result.error is None
        assert result.batch_id == "batch-123"
        assert result.citations is None  # OpenAI doesn't support citations
    
    def test_error_result_parsing(self):
        """Test parsing failed OpenAI batch results."""
        # Test different error scenarios
        error_results = [
            # API error response
            {
                "custom_id": "job-1",
                "error": {
                    "message": "Invalid authentication",
                    "type": "authentication_error"
                }
            },
            # HTTP error status
            {
                "custom_id": "job-2", 
                "response": {
                    "status_code": 429,
                    "body": {"error": {"message": "Rate limit exceeded"}}
                }
            },
            # Missing response
            {
                "custom_id": "job-3"
            }
        ]
        
        # Create job mapping
        job_mapping = {
            f"job-{i}": Job(
                id=f"job-{i}",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": f"Test {i}"}]
            )
            for i in range(1, 4)
        }
        
        # Parse error results
        job_results = parse_results(error_results, job_mapping)
        
        # Verify error handling
        assert len(job_results) == 3
        
        # Check specific error messages
        assert job_results[0].error == "Request failed: Invalid authentication"
        assert job_results[1].error == "HTTP error: 429"
        assert job_results[2].error == "No response in batch result"
        
        # All should have empty responses
        for result in job_results:
            assert result.raw_response == ""
            assert result.input_tokens == 0
            assert result.output_tokens == 0
    
    def test_structured_output_parsing(self):
        """Test parsing structured output with Pydantic models."""
        
        # Define a test model
        class TestResponse(BaseModel):
            name: str
            age: int
            score: float
        
        # Mock results with structured JSON output
        results = [
            {
                "custom_id": "job-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                "content": '{"name": "Alice", "age": 30, "score": 95.5}'
                            }
                        }],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 20}
                    }
                }
            },
            {
                "custom_id": "job-2",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                # Test JSON in markdown code block
                                "content": 'Here is the data:\n```json\n{"name": "Bob", "age": 25, "score": 88.0}\n```'
                            }
                        }],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 30}
                    }
                }
            }
        ]
        
        # Create jobs with response models
        job_mapping = {
            "job-1": Job(
                id="job-1",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": "Give me data"}],
                response_model=TestResponse
            ),
            "job-2": Job(
                id="job-2",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": "Give me more data"}],
                response_model=TestResponse
            )
        }
        
        # Parse results
        with patch('tokencost.calculate_cost_by_tokens', return_value=0.0):
            job_results = parse_results(results, job_mapping)
        
        # Verify structured output parsing
        assert len(job_results) == 2
        
        # Check first result - direct JSON
        assert job_results[0].parsed_response is not None
        assert isinstance(job_results[0].parsed_response, TestResponse)
        assert job_results[0].parsed_response.name == "Alice"
        assert job_results[0].parsed_response.age == 30
        assert job_results[0].parsed_response.score == 95.5
        
        # Check second result - JSON in markdown
        assert job_results[1].parsed_response is not None
        assert isinstance(job_results[1].parsed_response, TestResponse)
        assert job_results[1].parsed_response.name == "Bob"
        assert job_results[1].parsed_response.age == 25
        assert job_results[1].parsed_response.score == 88.0