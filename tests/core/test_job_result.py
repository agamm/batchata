"""Tests for JobResult class.

Testing:
1. Success/failure detection via is_success property
2. Token calculation via total_tokens property
3. Serialization and deserialization via to_dict/from_dict methods
4. Citation handling in serialization
5. Structured output handling
"""

import pytest
import json
from typing import List
from pydantic import BaseModel

from batchata.core.job_result import JobResult
from batchata.types import Citation


class TestJobResult:
    """Test JobResult functionality."""
    
    def test_is_success_property(self):
        """Test the is_success property correctly identifies success/failure."""
        # Successful result (no error)
        success_result = JobResult(
            job_id="success-job",
            raw_response="Success response",
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=20
        )
        assert success_result.is_success is True
        
        # Failed result (has error)
        failed_result = JobResult(
            job_id="failed-job",
            raw_response="",
            cost_usd=0.0,
            input_tokens=10,
            output_tokens=0,
            error="API timeout"
        )
        assert failed_result.is_success is False
        
        # Edge case: empty error string should still be considered failure
        empty_error_result = JobResult(
            job_id="empty-error-job",
            raw_response="Some response",
            error=""
        )
        assert empty_error_result.is_success is False
    
    def test_total_tokens_calculation(self):
        """Test the total_tokens property correctly sums input and output tokens."""
        result = JobResult(
            job_id="token-test",
            raw_response="Test response",
            input_tokens=150,
            output_tokens=75
        )
        assert result.total_tokens == 225
        
        # Test with zero tokens
        zero_result = JobResult(
            job_id="zero-tokens",
            raw_response="",
            input_tokens=0,
            output_tokens=0
        )
        assert zero_result.total_tokens == 0
        
        # Test with only input tokens
        input_only = JobResult(
            job_id="input-only",
            raw_response="",
            input_tokens=100,
            output_tokens=0
        )
        assert input_only.total_tokens == 100
    
    def test_serialization_and_deserialization(self):
        """Test to_dict and from_dict methods work correctly."""
        # Create a result with all fields populated
        original = JobResult(
            job_id="serialize-test",
            raw_response="Test response content",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0025,
            error=None
        )
        
        # Serialize to dict
        data = original.to_dict()
        
        # Verify dict structure
        assert data["job_id"] == "serialize-test"
        assert data["raw_response"] == "Test response content"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["cost_usd"] == 0.0025
        assert data["error"] is None
        assert data["citations"] is None
        assert data["parsed_response"] is None
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)
        
        # Deserialize back to JobResult
        restored = JobResult.from_dict(loaded_data)
        
        # Verify all fields match
        assert restored.job_id == original.job_id
        assert restored.raw_response == original.raw_response
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.cost_usd == original.cost_usd
        assert restored.error == original.error
        assert restored.citations == original.citations
        assert restored.parsed_response == original.parsed_response
    
    def test_citation_serialization(self):
        """Test that citations are properly serialized and deserialized."""
        citations = [
            Citation(text="First citation", source="doc1", page=1),
            Citation(text="Second citation", source="doc2", page=2, metadata={"section": "intro"})
        ]
        
        result = JobResult(
            job_id="citation-test",
            raw_response="Response with citations",
            citations=citations
        )
        
        # Serialize
        data = result.to_dict()
        
        # Check citations are properly converted to dicts
        assert len(data["citations"]) == 2
        assert data["citations"][0]["text"] == "First citation"
        assert data["citations"][0]["source"] == "doc1"
        assert data["citations"][0]["page"] == 1
        assert data["citations"][1]["metadata"] == {"section": "intro"}
        
        # Deserialize
        restored = JobResult.from_dict(data)
        
        # Verify citations are reconstructed correctly
        assert len(restored.citations) == 2
        assert restored.citations[0].text == "First citation"
        assert restored.citations[0].source == "doc1"
        assert restored.citations[0].page == 1
        assert restored.citations[1].metadata == {"section": "intro"}
    
    def test_structured_output_handling(self):
        """Test handling of structured output (Pydantic models vs dicts)."""
        # Test with Pydantic model
        class TestModel(BaseModel):
            name: str
            value: int
        
        model_instance = TestModel(name="test", value=42)
        
        result_with_model = JobResult(
            job_id="model-test",
            raw_response="{\"name\": \"test\", \"value\": 42}",
            parsed_response=model_instance
        )
        
        # Serialize - Pydantic models should be serialized to dict
        data = result_with_model.to_dict()
        assert data["parsed_response"] == {"name": "test", "value": 42}  # Pydantic models are serialized
        
        # Test with dict (which should be serialized)
        dict_response = {"name": "test", "value": 42}
        result_with_dict = JobResult(
            job_id="dict-test",
            raw_response="{\"name\": \"test\", \"value\": 42}",
            parsed_response=dict_response
        )
        
        data = result_with_dict.to_dict()
        assert data["parsed_response"] == dict_response
        
        # Deserialize and verify
        restored = JobResult.from_dict(data)
        assert restored.parsed_response == dict_response
    
    def test_error_result_serialization(self):
        """Test serialization of failed results with errors."""
        error_result = JobResult(
            job_id="error-test",
            raw_response="",
            cost_usd=0.0,
            input_tokens=50,
            output_tokens=0,
            error="Connection timeout after 30 seconds"
        )
        
        # Test is_success property
        assert error_result.is_success is False
        
        # Serialize
        data = error_result.to_dict()
        assert data["error"] == "Connection timeout after 30 seconds"
        
        # Deserialize
        restored = JobResult.from_dict(data)
        assert restored.error == "Connection timeout after 30 seconds"
        assert restored.is_success is False
    
    def test_default_values_in_deserialization(self):
        """Test that missing fields get proper default values during deserialization."""
        # Minimal data dict (missing optional fields)
        minimal_data = {
            "job_id": "minimal-test",
            "raw_response": "Minimal response"
        }
        
        result = JobResult.from_dict(minimal_data)
        
        # Check defaults are applied
        assert result.job_id == "minimal-test"
        assert result.raw_response == "Minimal response"
        assert result.input_tokens == 0  # Default
        assert result.output_tokens == 0  # Default
        assert result.cost_usd == 0.0  # Default
        assert result.error is None  # Default
        assert result.citations is None  # Default
        assert result.parsed_response is None  # Default
        
        # Verify computed properties work with defaults
        assert result.total_tokens == 0
        assert result.is_success is True  # No error = success