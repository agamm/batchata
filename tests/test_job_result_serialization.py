"""Test JobResult serialization with date fields."""

import json
from datetime import date
from pydantic import BaseModel, Field

from batchata.core.job_result import JobResult


class MockPropertyData(BaseModel):
    """Mock Pydantic model with date fields for testing."""
    property_name: str = Field(..., description="Property name")
    appraisal_date: date = Field(..., description="Date appraisal was performed")
    last_sale_date: date | None = Field(None, description="Date of last sale")
    appraised_value: float | None = Field(None, description="Appraised value")


def test_job_result_serialization_with_dates():
    """Test that JobResult can serialize Pydantic models with date fields to JSON."""
    # Create a mock Pydantic model with date fields
    property_data = MockPropertyData(
        property_name="Test Property",
        appraisal_date=date(2023, 10, 20),
        last_sale_date=date(2022, 5, 15),
        appraised_value=1500000.0
    )
    
    # Create JobResult with the mock data
    job_result = JobResult(
        job_id="test-job-123",
        raw_response="Test raw response",
        parsed_response=property_data,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.15
    )
    
    # Test to_dict() method
    result_dict = job_result.to_dict()
    
    # Verify the parsed_response contains serialized dates as strings
    assert result_dict["parsed_response"]["appraisal_date"] == "2023-10-20"
    assert result_dict["parsed_response"]["last_sale_date"] == "2022-05-15"
    assert result_dict["parsed_response"]["property_name"] == "Test Property"
    assert result_dict["parsed_response"]["appraised_value"] == 1500000.0
    
    # Test that the result can be serialized to JSON (this was failing before the fix)
    json_str = json.dumps(result_dict, indent=2)
    
    # Verify we can parse it back
    parsed_back = json.loads(json_str)
    assert parsed_back["parsed_response"]["appraisal_date"] == "2023-10-20"
    assert parsed_back["parsed_response"]["last_sale_date"] == "2022-05-15"
    
    # Test save_to_json method
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # This should not raise an exception
        job_result.save_to_json(tmp_path)
        
        # Verify the file was created and contains valid JSON
        with open(tmp_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["parsed_response"]["appraisal_date"] == "2023-10-20"
        assert saved_data["parsed_response"]["last_sale_date"] == "2022-05-15"
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_job_result_serialization_with_dict_response():
    """Test that JobResult handles dict responses correctly (should not change)."""
    # Test with dict response (no Pydantic model)
    dict_response = {
        "property_name": "Test Property",
        "appraisal_date": "2023-10-20",  # Already a string
        "appraised_value": 1500000.0
    }
    
    job_result = JobResult(
        job_id="test-job-dict",
        raw_response="Test raw response",
        parsed_response=dict_response,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.15
    )
    
    # Test to_dict() method
    result_dict = job_result.to_dict()
    
    # Should be unchanged since it's already a dict
    assert result_dict["parsed_response"]["appraisal_date"] == "2023-10-20"
    assert result_dict["parsed_response"]["property_name"] == "Test Property"
    
    # Should serialize to JSON without issues
    json_str = json.dumps(result_dict, indent=2)
    parsed_back = json.loads(json_str)
    assert parsed_back["parsed_response"]["appraisal_date"] == "2023-10-20"


if __name__ == "__main__":
    test_job_result_serialization_with_dates()
    test_job_result_serialization_with_dict_response()
    print("All tests passed!")