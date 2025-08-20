"""Test citation mapping with real block patterns from Anthropic responses."""

from typing import List, Tuple, Any, Dict
from pydantic import BaseModel, Field
from datetime import date

from batchata.providers.anthropic.parse_results import _parse_content
from batchata.providers.anthropic.citation_mapper import map_citations_to_fields
from batchata.types import Citation


class MockJob:
    """Mock job object for testing."""
    def __init__(self, enable_citations=True):
        self.enable_citations = enable_citations


class MockPropertyData(BaseModel):
    """Mock Pydantic model matching the real property data structure."""
    property_name: str = Field(..., description="Property name")
    land_area_square_feet: float | None = Field(None, description="Land area in square feet") 


class MockBlock:
    """Mock block object matching Anthropic API structure."""
    def __init__(self, text: str, citations: List[Dict] = None):
        self.text = text
        self.citations = citations or []
        self.type = "text"


class MockCitation:
    """Mock citation object matching Anthropic API structure.""" 
    def __init__(self, cited_text: str, document_title: str = "Test Document", start_page: int = 1):
        self.cited_text = cited_text
        self.document_title = document_title
        self.start_page_number = start_page
        self.end_page_number = start_page + 1
        self.document_index = 0
        self.type = "page_location"


def test_n_plus_one_block_pattern():
    """Test the common N + N+1 pattern: label block + value block."""
    
    # Mock N+1 pattern: label block + value block
    content_blocks = [
        MockBlock("- **Property name**: "),  # Block N (no citations)
        MockBlock("Test Building", [  # Block N+1 (with citations)
            MockCitation("MOCK DOCUMENT\r\nTest Building\r\n456 Mock Ave")
        ])
    ]
    
    job = MockJob(enable_citations=True)
    full_text, citation_blocks = _parse_content(content_blocks, job)
    
    # Verify the context was combined correctly
    assert len(citation_blocks) == 1
    block_text, citation = citation_blocks[0]
    assert "**Property name**:" in block_text
    assert "Test Building" in block_text
    
    # Test with mock property data
    property_data = MockPropertyData(property_name="Test Building")
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, property_data)
    
    # Should successfully map property_name with combined context
    assert "property_name" in field_mappings, "Should map property_name with N+N+1 context"
    assert warning is None, "Should have no warning with successful mapping"
    assert len(field_mappings["property_name"]) > 0, "Should have at least one citation"
    
    # Check confidence level
    mapping = field_mappings["property_name"][0]
    assert mapping.confidence in ["high", "medium", "low"], "Should have valid confidence level"


def test_four_block_pattern():
    """Test the 4-block pattern: label + value1 + connector + value2."""
    
    # Mock 4-block pattern: label + value1 + connector + value2
    content_blocks = [
        MockBlock("- **Land area**: "),  # Block N (no citations)
        MockBlock("0.25-acre parcel", [  # Block N+1 (with citations)
            MockCitation("Mock property contains office building on 0.25-acre parcel of land.")
        ]),
        MockBlock(" and "),  # Block N+2 (no citations) 
        MockBlock("10,890 square feet", [  # Block N+3 (with citations)
            MockCitation("Site area: 10,890 square feet (0.25 acres)")
        ])
    ]
    
    job = MockJob(enable_citations=True)
    full_text, citation_blocks = _parse_content(content_blocks, job)
    
    # Should have 2 citation blocks (one for each block with citations)
    assert len(citation_blocks) == 2
    
    # Second citation should include continuation context
    second_block_text, second_citation = citation_blocks[1]
    assert "**Land area**" in second_block_text or "and " in second_block_text
    assert "10,890" in second_citation.text
    
    # Test with mock property data
    property_data = MockPropertyData(
        property_name="Test Property",
        land_area_square_feet=10890.0
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, property_data)
    
    # Should successfully map land_area_square_feet
    assert "land_area_square_feet" in field_mappings, "Should map land_area_square_feet with 4-block pattern"
    assert len(field_mappings["land_area_square_feet"]) > 0, "Should have at least one citation"


def test_combined_block_context():
    """Test what happens when we manually combine block context."""
    
    # Simulate what the fix should produce with mock data
    manual_citation_blocks = [
        ("- **Property name**: Test Building", Citation(
            text="MOCK DOCUMENT\r\nTest Building\r\n456 Mock Ave",
            source="Test Document", 
            page=1
        )),
        ("- **Land area**: 0.25-acre parcel and 10,890 square feet", Citation(
            text="Site area: 10,890 square feet (0.25 acres)",
            source="Test Document",
            page=1
        ))
    ]
    
    property_data = MockPropertyData(
        property_name="Test Building",
        land_area_square_feet=10890.0
    )
    
    field_mappings, warning = map_citations_to_fields(manual_citation_blocks, property_data)
    
    # Both fields should be mapped with manually combined context
    assert "property_name" in field_mappings, "property_name should be mapped with context"
    assert "land_area_square_feet" in field_mappings, "land_area_square_feet should be mapped"
    assert warning is None, "Should have no warnings"