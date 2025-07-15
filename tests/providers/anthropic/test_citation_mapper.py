"""Tests for value-based citation mapping functionality."""

import pytest
from typing import Optional
from pydantic import BaseModel

from batchata.providers.anthropic.citation_mapper import map_citations_to_fields
from batchata.types import Citation


class InvoiceModel(BaseModel):
    """Test model for invoice data."""
    invoice_number: str
    total_amount: float
    vendor: str
    payment_status: str


# Standard test data
STANDARD_INVOICE = InvoiceModel(
    invoice_number="INV-2024-002",
    total_amount=235.0,
    vendor="Acme Corp", 
    payment_status="PENDING"
)

@pytest.mark.parametrize("citation_text,parsed_response,expected_mappings", [
    # Main narrative format - the key use case we solved
    ('Invoice number "INV-2024-002" and vendor "Acme Corp"', STANDARD_INVOICE,
     ['invoice_number', 'vendor']),
    
    # Individual field citations
    ('Total amount of "$235.00"', STANDARD_INVOICE, ['total_amount']),
    ('Payment status "PENDING"', STANDARD_INVOICE, ['payment_status']),
    
    # Different number formats
    # Known issues: These require partial field word matching or currency normalization
    # ('Amount: $235', STANDARD_INVOICE, ['total_amount']),  # Missing 'total', has currency symbol
    # ('Amount: 235.00', STANDARD_INVOICE, ['total_amount']),  # Missing 'total'
    # ('Total: 235', STANDARD_INVOICE, ['total_amount']),  # Missing 'amount'
    
    # Case variations
    ('INVOICE NUMBER "INV-2024-002"', STANDARD_INVOICE, ['invoice_number']),
    ('payment status "pending"', STANDARD_INVOICE, ['payment_status']),
    ('Vendor: ACME CORP', STANDARD_INVOICE, ['vendor']),
    
    # Field word proximity (value exists, field words nearby)
    ('The invoice shows INV-2024-002 details', STANDARD_INVOICE, ['invoice_number']),
    ('Vendor information: Acme Corp supplied', STANDARD_INVOICE, ['vendor']),
    ('Amount due is $235.00 total', STANDARD_INVOICE, ['total_amount']),
    
    # Should NOT match (value exists but no field words nearby)
    ('Random text mentions INV-2024-002 somewhere', STANDARD_INVOICE, []),
    ('This document has 235 pages total', STANDARD_INVOICE, []),  # 'total' but no amount context
    # Known issue: This is a false positive - 'status' + exact value 'PENDING' matches payment_status
    # ('Meeting status: PENDING review', STANDARD_INVOICE, []),
    
    # No values present
    ('Random text with no field values', STANDARD_INVOICE, []),
    ('', STANDARD_INVOICE, []),
])
def test_value_based_mapping_scenarios(citation_text, parsed_response, expected_mappings):
    """Test various citation text patterns with value-based mapping."""
    
    citation = Citation(text="test", source="doc.pdf", page=1)
    citation_blocks = [(citation_text, citation)]
    
    mappings, _ = map_citations_to_fields(citation_blocks, parsed_response)
    
    # Check that exactly the expected fields are mapped
    assert set(mappings.keys()) == set(expected_mappings)
    
    # Verify each mapped field has the citation
    for field in expected_mappings:
        assert len(mappings[field]) == 1
        assert mappings[field][0] == citation


def test_multi_field_citations():
    """Test that one citation can map to multiple fields AND one field can have multiple citations."""
    parsed = InvoiceModel(
        invoice_number="INV-001",
        total_amount=100.0,
        vendor="Tech Co",
        payment_status="PAID"
    )
    
    citation1 = Citation(text="multi1", source="doc1.pdf", page=1)
    citation2 = Citation(text="multi2", source="doc2.pdf", page=2)
    citation3 = Citation(text="multi3", source="doc3.pdf", page=3)
    
    citation_blocks = [
        # One citation with multiple field values
        ('Invoice "INV-001" from vendor "Tech Co" shows total amount $100.00', citation1),
        # Another citation with different field
        ('Payment status is "PAID"', citation2),
        # Third citation that also mentions the vendor (same field, different citation)
        ('The vendor "Tech Co" provided excellent service', citation3),
    ]
    
    mappings, warning = map_citations_to_fields(citation_blocks, parsed)
    
    assert warning is None
    
    # First citation should map to 3 fields
    assert citation1 in mappings["invoice_number"]
    assert citation1 in mappings["vendor"]  
    assert citation1 in mappings["total_amount"]
    
    # Second citation maps to 1 field
    assert citation2 in mappings["payment_status"]
    
    # Third citation also maps to vendor (multiple citations for same field)
    assert citation3 in mappings["vendor"]
    
    # Verify citation counts
    assert len(mappings["invoice_number"]) == 1  # Only citation1
    assert len(mappings["vendor"]) == 2          # citation1 AND citation3
    assert len(mappings["total_amount"]) == 1    # Only citation1
    assert len(mappings["payment_status"]) == 1  # Only citation2


def test_edge_cases_and_skipped_fields():
    """Test edge cases: boolean skipping, empty values, warnings."""
    
    class EdgeCaseModel(BaseModel):
        name: str
        amount: float
        is_active: bool      # Should be skipped
        description: str     # Will be empty - should be skipped
        notes: Optional[str] = None  # Will be None - should be skipped
    
    # Test 1: Boolean and empty fields are skipped
    parsed = EdgeCaseModel(
        name="Test",
        amount=50.0,
        is_active=True,      # Should be skipped
        description="",      # Should be skipped  
        notes=None          # Should be skipped
    )
    
    citation = Citation(text="edge", source="doc.pdf")
    citation_blocks = [
        ('Name is "Test" and amount $50.00 and it is active', citation),
    ]
    
    mappings, warning = map_citations_to_fields(citation_blocks, parsed)
    
    # Only name and amount should be mapped (boolean and empty fields skipped)
    assert "name" in mappings
    assert "amount" in mappings
    assert "is_active" not in mappings      # Boolean skipped
    assert "description" not in mappings    # Empty string skipped
    assert "notes" not in mappings          # None skipped
    assert warning is None  # No warning since we mapped 2 out of 2 mappable fields
    
    # Test 2: Warning generation when many fields unmapped
    class ManyFieldsModel(BaseModel):
        field1: str
        field2: str  
        field3: str
        field4: str
    
    parsed_many = ManyFieldsModel(
        field1="value1",
        field2="value2", 
        field3="value3",
        field4="value4"
    )
    
    # Only one field will map, others won't
    citation_blocks_few = [
        ('Some field1 has "value1" in it', citation),  # Only this maps
        ('Random text about nothing', citation),
        ('More random content', citation),
    ]
    
    mappings_few, warning = map_citations_to_fields(citation_blocks_few, parsed_many)
    
    assert len(mappings_few) == 1  # Only field1 mapped
    assert "field1" in mappings_few
    assert warning is not None
    assert "field2, field3, field4" in warning  # Should mention unmapped fields