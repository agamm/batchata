"""Test the new systematic citation mapping algorithm."""

from typing import List
from pydantic import BaseModel, Field
from datetime import date

from batchata.providers.anthropic.citation_mapper import (
    map_citations_to_fields,
    _calculate_field_match_score,
    _fuzzy_word_match,
    _find_citations_with_value,
)
from batchata.types import Citation


class SampleModel(BaseModel):
    """Test model with various field types."""
    name: str = Field(..., description="Name field")
    amount: float = Field(..., description="Amount field")
    count: int = Field(..., description="Count field")
    tax_amount: float = Field(..., description="Tax amount")
    fiscal_year: int = Field(..., description="Fiscal year")
    building_count: int = Field(..., description="Number of buildings")
    story_count: int = Field(..., description="Number of stories")
    evaluation_date: date = Field(..., description="Date of evaluation")


def test_high_confidence_exact_field_match():
    """Test high confidence mapping with exact field pattern match."""
    # Create citation blocks with exact field patterns
    citation_blocks = [
        ("- **Name**: John Doe", Citation(text="Document mentions John Doe", source="doc", page=1)),
        ("- **Amount**: $1,500.00", Citation(text="Total amount is $1,500.00", source="doc", page=2)),
    ]
    
    test_data = SampleModel(
        name="John Doe",
        amount=1500.0,
        count=5,
        tax_amount=100.0,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map name and amount with high confidence
    assert "name" in field_mappings
    assert field_mappings["name"][0].confidence == "high"
    assert "exact field match" in field_mappings["name"][0].match_reason
    
    assert "amount" in field_mappings
    assert field_mappings["amount"][0].confidence == "high"


def test_medium_confidence_partial_field_match():
    """Test medium confidence mapping with partial field word matches."""
    # Create citations with partial field matches
    citation_blocks = [
        ("Tax is $250.50", Citation(text="Annual tax payment $250.50", source="doc", page=1)),
        ("Year 2024 report", Citation(text="Fiscal report for 2024", source="doc", page=2)),
    ]
    
    test_data = SampleModel(
        name="Test",
        amount=1000.0,
        count=5,
        tax_amount=250.50,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map tax_amount and fiscal_year with medium confidence
    assert "tax_amount" in field_mappings
    assert field_mappings["tax_amount"][0].confidence in ["medium", "low"]
    
    assert "fiscal_year" in field_mappings
    assert field_mappings["fiscal_year"][0].confidence in ["medium", "low"]


def test_low_confidence_value_only_match():
    """Test low confidence mapping when only value matches."""
    # Create citations with values but no field context
    citation_blocks = [
        ("Random text with 42 in it", Citation(text="Some document with 42", source="doc", page=1)),
    ]
    
    test_data = SampleModel(
        name="Test",
        amount=1000.0,
        count=42,
        tax_amount=100.0,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map count with low confidence (value only)
    assert "count" in field_mappings
    assert field_mappings["count"][0].confidence == "low"
    assert "value-only" in field_mappings["count"][0].match_reason


def test_fuzzy_word_matching():
    """Test fuzzy word matching for field names."""
    # Test plural/singular transformations
    assert _fuzzy_word_match("buildings everywhere", "building") == True
    assert _fuzzy_word_match("stories tall", "story") == True  # Fixed: story should match stories
    assert _fuzzy_word_match("property taxes", "tax") == True
    assert _fuzzy_word_match("assessed value", "assess") == True  # Fixed: assess should match assessed
    
    # Test that it doesn't match unrelated words
    assert _fuzzy_word_match("random text", "building") == False


def test_field_match_scoring():
    """Test field match score calculation."""
    # Test exact pattern match
    score = _calculate_field_match_score("- **tax_amount**: $500", "tax_amount")
    assert score == 1.0, "Should have perfect score for exact field pattern"
    
    # Test partial word match
    score = _calculate_field_match_score("The tax is $500", "tax_amount")
    assert 0.3 <= score < 1.0, "Should have partial score for partial word match"
    
    # Test no match - should get 0.0 when no field words match
    score = _calculate_field_match_score("Random text", "tax_amount")
    assert score == 0.0, "Should have zero score when no field words match"


def test_value_variants_with_commas():
    """Test that numeric values with commas are properly matched."""
    citation_blocks = [
        ("Real Estate Taxes $125,000", Citation(text="Taxes are $125,000", source="doc", page=1)),
    ]
    
    test_data = SampleModel(
        name="Test",
        amount=1000.0,
        count=5,
        tax_amount=125000.0,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map tax_amount even with comma formatting
    assert "tax_amount" in field_mappings
    assert len(field_mappings["tax_amount"]) > 0


def test_date_value_mapping():
    """Test that date values are properly matched."""
    citation_blocks = [
        ("**Evaluation date**: January 15, 2024", 
         Citation(text="Evaluated on January 15, 2024", source="doc", page=1)),
    ]
    
    test_data = SampleModel(
        name="Test",
        amount=1000.0,
        count=5,
        tax_amount=100.0,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map evaluation_date
    assert "evaluation_date" in field_mappings
    assert field_mappings["evaluation_date"][0].confidence == "high"


def test_systematic_fallback():
    """Test that algorithm uses systematic fallback (exact → partial → value-only)."""
    citation_blocks = [
        # High confidence match
        ("**Name**: Alice", Citation(text="Name is Alice", source="doc", page=1)),
        # Medium confidence match (partial field)
        ("Tax of 200", Citation(text="Tax payment 200", source="doc", page=2)),
        # Low confidence match (value only)
        ("Random 99", Citation(text="Number 99 appears", source="doc", page=3)),
    ]
    
    test_data = SampleModel(
        name="Alice",
        amount=1000.0,
        count=99,
        tax_amount=200.0,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Check confidence levels follow systematic fallback
    assert field_mappings["name"][0].confidence == "high"
    assert field_mappings["tax_amount"][0].confidence in ["medium", "low"]
    assert field_mappings["count"][0].confidence == "low"


def test_find_citations_with_value():
    """Test finding all citations containing a value."""
    citation_blocks = [
        ("First mention of 42", Citation(text="Contains 42", source="doc", page=1)),
        ("Second mention of 42", Citation(text="Also has 42", source="doc", page=2)),
        ("No match here", Citation(text="Different content", source="doc", page=3)),
    ]
    
    matches = _find_citations_with_value(citation_blocks, 42)
    
    assert len(matches) == 2, "Should find 2 citations with value 42"
    assert "First mention" in matches[0][0]
    assert "Second mention" in matches[1][0]


def test_non_markdown_field_patterns():
    """Test that non-markdown field patterns are detected."""
    # Test plain colon pattern
    score = _calculate_field_match_score("Tax amount: $500", "tax_amount")
    assert score >= 0.9, "Should have high score for plain colon pattern"
    
    # Test dash separator pattern
    score = _calculate_field_match_score("Tax amount - $500", "tax_amount")
    assert score >= 0.9, "Should have high score for dash separator"
    
    # Test 'is/are' pattern
    score = _calculate_field_match_score("The tax amount is $500", "tax_amount")
    assert score >= 0.9, "Should have high score for 'is' pattern"
    
    # Test that random text still gets low score
    score = _calculate_field_match_score("Random text with 500", "tax_amount")
    assert score == 0.0, "Should have zero score for unstructured text with no field words"


def test_non_markdown_citation_mapping():
    """Test citation mapping with non-markdown field patterns."""
    citation_blocks = [
        ("Property name: Test Building", 
         Citation(text="The property name is Test Building", source="doc", page=1)),
        ("Tax amount - $350.75", 
         Citation(text="Tax amount - $350.75", source="doc", page=2)),
        ("The fiscal year is 2024", 
         Citation(text="Fiscal year is 2024", source="doc", page=3)),
    ]
    
    test_data = SampleModel(
        name="Test Building",
        amount=1000.0,
        count=5,
        tax_amount=350.75,
        fiscal_year=2024,
        building_count=1,
        story_count=3,
        evaluation_date=date(2024, 1, 15)
    )
    
    field_mappings, warning = map_citations_to_fields(citation_blocks, test_data)
    
    # Should map fields even without markdown formatting
    assert "name" in field_mappings, "Should map name without markdown"
    assert field_mappings["name"][0].confidence == "high", "Should have high confidence"
    
    assert "tax_amount" in field_mappings, "Should map tax_amount without markdown"
    assert field_mappings["tax_amount"][0].confidence == "high", "Should have high confidence"
    
    assert "fiscal_year" in field_mappings, "Should map fiscal_year without markdown"
    assert field_mappings["fiscal_year"][0].confidence in ["high", "medium"], "Should have good confidence"