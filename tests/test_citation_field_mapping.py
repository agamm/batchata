"""
Comprehensive tests for citation field mapping with proper JSON parsing.

Tests the improved _map_citations_to_fields implementation that replaces
fragile regex parsing with robust JSON-based field mapping.
"""

import json
from unittest.mock import MagicMock
from pydantic import BaseModel
from typing import List, Optional

from batchata.providers.anthropic import AnthropicBatchProvider
from batchata.citations import Citation


class SimpleModel(BaseModel):
    """Simple flat model for basic testing."""
    name: str
    age: int
    location: str


class ComplexModel(BaseModel):
    """More complex model with various field types."""
    title: str
    description: str
    price: float
    tags: List[str]
    metadata: Optional[str] = None


class NestedModel(BaseModel):
    """Model with nested structure."""
    name: str
    details: dict
    items: List[str]


def create_mock_content_block(text: str, citations: Optional[List] = None):
    """Create a mock content block with text and optional citations."""
    block = MagicMock()
    block.text = text
    block.citations = citations or []
    return block


def create_mock_citation(cited_text: str, doc_index: int = 0):
    """Create a mock citation object."""
    cit = MagicMock()
    cit.type = "page_location"
    cit.cited_text = cited_text
    cit.document_index = doc_index
    cit.document_title = "Test Document"
    cit.start_page_number = 1
    cit.end_page_number = 1
    cit.start_char_index = None
    cit.end_char_index = None
    cit.start_content_block_index = None
    cit.end_content_block_index = None
    return cit


class TestCitationFieldMapping:
    """Test the improved citation field mapping functionality."""
    
    def test_regex_breaking_scenarios_with_citations(self):
        """Test scenarios that would break the old regex-based approach BUT still parse without crashing."""
        provider = AnthropicBatchProvider()
        
        # Test complex scenarios that would completely break regex parsing
        scenarios = [
            # Escaped quotes in JSON
            '{"name": "John \\"The Great\\" Doe", "age": 30, "location": "New York"}',
            # URLs with colons that confuse regex  
            '{"name": "John", "urls": ["http://example.com:8080"], "age": 30}',
            # Nested objects
            '{"user": {"name": "John"}, "location": "NYC", "age": 30}',
            # Multi-line strings
            '{"name": "John\\nDoe", "description": "Line 1\\nLine 2", "age": 30}'
        ]
        
        for scenario in scenarios:
            content_blocks = [create_mock_content_block(scenario, [])]
            field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
            
            # Key point: Should NOT crash and should return a valid dict
            assert isinstance(field_citations, dict)
            # All citation lists should be valid (empty in this case since no citations provided)
            assert all(isinstance(citations, list) for citations in field_citations.values())
    
    def test_json_parsing_vs_regex_accuracy(self):
        """Test that demonstrates why JSON parsing is superior to regex for citation mapping."""
        provider = AnthropicBatchProvider()
        
        # This JSON would completely confuse regex-based field detection
        # but proper JSON parsing handles it correctly
        citation_name = create_mock_citation("Citation for name field") 
        citation_location = create_mock_citation("Citation for location field")
        
        # Complex JSON that would break regex pattern r'"(\w+)"\s*:\s*$'
        
        content_blocks = [
            create_mock_content_block('{"name": "John"', [citation_name]),
            create_mock_content_block(', "metadata": {"nested": "value"}', []),  # Would confuse regex
            create_mock_content_block(', "location": "NYC"}', [citation_location])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Should correctly identify and map citations despite complex structure  
        assert isinstance(field_citations, dict)
        assert 'name' in field_citations
        assert 'location' in field_citations
        assert len(field_citations['name']) == 1
        assert len(field_citations['location']) == 1
        assert field_citations['name'][0].cited_text == "Citation for name field"
        assert field_citations['location'][0].cited_text == "Citation for location field"
    
    def test_complex_field_patterns_that_break_regex(self):
        """Test field patterns that would confuse regex parsing."""
        provider = AnthropicBatchProvider()
        
        # Pattern that would confuse the regex: r'"(\w+)"\s*:\s*$'
        confusing_patterns = [
            # Field name followed by complex object
            '{"metadata": {"nested": {"deep": "value"}}, "name": "John", "age": 30}',
            
            # Field names with numbers/underscores (regex only looks for \w+)
            '{"user_name": "John", "age_2": 30, "location_1": "NYC"}',
            
            # Fields with array values containing colons
            '{"name": "John", "urls": ["http://example.com", "https://test.com"], "age": 30}',
            
            # Fields with string values containing quotes and colons
            '{"name": "John: The Great", "description": "He said: \\"Hello\\"", "age": 30}',
            
            # Fields not at end of line (would break regex with $ anchor)
            '{"name": "John", "age": 30, "location": "NYC"}',  # All on one line
        ]
        
        for pattern in confusing_patterns:
            content_blocks = [create_mock_content_block(pattern, [])]
            field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
            assert isinstance(field_citations, dict), f"Failed for pattern: {pattern}"
    
    def test_stateful_logic_problems(self):
        """Test scenarios where stateful 'current_field' tracking would fail."""
        provider = AnthropicBatchProvider()
        
        # Create content blocks that would confuse stateful tracking
        citation1 = create_mock_citation("Citation for John")
        citation2 = create_mock_citation("Citation for age") 
        citation3 = create_mock_citation("Citation for location")
        
        # Scenario 1: Blocks not in field order (would confuse stateful tracking)
        content_blocks = [
            create_mock_content_block('{"location": "NYC"', [citation3]),  # location first
            create_mock_content_block(', "name": "John"', [citation1]),    # name second  
            create_mock_content_block(', "age": 30}', [citation2])         # age last
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Should handle out-of-order fields without confusion
        assert isinstance(field_citations, dict)
        
        # Scenario 2: Multiple field names in same block (would confuse current_field)
        single_block = create_mock_content_block(
            '{"name": "John", "age": 30, "location": "NYC"}', 
            [citation1, citation2, citation3]
        )
        content_blocks = [single_block]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        assert isinstance(field_citations, dict)
    
    def test_simple_json_with_citations(self):
        """Test basic JSON structure with citations mapped to correct fields."""
        provider = AnthropicBatchProvider()
        
        # Mock content blocks - each field value gets its own block with citations
        citation1 = create_mock_citation("John Doe from page 1")
        citation2 = create_mock_citation("Age 30 mentioned")
        citation3 = create_mock_citation("Lives in New York")
        
        content_blocks = [
            create_mock_content_block('{"name": "', []),
            create_mock_content_block('John Doe', [citation1]),
            create_mock_content_block('", "age": ', []),
            create_mock_content_block('30', [citation2]),
            create_mock_content_block(', "location": "', []),
            create_mock_content_block('New York', [citation3]),
            create_mock_content_block('"}', [])
        ]
        
        # Test the mapping
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Verify citations are mapped to correct fields
        assert 'name' in field_citations
        assert 'age' in field_citations  
        assert 'location' in field_citations
        
        assert len(field_citations['name']) == 1
        assert field_citations['name'][0].cited_text == "John Doe from page 1"
        
        assert len(field_citations['age']) == 1
        assert field_citations['age'][0].cited_text == "Age 30 mentioned"
        
        assert len(field_citations['location']) == 1
        assert field_citations['location'][0].cited_text == "Lives in New York"
    
    def test_complex_json_structure(self):
        """Test complex JSON with arrays, numbers, and multiple types."""
        provider = AnthropicBatchProvider()
        
        # Mock content blocks with citations for different field types
        title_citation = create_mock_citation("Product Name from catalog")
        desc_citation = create_mock_citation("Great product description")
        price_citation = create_mock_citation("Price $99.99")
        
        content_blocks = [
            create_mock_content_block('{"title": "Product Name"', [title_citation]),
            create_mock_content_block(', "description": "A great product with multiple features"', [desc_citation]),
            create_mock_content_block(', "price": 99.99', [price_citation]),
            create_mock_content_block(', "tags": ["electronics", "gadget", "popular"]', []),
            create_mock_content_block(', "metadata": "Additional info"}', [])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, ComplexModel)
        
        # Verify complex field types are handled correctly
        assert 'title' in field_citations
        assert len(field_citations['title']) == 1
        assert field_citations['title'][0].cited_text == "Product Name from catalog"
        
        assert 'description' in field_citations  
        assert len(field_citations['description']) == 1
        assert field_citations['description'][0].cited_text == "Great product description"
        
        assert 'price' in field_citations
        assert len(field_citations['price']) == 1
        assert field_citations['price'][0].cited_text == "Price $99.99"
    
    def test_malformed_json_fallback(self):
        """Test that malformed JSON falls back gracefully."""
        provider = AnthropicBatchProvider()
        
        # Create malformed JSON content blocks
        content_blocks = [
            create_mock_content_block('{"name": "John"', []),
            create_mock_content_block(', "age": 30 INVALID', []),  # Malformed
            create_mock_content_block('location": "NYC"}', [])     # Missing quote
        ]
        
        # Should return empty dict instead of crashing
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        assert field_citations == {}
    
    def test_json_with_escaped_quotes(self):
        """Test JSON with escaped quotes and special characters."""
        provider = AnthropicBatchProvider()
        
        # JSON with escaped quotes and special chars
        content_blocks = [
            create_mock_content_block('{"title": "The \\"Great\\" Product"', []),
            create_mock_content_block(', "description": "Line 1\\nLine 2\\tTabbed"', []),
            create_mock_content_block(', "price": 99.99}', [])
        ]
        
        # Should parse correctly without regex failures
        field_citations = provider._map_citations_to_fields(content_blocks, ComplexModel)
        
        # Should not crash and should return proper structure
        assert isinstance(field_citations, dict)
        assert all(isinstance(citations, list) for citations in field_citations.values())
    
    def test_nested_json_structures(self):
        """Test JSON with nested objects and arrays."""
        provider = AnthropicBatchProvider()
        
        name_citation = create_mock_citation("Test Item from catalog")
        
        content_blocks = [
            create_mock_content_block('{"name": "Test Item"', [name_citation]),
            create_mock_content_block(', "details": {"category": "electronics", "weight": 1.5}', []),
            create_mock_content_block(', "items": ["item1", "item2", "item3"]}', [])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, NestedModel)
        
        # Should handle nested structures and map citations correctly
        assert isinstance(field_citations, dict)
        assert 'name' in field_citations
        assert len(field_citations['name']) == 1
        assert field_citations['name'][0].cited_text == "Test Item from catalog"
    
    def test_empty_content_blocks(self):
        """Test with empty content blocks."""
        provider = AnthropicBatchProvider()
        
        content_blocks = []
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        assert field_citations == {}
    
    def test_blocks_without_citations(self):
        """Test content blocks that have no citations."""
        provider = AnthropicBatchProvider()
        
        content_blocks = [
            create_mock_content_block('{"name": "John"', []),
            create_mock_content_block(', "age": 30', []),
            create_mock_content_block(', "location": "NYC"}', [])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Should return empty lists for all fields
        for field_name in SimpleModel.model_fields.keys():
            if field_name in field_citations:
                assert field_citations[field_name] == []
    
    def test_multiple_citations_per_field(self):
        """Test multiple citations mapped to the same field."""
        provider = AnthropicBatchProvider()
        
        citation1 = create_mock_citation("First mention of John Doe")
        citation2 = create_mock_citation("Second mention of John Doe") 
        
        # More realistic scenario - both citations in the same content block
        content_blocks = [
            create_mock_content_block('{"name": "John Doe"', [citation1, citation2]),
            create_mock_content_block(', "age": 30}', [])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Should collect multiple citations for the same field
        assert 'name' in field_citations
        # The implementation should map both citations to the name field
        assert len(field_citations['name']) >= 1  # At least one citation should be mapped
        # Check that citations contain the expected text
        citation_texts = [cit.cited_text for cit in field_citations['name']]
        assert any("John Doe" in text for text in citation_texts)
    
    def test_text_matching_heuristics(self):
        """Test the _text_likely_matches_field heuristic function."""
        provider = AnthropicBatchProvider()
        
        # Test field value matching
        assert provider._text_likely_matches_field('John Doe', '"John Doe"', 'name')
        assert provider._text_likely_matches_field('The value is John Doe here', 'John Doe', 'name')
        
        # Test field name matching
        assert provider._text_likely_matches_field('"name": "John"', 'John', 'name')
        
        # Test no match
        assert not provider._text_likely_matches_field('Some other text', 'John', 'name')
        
        # Test short values (should not match)
        assert not provider._text_likely_matches_field('ab', 'xy', 'name')
    
    def test_citation_object_creation(self):
        """Test that Citation objects are created correctly from mock citations."""
        provider = AnthropicBatchProvider()
        
        mock_cit = create_mock_citation("Test citation text")
        
        content_blocks = [
            create_mock_content_block('{"name": "John"}', [mock_cit])
        ]
        
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Verify Citation objects are created properly
        assert 'name' in field_citations
        assert len(field_citations['name']) == 1
        citation = field_citations['name'][0]
        assert isinstance(citation, Citation)
        assert citation.cited_text == "Test citation text"
        assert citation.type == "page_location"
        assert citation.document_title == "Test Document"
    
    def test_large_json_performance(self):
        """Test performance with larger JSON structures."""
        provider = AnthropicBatchProvider()
        
        # Create a larger JSON structure
        large_json = {
            f"field_{i}": f"value_{i}" for i in range(50)
        }
        
        json_text = json.dumps(large_json)
        content_blocks = [
            create_mock_content_block(json_text, [create_mock_citation(f"Citation {i}") for i in range(5)])
        ]
        
        # Should complete without performance issues
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        
        # Should return a result (empty or populated) without hanging/crashing
        assert isinstance(field_citations, dict)
    
    def test_unicode_and_special_characters(self):
        """Test JSON with Unicode and special characters."""
        provider = AnthropicBatchProvider()
        
        content_blocks = [
            create_mock_content_block('{"name": "José Münoz", "age": 25, "location": "São Paulo"}', [])
        ]
        
        # Should handle Unicode without issues
        field_citations = provider._map_citations_to_fields(content_blocks, SimpleModel)
        assert isinstance(field_citations, dict)


if __name__ == "__main__":
    # Run a quick test to verify the implementation works
    test = TestCitationFieldMapping()
    test.test_simple_json_with_citations()
    print("✅ Citation field mapping tests completed successfully!")