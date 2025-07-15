"""Tests for Anthropic result parsing.

Testing:
1. Successful result parsing with content blocks
2. Error result handling
3. Citation extraction from responses
4. JSON model parsing
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from typing import Optional

from batchata.providers.anthropic.parse_results import parse_results
from batchata.core.job import Job


class TestParseResults:
    """Test parsing Anthropic API results."""
    
    def test_successful_text_result_parsing(self):
        """Test parsing successful text responses."""
        # Mock Anthropic result object
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [
            MagicMock(type="text", text="This is the response text")
        ]
        mock_result.result.message.usage = MagicMock(
            input_tokens=150,
            output_tokens=50
        )
        mock_result.custom_id = "test-job-1"
        
        job = Job(
            id="test-job-1",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            enable_citations=False
        )
        
        job_mapping = {"test-job-1": job}
        results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "test-job-1"
        assert result.raw_response == "This is the response text"
        assert result.input_tokens == 150
        assert result.output_tokens == 50
        assert result.error is None
    
    def test_error_result_handling(self):
        """Test parsing error responses."""
        # Mock error result
        mock_result = MagicMock()
        mock_result.result.type = "errored"
        mock_result.result.error = MagicMock(
            type="invalid_request",
            message="Invalid model parameter"
        )
        mock_result.custom_id = "error-job"
        
        job = Job(
            id="error-job",
            model="invalid-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        job_mapping = {"error-job": job}
        results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "error-job"
        assert result.raw_response == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert "Request failed: Invalid model parameter" in result.error
    
    def test_citation_extraction(self):
        """Test extracting citations from Anthropic content blocks."""
        # Mock content block with citations
        mock_citation1 = MagicMock()
        mock_citation1.cited_text = "Python is a high-level programming language"
        mock_citation1.document_title = "Python Documentation"
        mock_citation1.type = "direct_quote"
        mock_citation1.document_index = 0
        mock_citation1.start_page_number = 1
        mock_citation1.end_page_number = 1
        
        mock_citation2 = MagicMock()
        mock_citation2.cited_text = "created by Guido van Rossum"
        mock_citation2.document_title = "Python History"
        mock_citation2.type = "paraphrase"
        mock_citation2.document_index = 1
        mock_citation2.start_page_number = 5
        mock_citation2.end_page_number = 5
        
        # Mock content block with text and citations
        mock_content_block = MagicMock()
        mock_content_block.text = "Python is a programming language created by Guido van Rossum."
        mock_content_block.citations = [mock_citation1, mock_citation2]
        
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [mock_content_block]
        mock_result.result.message.usage = MagicMock(
            input_tokens=100,
            output_tokens=80
        )
        mock_result.custom_id = "citation-job"
        
        job = Job(
            id="citation-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Tell me about Python"}],
            enable_citations=True
        )
        
        job_mapping = {"citation-job": job}
        
        # Mock the cost calculation to avoid tokencost dependency
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.05):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "citation-job"
        assert result.raw_response == "Python is a programming language created by Guido van Rossum."
        assert result.input_tokens == 100
        assert result.output_tokens == 80
        assert result.cost_usd == 0.05
        
        # Check citations were extracted
        assert result.citations is not None
        assert len(result.citations) == 2
        
        # Check first citation
        citation1 = result.citations[0]
        assert citation1.text == "Python is a high-level programming language"
        assert citation1.source == "Python Documentation"
        assert citation1.metadata['type'] == "direct_quote"
        assert citation1.metadata['document_index'] == 0
        assert citation1.metadata['start_page_number'] == 1
        
        # Check second citation
        citation2 = result.citations[1]
        assert citation2.text == "created by Guido van Rossum"
        assert citation2.source == "Python History"
        assert citation2.metadata['type'] == "paraphrase"
        assert citation2.metadata['document_index'] == 1
        assert citation2.metadata['start_page_number'] == 5
    
    def test_citation_field_mapping(self):
        """Test that citations are mapped to fields when using response_model."""
        # Define test model
        class InvoiceInfo(BaseModel):
            invoice_number: str
            total_amount: float
            vendor: str
            payment_status: str
        
        # Mock citation
        mock_citation = MagicMock()
        mock_citation.cited_text = "INVOICE #INV-2024-002\r\nVendor: Acme Corp\r\nTotal: $235.00\r\nPayment Status: PAID"
        mock_citation.document_title = "invoice_002.pdf"
        mock_citation.type = "page_location"
        mock_citation.document_index = 0
        mock_citation.start_page_number = 1
        mock_citation.end_page_number = 2
        
        # Create content blocks matching API response pattern
        blocks = []
        
        # Block with JSON
        block1 = MagicMock()
        block1.text = 'Based on the invoice:\n\n```json\n{\n  "invoice_number": "INV-2024-002",\n  "total_amount": 235.00,\n  "vendor": "Acme Corp",\n  "payment_status": "PAID"\n}\n```\n\nThe information was extracted from:\n- '
        block1.citations = None
        blocks.append(block1)
        
        # Blocks with citations
        block2 = MagicMock()
        block2.text = "Invoice number: INV-2024-002"
        block2.citations = [mock_citation]
        blocks.append(block2)
        
        block3 = MagicMock()
        block3.text = "\n- "
        block3.citations = None
        blocks.append(block3)
        
        block4 = MagicMock()
        block4.text = "Vendor: Acme Corp"
        block4.citations = [mock_citation]
        blocks.append(block4)
        
        block5 = MagicMock()
        block5.text = "\n- "
        block5.citations = None
        blocks.append(block5)
        
        block6 = MagicMock()
        block6.text = "Total amount: $235.00"
        block6.citations = [mock_citation]
        blocks.append(block6)
        
        block7 = MagicMock()
        block7.text = "\n- "
        block7.citations = None
        blocks.append(block7)
        
        block8 = MagicMock()
        block8.text = "Payment status: PAID"
        block8.citations = [mock_citation]
        blocks.append(block8)
        
        # Create mock result
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = blocks
        mock_result.result.message.usage = MagicMock(
            input_tokens=2397,
            output_tokens=194
        )
        mock_result.custom_id = "job-mapped"
        
        job = Job(
            id="job-mapped",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Extract invoice"}],
            response_model=InvoiceInfo,
            enable_citations=True
        )
        
        job_mapping = {"job-mapped": job}
        
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.005):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        
        # Check parsed response
        assert result.parsed_response is not None
        assert isinstance(result.parsed_response, InvoiceInfo)
        assert result.parsed_response.invoice_number == "INV-2024-002"
        assert result.parsed_response.total_amount == 235.0
        assert result.parsed_response.vendor == "Acme Corp"
        assert result.parsed_response.payment_status == "PAID"
        
        # Check citations list (always present)
        assert result.citations is not None
        assert len(result.citations) == 4  # One for each field
        
        # Check citation mappings
        assert result.citation_mappings is not None
        assert isinstance(result.citation_mappings, dict)
        
        # Verify each field has citations mapped
        assert "invoice_number" in result.citation_mappings
        assert "vendor" in result.citation_mappings
        assert "total_amount" in result.citation_mappings
        assert "payment_status" in result.citation_mappings
        
        # Check citation content
        for field, citations in result.citation_mappings.items():
            assert len(citations) == 1
            assert citations[0].source == "invoice_002.pdf"
            assert citations[0].page == 1  # Should use start_page_number
    
    def test_citations_without_response_model(self):
        """Test that citations remain as list when no response_model is used."""
        # Mock citation
        mock_citation = MagicMock()
        mock_citation.cited_text = "Important information from document"
        mock_citation.document_title = "source.pdf"
        mock_citation.type = "direct_quote"
        mock_citation.document_index = 0
        mock_citation.start_page_number = 5
        mock_citation.end_page_number = 5
        
        # Create content blocks
        block1 = MagicMock()
        block1.text = "Here is the cited information: "
        block1.citations = None
        
        block2 = MagicMock()
        block2.text = "Important information"
        block2.citations = [mock_citation]
        
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [block1, block2]
        mock_result.result.message.usage = MagicMock(
            input_tokens=100,
            output_tokens=50
        )
        mock_result.custom_id = "job-no-model"
        
        # Job without response_model
        job = Job(
            id="job-no-model",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Get info"}],
            response_model=None,  # No response model
            enable_citations=True
        )
        
        job_mapping = {"job-no-model": job}
        
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.002):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        
        # Check that citations are present as list
        assert result.citations is not None
        assert isinstance(result.citations, list)
        assert len(result.citations) == 1
        assert result.citations[0].source == "source.pdf"
        assert result.citations[0].page == 5
        
        # Check that citation_mappings is None without response_model
        assert result.citation_mappings is None
    
    def test_json_model_parsing(self):
        """Test parsing structured JSON responses into Pydantic models."""
        # Define a test Pydantic model
        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: Optional[str] = None
        
        # Response text containing JSON
        response_text = '''Here is the information you requested:
        
        {
            "name": "Guido van Rossum",
            "age": 67,
            "occupation": "Software Engineer"
        }
        
        This person created Python programming language.'''
        
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [
            MagicMock(text=response_text)
        ]
        mock_result.result.message.usage = MagicMock(
            input_tokens=50,
            output_tokens=120
        )
        mock_result.custom_id = "json-job"
        
        job = Job(
            id="json-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Get person info"}],
            response_model=PersonInfo
        )
        
        job_mapping = {"json-job": job}
        
        # Mock the cost calculation to avoid tokencost dependency
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.03):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "json-job"
        assert result.raw_response == response_text
        assert result.input_tokens == 50
        assert result.output_tokens == 120
        assert result.cost_usd == 0.03
        
        # Check that JSON was parsed into the Pydantic model
        assert result.parsed_response is not None
        assert isinstance(result.parsed_response, PersonInfo)
        assert result.parsed_response.name == "Guido van Rossum"
        assert result.parsed_response.age == 67
        assert result.parsed_response.occupation == "Software Engineer"
    
    def test_citations_disabled(self):
        """Test that citations are not extracted when enable_citations=False."""
        # Mock content block with citations but citations disabled
        mock_citation = MagicMock()
        mock_citation.cited_text = "Some citation text"
        mock_citation.document_title = "Document"
        
        mock_content_block = MagicMock()
        mock_content_block.text = "Response text with potential citations."
        mock_content_block.citations = [mock_citation]
        
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [mock_content_block]
        mock_result.result.message.usage = MagicMock(
            input_tokens=30,
            output_tokens=40
        )
        mock_result.custom_id = "no-citations-job"
        
        job = Job(
            id="no-citations-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Test"}],
            enable_citations=False  # Citations disabled
        )
        
        job_mapping = {"no-citations-job": job}
        
        # Mock the cost calculation to avoid tokencost dependency
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.02):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "no-citations-job"
        assert result.raw_response == "Response text with potential citations."
        
        # Citations should not be extracted when disabled
        assert result.citations is None
    
    def test_multiple_content_blocks_with_citations(self):
        """Test parsing multiple content blocks with text and citations."""
        # Mock citations for different blocks
        mock_citation1 = MagicMock()
        mock_citation1.cited_text = "Python was created in 1991"
        mock_citation1.document_title = "Python History"
        mock_citation1.type = "direct_quote"
        mock_citation1.document_index = 0
        mock_citation1.start_page_number = 1
        mock_citation1.end_page_number = 1
        
        mock_citation2 = MagicMock()
        mock_citation2.cited_text = "Python emphasizes code readability"
        mock_citation2.document_title = "Python Philosophy"
        mock_citation2.type = "paraphrase"
        mock_citation2.document_index = 1
        mock_citation2.start_page_number = 5
        mock_citation2.end_page_number = 5
        
        mock_citation3 = MagicMock()
        mock_citation3.cited_text = "Python is widely used in data science"
        mock_citation3.document_title = "Python Applications"
        mock_citation3.type = "summary"
        mock_citation3.document_index = 2
        mock_citation3.start_page_number = 10
        mock_citation3.end_page_number = 12
        
        # Create multiple content blocks
        mock_block1 = MagicMock()
        mock_block1.text = "Python is a programming language "
        mock_block1.citations = [mock_citation1]
        
        mock_block2 = MagicMock()
        mock_block2.text = "that was designed for readability. "
        mock_block2.citations = [mock_citation2]
        
        mock_block3 = MagicMock()
        mock_block3.text = "It's popular in many fields including data science."
        mock_block3.citations = [mock_citation3]
        
        # Mock the result with multiple content blocks
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [mock_block1, mock_block2, mock_block3]
        mock_result.result.message.usage = MagicMock(
            input_tokens=80,
            output_tokens=60
        )
        mock_result.custom_id = "multi-block-job"
        
        job = Job(
            id="multi-block-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Tell me about Python with citations"}],
            enable_citations=True
        )
        
        job_mapping = {"multi-block-job": job}
        
        # Mock the cost calculation to avoid tokencost dependency
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.04):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "multi-block-job"
        
        # Check that all text blocks were concatenated
        expected_text = "Python is a programming language that was designed for readability. It's popular in many fields including data science."
        assert result.raw_response == expected_text
        
        assert result.input_tokens == 80
        assert result.output_tokens == 60
        assert result.cost_usd == 0.04
        
        # Check that all citations from all blocks were collected
        assert result.citations is not None
        assert len(result.citations) == 3
        
        # Check first citation (from block 1)
        citation1 = result.citations[0]
        assert citation1.text == "Python was created in 1991"
        assert citation1.source == "Python History"
        assert citation1.metadata['type'] == "direct_quote"
        assert citation1.metadata['document_index'] == 0
        assert citation1.metadata['start_page_number'] == 1
        
        # Check second citation (from block 2)
        citation2 = result.citations[1]
        assert citation2.text == "Python emphasizes code readability"
        assert citation2.source == "Python Philosophy"
        assert citation2.metadata['type'] == "paraphrase"
        assert citation2.metadata['document_index'] == 1
        assert citation2.metadata['start_page_number'] == 5
        
        # Check third citation (from block 3)
        citation3 = result.citations[2]
        assert citation3.text == "Python is widely used in data science"
        assert citation3.source == "Python Applications"
        assert citation3.metadata['type'] == "summary"
        assert citation3.metadata['document_index'] == 2
        assert citation3.metadata['start_page_number'] == 10
        assert citation3.metadata['end_page_number'] == 12
    
    def test_json_with_multiple_blocks_and_citations(self):
        """Test JSON parsing with multiple content blocks containing citations."""
        # Define a test Pydantic model
        class LanguageInfo(BaseModel):
            name: str
            year_created: int
            creator: str
            main_features: list[str]
        
        # Mock citations
        mock_citation1 = MagicMock()
        mock_citation1.cited_text = "Guido van Rossum created Python"
        mock_citation1.document_title = "Python Creator Biography"
        mock_citation1.type = "fact"
        mock_citation1.document_index = 0
        mock_citation1.start_page_number = 1
        mock_citation1.end_page_number = 1
        
        mock_citation2 = MagicMock()
        mock_citation2.cited_text = "Python first appeared in 1991"
        mock_citation2.document_title = "Programming Language Timeline"
        mock_citation2.type = "historical_fact"
        mock_citation2.document_index = 1
        mock_citation2.start_page_number = 15
        mock_citation2.end_page_number = 15
        
        # Create content blocks - some with JSON, some with citations
        mock_block1 = MagicMock()
        mock_block1.text = "Based on the research, here's the information: "
        mock_block1.citations = [mock_citation1]
        
        mock_block2 = MagicMock()
        mock_block2.text = '{"name": "Python", "year_created": 1991, "creator": "Guido van Rossum", "main_features": ["readable", "interpreted", "object-oriented"]} '
        mock_block2.citations = []
        
        mock_block3 = MagicMock()
        mock_block3.text = "This data was compiled from historical records."
        mock_block3.citations = [mock_citation2]
        
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [mock_block1, mock_block2, mock_block3]
        mock_result.result.message.usage = MagicMock(
            input_tokens=120,
            output_tokens=90
        )
        mock_result.custom_id = "json-multi-block-job"
        
        job = Job(
            id="json-multi-block-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Get language info as JSON with citations"}],
            response_model=LanguageInfo,
            enable_citations=True
        )
        
        job_mapping = {"json-multi-block-job": job}
        
        # Mock the cost calculation to avoid tokencost dependency
        with patch('batchata.providers.anthropic.parse_results._calculate_cost', return_value=0.06):
            results = parse_results([mock_result], job_mapping)
        
        assert len(results) == 1
        result = results[0]
        assert result.job_id == "json-multi-block-job"
        
        # Check that all text blocks were concatenated
        expected_text = 'Based on the research, here\'s the information: {"name": "Python", "year_created": 1991, "creator": "Guido van Rossum", "main_features": ["readable", "interpreted", "object-oriented"]} This data was compiled from historical records.'
        assert result.raw_response == expected_text
        
        # Check that JSON was extracted and parsed despite being in the middle of text
        assert result.parsed_response is not None
        assert isinstance(result.parsed_response, LanguageInfo)
        assert result.parsed_response.name == "Python"
        assert result.parsed_response.year_created == 1991
        assert result.parsed_response.creator == "Guido van Rossum"
        assert len(result.parsed_response.main_features) == 3
        assert "readable" in result.parsed_response.main_features
        assert "interpreted" in result.parsed_response.main_features
        assert "object-oriented" in result.parsed_response.main_features
        
        # Check that citations from multiple blocks were collected
        assert result.citations is not None
        assert len(result.citations) == 2
        
        # Check citations
        citation1 = result.citations[0]
        assert citation1.text == "Guido van Rossum created Python"
        assert citation1.source == "Python Creator Biography"
        assert citation1.metadata['type'] == "fact"
        
        citation2 = result.citations[1]
        assert citation2.text == "Python first appeared in 1991"
        assert citation2.source == "Programming Language Timeline"
        assert citation2.metadata['type'] == "historical_fact"
        assert citation2.metadata['start_page_number'] == 15