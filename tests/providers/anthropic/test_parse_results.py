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
        assert result.response == "This is the response text"
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
        assert result.response == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert "Request failed: errored" in result.error
    
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
        assert result.response == "Python is a programming language created by Guido van Rossum."
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
        assert result.response == response_text
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
        assert result.response == "Response text with potential citations."
        
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
        assert result.response == expected_text
        
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
        assert result.response == expected_text
        
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