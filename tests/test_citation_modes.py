"""
Test Citation Modes

Test the 4 output modes of the citation API:
1. Plain text (no response_model, no citations)
2. Structured only (response_model, no citations) 
3. Text + Citations (no response_model, citations)
4. Structured + Field Citations (response_model + citations)
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from batchata import batch, Citation
from tests.utils import create_pdf


class InvoiceTestData(BaseModel):
    """Test model for structured output."""
    name: str
    value: str


class TestCitationModes:
    
    def test_mode_1_plain_text(self):
        """Test Mode 1: Plain text output without citations."""
        test_pdf = create_pdf(["Test document content"])
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = False
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [{"result": "Test response", "citations": None}]
            
            job = batch(
                files=[test_pdf],
                prompt="Summarize this document",
                model="claude-3-haiku-20240307"
            )
            
            results = job.results()
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0]["result"], str)
            assert results[0]["result"] == "Test response"
            assert results[0]["citations"] is None
            assert not job._enable_citations
    
    def test_mode_2_structured_only(self):
        """Test Mode 2: Structured output without citations."""
        test_pdf = create_pdf(["Test document content"])
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = False
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [{"result": InvoiceTestData(name="test", value="123"), "citations": None}]
            
            job = batch(
                files=[test_pdf],
                prompt="Extract data",
                model="claude-3-haiku-20240307",
                response_model=InvoiceTestData
            )
            
            results = job.results()
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0]["result"], InvoiceTestData)
            assert results[0]["result"].name == "test"
            assert results[0]["result"].value == "123"
            assert results[0]["citations"] is None
            assert not job._enable_citations
    
    def test_mode_3_text_citations(self):
        """Test Mode 3: Text output with citations."""
        test_pdf = create_pdf(["Test document content"])
        
        mock_citation = Citation(
            type="page_location",
            cited_text="Test document content",
            document_index=0,
            start_page_number=1,
            end_page_number=1
        )
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = True
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [{"result": "Test response", "citations": [mock_citation]}]
            
            job = batch(
                files=[test_pdf],
                prompt="Analyze document",
                model="claude-3-haiku-20240307",
                enable_citations=True
            )
            
            results = job.results()
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0]["result"], str)
            assert results[0]["result"] == "Test response"
            citations = results[0]["citations"]
            assert isinstance(citations, list)
            assert len(citations) == 1
            assert isinstance(citations[0], Citation)
            assert citations[0].type == "page_location"
            assert citations[0].cited_text == "Test document content"
            assert job._enable_citations
    
    def test_mode_4_structured_field_citations(self):
        """Test Mode 4: Structured output with field-level citations."""
        test_pdf = create_pdf(["Test document content"])
        
        mock_citation = Citation(
            type="page_location",
            cited_text="Test document content",
            document_index=0,
            start_page_number=1,
            end_page_number=1
        )
        
        field_citations = {"name": [mock_citation], "value": [mock_citation]}
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = True
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [{"result": InvoiceTestData(name="test", value="123"), "citations": field_citations}]
            
            job = batch(
                files=[test_pdf],
                prompt="Extract data with citations",
                model="claude-3-haiku-20240307",
                response_model=InvoiceTestData,
                enable_citations=True
            )
            
            results = job.results()
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0]["result"], InvoiceTestData)
            assert results[0]["result"].name == "test"
            assert results[0]["result"].value == "123"
            citations = results[0]["citations"]
            assert isinstance(citations, dict)
            assert "name" in citations
            assert "value" in citations
            assert isinstance(citations["name"], list)
            assert isinstance(citations["name"][0], Citation)
            assert job._enable_citations
    
    def test_empty_batch(self):
        """Test empty batch returns empty results."""
        job = batch(
            messages=[],
            model="claude-3-haiku-20240307"
        )
        
        assert job._batch_id == "empty_batch"
        assert job.is_complete()
        # Empty batch should have no results
        assert job.results() == []
    
    def test_batch_job_stats(self):
        """Test BatchJob stats method."""
        test_pdf = create_pdf(["Test content"])
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = True
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_batch_status.return_value = "ended"
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [{"result": "Test", "citations": []}]
            
            job = batch(
                files=[test_pdf],
                prompt="Test",
                model="claude-3-haiku-20240307",
                enable_citations=True
            )
            
            stats = job.stats()
            
            assert stats["batch_id"] == "batch_123"
            assert stats["is_complete"] == True
            assert stats["citations_enabled"] == True
            assert stats["has_response_model"] == False
            assert stats["total_results"] == 1
            assert stats["total_citations"] == 0
    
    def test_pdf_creation_utility(self):
        """Test PDF creation utility."""
        pages = ["Page 1 content", "Page 2 content"]
        pdf_bytes = create_pdf(pages)
        
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-1.4")
    
    
    def test_nested_model_with_citations_raises_error(self):
        """Test that nested Pydantic models with citations raise an error."""
        # Create nested model structure
        class Address(BaseModel):
            street: str
            city: str
            
        class PersonWithAddress(BaseModel):
            name: str
            age: int
            address: Address  # Nested model
        
        test_pdf = create_pdf(["Test document with person information"])
        
        # Test should raise ValueError when trying to use nested model with citations
        with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models.*contains nested model"):
            batch(
                files=[test_pdf],
                prompt="Extract person information",
                model="claude-3-5-sonnet-20241022",
                response_model=PersonWithAddress,
                enable_citations=True
            )
    
    def test_flat_model_with_citations_works(self):
        """Test that flat Pydantic models work fine with citations."""
        # Create flat model structure
        class PersonFlat(BaseModel):
            name: str
            age: int
            street: str
            city: str
        
        test_pdf = create_pdf(["John Doe, 30 years old, lives at 123 Main St, New York"])
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = True
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            
            # Mock field citations
            field_citations = {
                "name": [Citation(type="page_location", cited_text="John Doe", document_index=0)],
                "age": [Citation(type="page_location", cited_text="30 years old", document_index=0)],
                "street": [Citation(type="page_location", cited_text="123 Main St", document_index=0)],
                "city": [Citation(type="page_location", cited_text="New York", document_index=0)]
            }
            mock_provider.parse_results.return_value = [
                {"result": PersonFlat(name="John Doe", age=30, street="123 Main St", city="New York"), "citations": field_citations}
            ]
            
            # This should work without raising an error
            job = batch(
                files=[test_pdf],
                prompt="Extract person information",
                model="claude-3-5-sonnet-20241022",
                response_model=PersonFlat,
                enable_citations=True
            )
            
            results = job.results()
            
            assert len(results) == 1
            assert isinstance(results[0]["result"], PersonFlat)
            citations = results[0]["citations"]
            assert citations is not None
            assert isinstance(citations, dict)
    
    def test_complex_field_types_with_citations(self):
        """Test that models with Optional and Union types work with citations."""
        from typing import Optional, Union, List
        
        class ComplexModel(BaseModel):
            name: str
            age: Optional[int]
            score: Union[float, int]
            tags: List[str]
        
        test_pdf = create_pdf(["Test document with complex data"])
        
        with patch('batchata.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0'}]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = True
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            
            # Mock results
            mock_provider.parse_results.return_value = [
                {"result": ComplexModel(name="Test", age=25, score=95.5, tags=["tag1", "tag2"]), 
                 "citations": {"name": [Citation(type="page_location", cited_text="Test", document_index=0)]}}
            ]
            
            # This should work - Optional, Union, and List are allowed
            job = batch(
                files=[test_pdf],
                prompt="Extract data",
                model="claude-3-5-sonnet-20241022",
                response_model=ComplexModel,
                enable_citations=True
            )
            
            results = job.results()
            assert len(results) == 1
            assert isinstance(results[0]["result"], ComplexModel)