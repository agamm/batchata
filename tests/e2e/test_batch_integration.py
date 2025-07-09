"""
End-to-end integration tests for ai_batch.

These tests require a real API key and make actual calls to Anthropic's API.
They test the happy path scenarios with real data.
"""

from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.core import batch
from src import batch_files, Citation
from tests.utils import create_pdf
import time


class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float



def test_spam_detection_happy_path():
    """Test spam detection with real API - happy path only."""
    emails = [
        "You've won $1,000,000! Click here now!",  # Obviously spam
        "Meeting tomorrow at 3pm to discuss Q3 results"  # Obviously not spam
    ]
    
    messages = [[{"role": "user", "content": f"You are a spam detection expert. Is this spam? {email}"}] for email in emails]
    
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    # Wait for completion
    import time
    while not job.is_complete():
        time.sleep(2)
    
    results = job.results()
    
    assert len(results) == 2
    assert all(isinstance(result, SpamResult) for result in results)
    
    # Type assertions for proper type checking
    first_result = results[0]
    second_result = results[1]
    assert isinstance(first_result, SpamResult)
    assert isinstance(second_result, SpamResult)
    
    # Verify first email is detected as spam
    assert first_result.is_spam == True
    assert first_result.confidence > 0.0
    assert len(first_result.reason) > 0
    
    # Verify second email is not spam
    assert second_result.is_spam == False
    assert second_result.confidence > 0.0  # Confidence represents how sure the model is, not spam probability
    assert len(second_result.reason) > 0


class InvoiceData(BaseModel):
    """Invoice data extraction model."""
    company_name: str
    amount: str
    date: str

def test_structured_field_citations_e2e():
    """Test structured output with field-level citations - e2e with real API."""
    # Create a test invoice PDF (same as working debug script)
    test_document = create_pdf([
        """INVOICE

TechCorp Solutions Inc.
123 Technology Drive
San Francisco, CA 94105

Invoice #: INV-2024-001
Date: March 15, 2024""","""

Bill To: Client Company
456 Customer Ave
New York, NY 10001

DESCRIPTION                           AMOUNT
Professional Services                 $12,500.00
""","""
TOTAL: $12,500.00

Payment Terms: Net 30
Due Date: April 15, 2024

Thank you for your business!"""
    ])
    
    job = batch_files(
        files=[test_document],
        prompt="Extract the company name, total amount, and invoice date. Use citations to reference where you found each piece of information.",
        model="claude-3-5-sonnet-20241022",
        response_model=InvoiceData,
        enable_citations=True,
        verbose=True
    )
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    citations = job.citations()
    
    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 1
    
    result = results[0]
    assert isinstance(result, InvoiceData)
    assert len(result.company_name) > 0
    assert len(result.amount) > 0
    assert len(result.date) > 0
    
    # Verify citations structure for field-level citations
    assert isinstance(citations, list)
    
    # Skip test if API didn't return citations (known Batch API limitation)
    if len(citations) == 0 or (len(citations) == 1 and isinstance(citations[0], dict) and len(citations[0]) == 0):
        import pytest
        pytest.skip("Anthropic Batch API doesn't return citation metadata - known limitation")
    
    assert len(citations) == 1  # One FieldCitations dict per result
    
    field_citations = citations[0]
    assert isinstance(field_citations, dict)
    
    # Check that we have field-level citations
    expected_fields = ["company_name", "amount", "date"]
    found_fields = list(field_citations.keys())
    
    # At least some fields should have citations
    assert len(found_fields) > 0
    assert all(field in expected_fields for field in found_fields)
    
    # Verify citation structure for each field
    for field_cit_list in field_citations.values():
        assert isinstance(field_cit_list, list)
        assert len(field_cit_list) > 0
        
        for cit in field_cit_list:
            assert isinstance(cit, Citation)
            assert len(cit.cited_text) > 0
            assert cit.type in ["page_location", "document_location"]
            if cit.start_page_number is not None:
                assert cit.start_page_number >= 1
                if cit.end_page_number is not None:
                    assert cit.end_page_number >= cit.start_page_number

