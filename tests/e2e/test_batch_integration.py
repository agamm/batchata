"""
End-to-end integration tests for bachata.

These tests require a real API key and make actual calls to Anthropic's API.
They test the happy path scenarios with real data.
"""

from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src import batch
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
    assert all(isinstance(result_entry["result"], SpamResult) for result_entry in results)
    
    # Type assertions for proper type checking
    first_result = results[0]["result"]
    second_result = results[1]["result"]
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
    
    job = batch(
        files=[test_document],
        prompt="Extract the company name, total amount, and invoice date.",
        model="claude-3-5-sonnet-20241022",
        response_model=InvoiceData,
        enable_citations=True,
        verbose=True
    )
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    
    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 1
    
    result_entry = results[0]
    assert isinstance(result_entry, dict)
    assert "result" in result_entry
    assert "citations" in result_entry
    
    # Verify the actual result model
    result = result_entry["result"]
    assert isinstance(result, InvoiceData)
    assert len(result.company_name) > 0
    assert len(result.amount) > 0
    assert len(result.date) > 0
    
    # Verify citations structure for field-level citations
    citations = result_entry["citations"]
    assert isinstance(citations, dict)
    
    # Citations should contain the relevant text, but API may return larger chunks
    assert "TechCorp Solutions Inc." in citations['company_name'][0].cited_text
    assert citations['company_name'][0].start_page_number is not None
    assert citations['company_name'][0].end_page_number is not None
    assert citations['company_name'][0].document_index == 0
    
    # Test amount field citation
    assert '$12,500.00' in citations['amount'][0].cited_text
    assert citations['amount'][0].document_index == 0
    
    # Test date field citation  
    assert 'date' in citations and len(citations['date']) > 0
    assert citations['date'][0].document_index == 0
    


