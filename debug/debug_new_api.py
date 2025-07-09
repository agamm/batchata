"""
Test New Simplified API

Test all 4 modes of the new API:
1. Plain text (no response_model, no citations)
2. Structured only (response_model, no citations)
3. Text + Citations (no response_model, citations)
4. Structured + Field Citations (response_model + citations)
"""

import sys
from pathlib import Path
# Add project root to path for debug scripts
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import batch_files, Citation
from tests.utils import create_pdf
from pydantic import BaseModel
import time


class InvoiceData(BaseModel):
    """Invoice data extraction model."""
    company_name: str
    amount: str
    date: str


def create_test_invoice() -> bytes:
    """Create a test invoice PDF."""
    pages = [
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
    ]
    
    return create_pdf(pages)


def test_mode_1_plain_text():
    """Mode 1: Plain text (no response_model, no citations)"""
    print("\n" + "="*60)
    print("MODE 1: Plain Text")
    print("="*60)
    
    invoice = create_test_invoice()
    
    job = batch_files(
        files=[invoice],
        prompt="Summarize this invoice in one sentence.",
        model="claude-3-5-sonnet-20241022",
        verbose=True
    )
    
    print(f"Job created: {job._batch_id}")
    print(f"Enable citations: {job._enable_citations}")
    print(f"Response model: {job._response_model}")
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    citations = job.citations()
    
    print(f"\nResults type: {type(results)}")
    print(f"Results: {results}")
    print(f"Citations: {citations}")
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert citations is None


def test_mode_2_structured_only():
    """Mode 2: Structured only (response_model, no citations)"""
    print("\n" + "="*60)
    print("MODE 2: Structured Only")
    print("="*60)
    
    invoice = create_test_invoice()
    
    job = batch_files(
        files=[invoice],
        prompt="Extract the company name, total amount, and invoice date.",
        model="claude-3-5-sonnet-20241022",
        response_model=InvoiceData,
        verbose=True
    )
    
    print(f"Job created: {job._batch_id}")
    print(f"Enable citations: {job._enable_citations}")
    print(f"Response model: {job._response_model}")
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    citations = job.citations()
    
    print(f"\nResults type: {type(results)}")
    print(f"Results: {results}")
    print(f"Citations: {citations}")
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], InvoiceData)
    assert citations is None


def test_mode_3_text_citations():
    """Mode 3: Text + Citations (no response_model, citations)"""
    print("\n" + "="*60)
    print("MODE 3: Text + Citations")
    print("="*60)
    
    invoice = create_test_invoice()
    
    job = batch_files(
        files=[invoice],
        prompt="What is the company name and total amount? Cite your sources.",
        model="claude-3-5-sonnet-20241022",
        enable_citations=True,
        verbose=True
    )
    
    print(f"Job created: {job._batch_id}")
    print(f"Enable citations: {job._enable_citations}")
    print(f"Response model: {job._response_model}")
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    citations = job.citations()
    
    print(f"\nResults type: {type(results)}")
    print(f"Results: {results}")
    print(f"\nCitations type: {type(citations)}")
    print(f"Number of citations: {len(citations) if citations else 0}")
    
    if citations:
        for i, cit in enumerate(citations):
            print(f"\nCitation {i+1}:")
            # Type assertion to help type checker
            assert isinstance(cit, Citation)
            print(f"  Type: {cit.type}")
            print(f"  Cited text: {cit.cited_text[:50]}...")
            if cit.start_page_number:
                print(f"  Pages: {cit.start_page_number}-{cit.end_page_number}")
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert isinstance(citations, list)
    assert all(isinstance(c, Citation) for c in citations)


def test_mode_4_structured_field_citations():
    """Mode 4: Structured + Field Citations (response_model + citations)"""
    print("\n" + "="*60)
    print("MODE 4: Structured + Field Citations")
    print("="*60)
    
    invoice = create_test_invoice()
    
    job = batch_files(
        files=[invoice],
        prompt="Extract the company name, total amount, and invoice date. Use citations to reference where you found each piece of information.",
        model="claude-3-5-sonnet-20241022",
        response_model=InvoiceData,
        enable_citations=True,
        verbose=True
    )
    
    print(f"Job created: {job._batch_id}")
    print(f"Enable citations: {job._enable_citations}")
    print(f"Response model: {job._response_model}")
    
    # Wait for completion
    while not job.is_complete():
        time.sleep(3)
    
    results = job.results()
    citations = job.citations()
    
    print(f"\nResults type: {type(results)}")
    print(f"Results: {results}")
    print(f"\nCitations type: {type(citations)}")
    
    if citations:
        print(f"Number of field citation dicts: {len(citations)}")
        for i, field_cit_dict in enumerate(citations):
            print(f"\n📄 Result {i+1} field citations:")
            # Type assertion to help type checker
            assert isinstance(field_cit_dict, dict)
            for field_name, field_citations in field_cit_dict.items():
                print(f"  🔍 {field_name}: {len(field_citations)} citation(s)")
                for j, cit in enumerate(field_citations):
                    # Type assertion to help type checker
                    assert isinstance(cit, Citation)
                    print(f"    [{j+1}] \"{cit.cited_text[:80]}{'...' if len(cit.cited_text) > 80 else ''}\"")
                    print(f"        Type: {cit.type}")
                    if cit.start_page_number:
                        print(f"        Pages: {cit.start_page_number}-{cit.end_page_number}")
                    if cit.document_title:
                        print(f"        Document: {cit.document_title}")
                    print()
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], InvoiceData)
    assert isinstance(citations, list)
    assert len(citations) == 1  # One FieldCitations dict per result
    assert isinstance(citations[0], dict)
    
    # Check field citations structure
    field_cits = citations[0]
    assert all(isinstance(field, str) for field in field_cits.keys())
    assert all(isinstance(cits, list) for cits in field_cits.values())
    assert all(isinstance(c, Citation) for cits in field_cits.values() for c in cits)


def main():
    """Run all tests."""
    print("🧪 Testing New Simplified API")
    
    try:
        test_mode_1_plain_text()
        print("\n✅ Mode 1: Plain Text - PASSED")
    except Exception as e:
        print(f"\n❌ Mode 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_mode_2_structured_only()
        print("\n✅ Mode 2: Structured Only - PASSED")
    except Exception as e:
        print(f"\n❌ Mode 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_mode_3_text_citations()
        print("\n✅ Mode 3: Text + Citations - PASSED")
    except Exception as e:
        print(f"\n❌ Mode 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_mode_4_structured_field_citations()
        print("\n✅ Mode 4: Structured + Field Citations - PASSED")
    except Exception as e:
        print(f"\n❌ Mode 4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()