"""
Invoice Citation Example

Demonstrates extracting structured data from invoice PDFs with citations enabled.
"""

from pydantic import BaseModel
from typing import Optional
from src import batch, Citation, BatchJob
from tests.utils.pdf_utils import create_pdf


class InvoiceData(BaseModel):
    """Structured model for invoice data extraction."""
    amount: str
    date: str
    product_title: str




def main():
    """Demonstrate invoice data extraction with citations."""
    
    print("Invoice Citation Example")
    print("=" * 60)
    
    # Create 3 different invoice PDFs
    invoice1 = create_pdf([
        """INVOICE

Invoice #: INV-001
Date: 2024-01-15
Bill To: Customer Name

From: TechCorp Solutions
123 Business Street
City, State 12345

ITEM DESCRIPTION                    AMOUNT
Cloud Computing Platform License    $2,450.00

TOTAL: $2,450.00

Payment Terms: Net 30
Thank you for your business!"""
    ])
    
    invoice2 = create_pdf([
        """INVOICE

Invoice #: INV-002
Date: 2024-02-20
Bill To: Customer Name

From: Design Studio LLC
123 Business Street
City, State 12345

ITEM DESCRIPTION                    AMOUNT
Website Design and Development      $3,200.00

TOTAL: $3,200.00

Payment Terms: Net 30
Thank you for your business!"""
    ])
    
    invoice3 = create_pdf([
        """INVOICE

Invoice #: INV-003
Date: 2024-03-10
Bill To: Customer Name

From: Equipment Rental Co
123 Business Street
City, State 12345

ITEM DESCRIPTION                    AMOUNT
Professional Camera Equipment Rental $850.00

TOTAL: $850.00

Payment Terms: Net 30
Thank you for your business!"""
    ])
    
    print("\nProcessing 3 invoice PDFs for data extraction...")
    print("This may take a few minutes due to batch processing...")
    
    try:
        # Process invoice PDFs with citations enabled and structured output
        job = batch(
            files=[invoice1, invoice2, invoice3],
            prompt="""
            Extract the following information from this invoice:
            1. Total amount (including currency symbol)
            2. Invoice date 
            3. Product or service title/description
            
            Return the data in the exact format requested.
            """,
            model="claude-3-5-sonnet-20241022",
            response_model=InvoiceData,
            enable_citations=True,
            verbose=True
        )
        
        print(f"\nBatch started. Batch ID: {job._batch_id}")
        print("Waiting for completion...")
        
        # Wait for completion and get results
        while not job.is_complete():
            import time
            time.sleep(5)
            job.stats(print_stats=True)
        
        results = job.results()
        print(f"\nProcessing complete! Got {len(results)} results.")
        
        # Display structured results
        company_names = ["TechCorp Solutions", "Design Studio LLC", "Equipment Rental Co"]
        
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"INVOICE {i+1}: {company_names[i]}")
            print('='*60)
            
            if isinstance(result, InvoiceData):
                print(f"💰 Amount: {result.amount}")
                print(f"📅 Date: {result.date}")
                print(f"📦 Product: {result.product_title}")
                
                print(f"\n📚 This response was extracted WITHOUT citation data")
                print("   (Pydantic models don't automatically include citations)")
                
            elif isinstance(result, CitedText):
                print(f"📄 Raw Response: {result.text}")
                print(f"\n📚 Citations ({len(result.citations)}):")
                for j, citation in enumerate(result.citations, 1):
                    print(f"\n  [{j}] \"{citation.cited_text[:100]}...\"")
                    print(f"      Type: {citation.type}")
                    if citation.start_page_number:
                        print(f"      Pages: {citation.start_page_number}-{citation.end_page_number}")
            else:
                print(f"📄 Response: {result}")
        
        print(f"\n{'='*60}")
        print("💡 NOTE: When using response_model with Pydantic, you get structured")
        print("   data but lose citation information. For citations, use raw text")
        print("   responses (no response_model) with enable_citations=True.")
        print('='*60)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()