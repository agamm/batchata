"""
Invoice Citation Example

Demonstrates extracting structured data from invoice PDFs with citations enabled.
"""

import time
from pydantic import BaseModel
from typing import Optional
from batchata import batch
from batchata.citations import Citation
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
            print(f"Batch job is running. Batch ID: {job._batch_id}...")
            time.sleep(30)  # Check every 30 seconds
        
        results = job.results()
        print(f"\nProcessing complete! Got {len(results)} results.")
        
        # Display structured results
        company_names = ["TechCorp Solutions", "Design Studio LLC", "Equipment Rental Co"]
        
        for i, result_entry in enumerate(results):
            print(f"\n{'='*60}")
            print(f"INVOICE {i+1}: {company_names[i]}")
            print('='*60)
            
            invoice_data = result_entry['result']
            citations = result_entry['citations']
            
            print(f"Amount: {invoice_data.amount}")
            print(f"Date: {invoice_data.date}")
            print(f"Product: {invoice_data.product_title}")
            
            if citations:
                print(f"\n📚 Field Citations:")
                for field_name, field_citations in citations.items():
                    print(f"  {field_name}: {len(field_citations)} citation(s)")
                    for j, citation in enumerate(field_citations, 1):
                        print(f"    [{j}] \"{citation.cited_text}\"")
                        if hasattr(citation, 'start_page'):
                            print(f"        Page: {citation.start_page}")
            else:
                print(f"\nNo citations found")
        
        print(f"\n{'='*60}")
        print("NOTE: With response_model + enable_citations=True, you get")
        print("   structured data AND field-level citations mapping!")
        print('='*60)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()