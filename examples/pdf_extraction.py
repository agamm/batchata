"""
PDF Data Extraction Example

Demonstrates using bachata to extract structured data from multiple PDFs.
"""

import time
from dotenv import load_dotenv
from pydantic import BaseModel
from batchata import batch, pdf_to_document_block
from tests.utils.pdf_utils import create_pdf

# Load environment variables for examples
load_dotenv()


class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor_name: str




def main():
    """Extract data from multiple PDF invoices."""
    
    print("Extracting data from PDF invoices...")
    print("-" * 50)
    
    # Create sample PDFs (in real use, these would be actual PDF files)
    pdfs_data = [
        create_pdf([f"INVOICE INV-001 - Acme Corp - $1250.00 - Date: 2024-01-15"]),
        create_pdf([f"INVOICE INV-002 - Tech Supplies Ltd - $3499.99 - Date: 2024-01-15"]),
        create_pdf([f"INVOICE INV-003 - Office Depot - $245.50 - Date: 2024-01-15"]),
    ]
    
    try:
        # Method 1: Using batch() with PDF bytes
        job = batch(
            files=pdfs_data,
            prompt="Extract the invoice data from this PDF",
            model="claude-3-5-sonnet-20241022",
            response_model=InvoiceData,
            verbose=True
        )
        
        # Wait for completion
        while not job.is_complete():
            print(f"Batch job is running. Batch ID: {job._batch_id}...")
            time.sleep(30)  # Check every 30 seconds
        
        print("\nMethod 1 - Using batch() with PDF bytes:")
        batch_results = job.results()
        for i, result_entry in enumerate(batch_results):
            invoice = result_entry['result']  # Extract the InvoiceData from BatchResult
            print(f"\nPDF {i+1}:")
            print(f"  Invoice #: {invoice.invoice_number}")
            print(f"  Vendor: {invoice.vendor_name}")
            print(f"  Amount: ${invoice.total_amount:.2f}")
            print(f"  Date: {invoice.date}")
        
        print("\n" + "-" * 50)
        
        # Method 2: Using batch() directly with document blocks
        # messages = []
        # for pdf_data in pdfs_data:
        #     doc_block = pdf_to_document_block(pdf_data)
        #     messages.append([{
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "Extract invoice information from this PDF"},
        #             doc_block
        #         ]
        #     }])
        
        # results2 = batch(
        #     messages=messages,
        #     model="claude-3-5-sonnet-20241022",
        #     response_model=InvoiceData
        # )
        
        # print("\nMethod 2 - Using batch() with document blocks:")
        # for i, invoice in enumerate(results2):
        #     print(f"\nPDF {i+1}:")
        #     print(f"  Invoice #: {invoice.invoice_number}")
        #     print(f"  Vendor: {invoice.vendor_name}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set the ANTHROPIC_API_KEY environment variable.")


if __name__ == "__main__":
    main()