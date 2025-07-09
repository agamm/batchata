"""
Invoice Citation Example

Demonstrates extracting structured data from invoice PDFs with citations enabled.
"""

from pydantic import BaseModel
from typing import Optional
from src import batch_files, CitedText, BatchJob


class InvoiceData(BaseModel):
    """Structured model for invoice data extraction."""
    amount: str
    date: str
    product_title: str


def create_invoice_pdf(invoice_num: str, company: str, product: str, amount: str, date: str) -> bytes:
    """Create an invoice PDF."""
    content = f"""INVOICE

Invoice #: {invoice_num}
Date: {date}
Bill To: Customer Name

From: {company}
123 Business Street
City, State 12345

ITEM DESCRIPTION                    AMOUNT
{product}                          {amount}

TOTAL: {amount}

Payment Terms: Net 30
Thank you for your business!"""
    
    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> /F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length {len(content) + 100} >>
stream
BT
/F2 12 Tf
72 720 Td
({content}) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000229 00000 n 
0000000380 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
{500 + len(content)}
%%EOF"""
    return pdf_content.encode('latin-1')


def main():
    """Demonstrate invoice data extraction with citations."""
    
    print("Invoice Citation Example")
    print("=" * 60)
    
    # Create 3 different invoice PDFs
    invoice1 = create_invoice_pdf(
        "INV-001",
        "TechCorp Solutions",
        "Cloud Computing Platform License",
        "$2,450.00",
        "2024-01-15"
    )
    
    invoice2 = create_invoice_pdf(
        "INV-002", 
        "Design Studio LLC",
        "Website Design and Development",
        "$3,200.00",
        "2024-02-20"
    )
    
    invoice3 = create_invoice_pdf(
        "INV-003",
        "Equipment Rental Co",
        "Professional Camera Equipment Rental",
        "$850.00", 
        "2024-03-10"
    )
    
    print("\nProcessing 3 invoice PDFs for data extraction...")
    print("This may take a few minutes due to batch processing...")
    
    try:
        # Process invoice PDFs with citations enabled and structured output
        job = batch_files(
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