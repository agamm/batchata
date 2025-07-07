"""
PDF Data Extraction Example

Demonstrates using ai_batch to extract structured data from multiple PDFs.
"""

from pydantic import BaseModel
from src import batch_files, pdf_to_document_block, batch


class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor_name: str


def create_sample_pdf(invoice_num: str, vendor: str, amount: float) -> bytes:
    """Create a minimal valid PDF with invoice data for demo purposes."""
    # Create invoice content text
    content = f"INVOICE {invoice_num} - {vendor} - ${amount:.2f} - Date: 2024-01-15"
    
    # Create a minimal but valid PDF structure
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
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length {len(content) + 60} >>
stream
BT
/F1 12 Tf
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
0000000327 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
{400 + len(content)}
%%EOF"""
    return pdf_content.encode('latin-1')


def main():
    """Extract data from multiple PDF invoices."""
    
    print("Extracting data from PDF invoices...")
    print("-" * 50)
    
    # Create sample PDFs (in real use, these would be actual PDF files)
    pdfs_data = [
        create_sample_pdf("INV-001", "Acme Corp", 1250.00),
        create_sample_pdf("INV-002", "Tech Supplies Ltd", 3499.99),
        create_sample_pdf("INV-003", "Office Depot", 245.50),
    ]
    
    try:
        # Method 1: Using batch_files helper with PDF bytes
        results = batch_files(
            files=pdfs_data,
            prompt="Extract the invoice data from this PDF",
            model="claude-3-5-sonnet-20241022",
            response_model=InvoiceData,
            verbose=True
        )
        
        print("\nMethod 1 - Using batch_files helper with PDF bytes:")
        for i, invoice in enumerate(results):
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