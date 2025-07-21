"""Simple PDF demo with file and prompt for citations."""

import tempfile
import os
import random
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from batchata import Batch
from pydantic import BaseModel, Field
from batchata.utils.pdf import create_pdf



class InvoiceAnalysis(BaseModel):
    """Structured output for invoice analysis."""
    invoice_number: str = Field(description="The invoice number (e.g., INV-2024-001)")
    total_amount: float = Field(description="The total amount due on the invoice")
    vendor: str = Field(description="The name of the vendor/company issuing the invoice")
    payment_status: str = Field(description="Payment status: paid, pending, or overdue (always lowercase)")


def generate_invoice_pages(invoice_num: int) -> list[str]:
    """Generate invoice content split across multiple pages."""
    vendor = random.choice(["Acme Corp", "Tech Solutions", "Office Supplies"])
    total = random.randint(100, 1000)
    status = random.choice(["PAID", "PENDING", "OVERDUE"])
    
    # Split content across 3 pages for better citation testing
    page1 = f"""INVOICE #INV-2024-{invoice_num:03d}

Date: 2024-07-14
Vendor: {vendor}"""
    
    page2 = f"""Invoice Details

Total: ${total}.00
Tax: Included
Shipping: Free"""
    
    page3 = f"""Payment Information

Payment Status: {status}
Due Date: 2024-08-14
Terms: Net 30"""
    
    return [page1, page2, page3]


def create_temp_invoice_files(num_files: int = 3):
    """Create temporary invoice PDF files for testing.
    
    Args:
        num_files: Number of invoice files to generate (default: 3)
    """
    temp_dir = tempfile.mkdtemp(prefix="batchata_invoices_")
    
    files = []
    for i in range(1, num_files + 1):
        filepath = Path(temp_dir) / f"invoice_{i:03d}.pdf"
        pages = generate_invoice_pages(i)
        pdf_bytes = create_pdf(pages)
        filepath.write_bytes(pdf_bytes)
        files.append(filepath)
    
    return files, temp_dir


def main():
    """Run invoice processing demo with file and prompt."""
    # Create temporary invoice files
    invoice_files, temp_dir = create_temp_invoice_files()
    
    try:
        # Create batch configuration
        batch = (
            Batch(results_dir="./examples/pdf_output", max_parallel_batches=3, items_per_batch=2, raw_files=True)
            .set_state(file="./examples/demo_pdf_state.json", reuse_state=False)
            # .set_default_params(model="gpt-4o-mini-2024-07-18", temperature=0.7)
            .set_default_params(model="claude-sonnet-4-20250514", temperature=0.7)
            .add_cost_limit(usd=5.0)
            .set_verbosity("warn")
            
        )
        
        # Add jobs using file and prompt
        for invoice_file in invoice_files:
            batch.add_job(
                file=invoice_file,
                prompt="Extract the invoice number, total amount, vendor name, and payment status.",
                response_model=InvoiceAnalysis,
                # enable_citations=True
            )
        
        # Execute batch
        print("Starting batch processing...")
        run = batch.run(print_status=True)
        
        # Get results
        run.status(print_status=True)
        results = run.results()
        
        # Display results
        print("\nResults:")
        
        # Show successful results
        for result in results["completed"]:
            analysis = result.parsed_response
            print(f"\nJob {result.job_id}:")
            print(f"  Invoice: {analysis.invoice_number}")
            print(f"  Vendor: {analysis.vendor}")
            print(f"  Total: ${analysis.total_amount:.2f}")
            print(f"  Status: {analysis.payment_status}")
            
            print(f"  Tokens: {result.input_tokens} input, {result.output_tokens} output")
            print(f"  Cost: ${result.cost_usd:.6f}")
        
        # Show failed results
        for result in results["failed"]:
            print(f"\nJob {result.job_id} failed: {result.error}")
        
        # Show cancelled results
        for result in results["cancelled"]:
            print(f"\nJob {result.job_id} was cancelled: {result.error}")
    
    finally:
        # Clean up temporary files
        for file in invoice_files:
            file.unlink()
        os.rmdir(temp_dir)
        print(f"\nCleaned up temporary files in {temp_dir}")


if __name__ == "__main__":
    main()