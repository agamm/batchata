"""Simple PDF demo with file and prompt for citations."""

import tempfile
import os
import random
from pathlib import Path
from batchata import Batch
from pydantic import BaseModel
from dotenv import load_dotenv
from batchata.utils.pdf import create_pdf

load_dotenv()


class InvoiceAnalysis(BaseModel):
    """Structured output for invoice analysis."""
    invoice_number: str
    total_amount: float
    vendor: str
    payment_status: str


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
            Batch(results_dir="./examples/pdf_output", max_parallel_batches=3, items_per_batch=2)
            .set_state(file="./examples/demo_pdf_state.json", reuse_previous=False)
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
                enable_citations=True
            )
        
        # Execute batch
        print("Starting batch processing...")
        run = batch.run(print_status=True, on_progress=lambda s, t, b: \
                        print(f"\rProgress: {s['completed']}/{s['total']} jobs | "\
                              f"Batches: {s['batches_completed']}/{s['batches_total']} (pending: {s['batches_pending']}) | " \
                              f"Cost: ${round(s['cost_usd'],3)}/{s['cost_limit_usd']} | " \
                              f"Items per batch: {s['items_per_batch']} | Time: {round(t, 2)}s", end=""))
        
        # Get results
        run.status(print_status=True)
        results = run.results()
        
        # Display results
        print("\nResults:")
        for job_id, result in results.items():
            if result.is_success:
                analysis = result.parsed_response
                print(f"\nJob {job_id}:")
                print(f"  Invoice: {analysis.invoice_number}")
                print(f"  Vendor: {analysis.vendor}")
                print(f"  Total: ${analysis.total_amount:.2f}")
                print(f"  Status: {analysis.payment_status}")
                
                # Show citations if available
                if result.citations:
                    print(f"  Citations found: {len(result.citations)}")
                    for i, citation in enumerate(result.citations):
                        print(f"    - Page {citation.page}: {citation.text[:50]}...")
                
                # Show citation mappings if available
                if result.citation_mappings:
                    print(f"  \nCitation mappings:")
                    for field, field_citations in result.citation_mappings.items():
                        print(f"    {field}:")
                        for citation in field_citations:
                            print(f"      - Page {citation.page}: {citation.text.strip()}")
            else:
                print(f"\nJob {job_id} failed: {result.error}")
    
    finally:
        # Clean up temporary files
        for file in invoice_files:
            file.unlink()
        os.rmdir(temp_dir)
        print(f"\nCleaned up temporary files in {temp_dir}")


if __name__ == "__main__":
    main()