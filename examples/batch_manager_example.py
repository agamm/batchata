"""
BatchManager Example: Invoice Processing

This example demonstrates how to use BatchManager to process a large number
of invoices with structured output, citations, parallel processing, and
automatic state persistence.
"""

import os
from typing import List, Optional
from pydantic import BaseModel

from batchata import BatchManager
from batchata import load_results_from_disk
from tests.utils.pdf_utils import create_pdf


class Invoice(BaseModel):
    """Simple invoice structure matching the generated PDF content"""
    invoice_number: str
    company_name: str
    total_amount: float


def create_realistic_invoice_files(num_files: int = 100, output_dir: str = "invoices") -> List[str]:
    """Create simple invoice PDF files with random amounts."""
    import random
    os.makedirs(output_dir, exist_ok=True)
    
    invoice_files = []
    for i in range(1, num_files + 1):
        invoice_number = f"INV-{2024}{i:04d}"
        total_amount = random.uniform(100, 5000)
        company_hex = f"{random.randint(0x1000, 0xFFFF):04X}"
        
        invoice_content = f"""INVOICE

Invoice Number: {invoice_number}
Company: Acme Corporation #{company_hex}
Date: 2024-01-{(i % 28) + 1:02d}

TOTAL: ${total_amount:.2f}

Payment Terms: Net 30 days"""
        
        pdf_bytes = create_pdf([invoice_content])
        invoice_path = os.path.join(output_dir, f"invoice_{i:03d}.pdf")
        with open(invoice_path, 'wb') as f:
            f.write(pdf_bytes)
        
        invoice_files.append(invoice_path)
    
    return invoice_files


def main():
    """Main example function"""
    print("🧾 BatchManager Invoice Processing Example")
    print("=" * 50)
    
    # Setup paths - clean start each time
    base_dir = "./examples/batch_example_dir"
    
    # Remove existing directory if it exists (fresh start)
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir, exist_ok=True)
    
    invoice_dir = os.path.join(base_dir, "invoices")
    results_dir = os.path.join(base_dir, "results")
    state_file = os.path.join(base_dir, "invoice_processing.json")
        
    # Create realistic invoice PDF files
    print("📄 Creating realistic invoice PDF files...")
    num_invoices = 5  # Small number for quick example
    invoice_files = create_realistic_invoice_files(num_invoices, invoice_dir)
    print(f"Created {len(invoice_files)} realistic invoice PDF files")
        
    # Initialize BatchManager
    print("\nInitializing BatchManager...")
    manager = BatchManager(
        files=invoice_files,
        prompt="Extract invoice data from this document",
        model="claude-3-5-sonnet-20241022",
        response_model=Invoice,
        enable_citations=True,
        items_per_job=5,  # Process all 5 invoices in one job
        max_parallel_jobs=1,  # Only need 1 job for 5 invoices
        max_cost=2.0,  # Stop if cost exceeds $2.00
        max_wait_time=600,
        state_path=state_file,
        results_dir=results_dir
    )
        
    print(f"Configured to process {num_invoices} invoices:")
    print(f"  - Items per job: 5 invoices per job")
    print(f"  - Max parallel jobs: 1")
    print(f"  - Cost limit: $2.00")
    print(f"  - State file: {state_file}")
    print(f"  - Results directory: {results_dir}")
    print(f"  - Invoice files location: {invoice_dir}")
        
    # Display initial statistics
    print(f"\nInitial Statistics:")
    stats = manager.stats
    print(f"  - Total items: {stats['total_items']}")
    
    # Run processing
    print(f"\n⚡ Starting batch processing...")
    try:
        # Note: This will process actual PDF files with real invoice content
        # You'll need to set up your Anthropic API key for this to work
        results = manager.run(print_progress=True)
        
        # Display final statistics
        final_stats = manager.stats
        print(f"\nFinal Statistics:")
        print(f"  - Total items: {final_stats['total_items']}")
        print(f"  - Completed items: {final_stats['completed_items']}")
        print(f"  - Failed items: {final_stats['failed_items']}")
        print(f"  - Total cost: ${final_stats['total_cost']:.2f}")
            
        # Show output directories
        if os.path.exists(results_dir):
            print(f"\nOutput directories:")
            print(f"  Results: {results_dir}")
            print(f"  Invoices: {invoice_dir}")
            print(f"  State: {state_file}")
            
            # Demonstrate getting results directly from manager
            print(f"\nGetting results directly from BatchManager...")
            try:
                manager_results = manager.results()
                print(f"Got {len(manager_results)} results from manager.results()")
                
                # Show first result as example
                if manager_results:
                    first_result = manager_results[0]
                    if first_result:
                        invoice = first_result["result"]
                        citations = first_result["citations"]
                        print(f"  First invoice: {invoice.invoice_number}")
                        print(f"      Company: {invoice.company_name}")
                        print(f"      Amount: ${invoice.total_amount:.2f}")
                        if citations:
                            print(f"      Citations: {len(citations)} field citations")
                    else:
                        print("  First result is None (processing may have failed)")
                        
            except Exception as e:
                print(f"❌ Error getting results from manager: {e}")
                
            # Also demonstrate loading results from disk
            print(f"\nAlternative: Loading results from disk...")
            try:
                loaded_results = load_results_from_disk(results_dir, Invoice)
                print(f"Loaded {len(loaded_results)} results from disk")
                
                # Show first result as example
                if loaded_results:
                    first_result = loaded_results[0]
                    invoice = first_result["result"]  # This is an Invoice instance
                    citations = first_result["citations"]  # This contains Citation objects
                    print(f"   Example result: {invoice.company_name} - ${invoice.total_amount}")
                    if citations:
                        print(f"   Citations: {len(citations)} field citations found")
            except Exception as e:
                print(f"❌ Error loading results: {e}")
        # Demonstrate retry functionality
        if final_stats['failed_items'] > 0:
            print(f"\nRetrying failed items...")
            retry_summary = manager.retry_failed()
            print(f"Retry completed: {retry_summary.get('retry_count', 0)} items processed")
            
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        print(f"This may be expected if you don't have an Anthropic API key configured.")
        
        # Show that state was saved even after failure
        if os.path.exists(state_file):
            print(f"\nState file was saved: {state_file}")
            import json
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            print(f"  Manager ID: {state_data['manager_id']}")
            print(f"  Jobs created: {len(state_data['jobs'])}")
            print(f"  Total cost: ${state_data['total_cost']:.2f}")
    
    print(f"\nFiles created in: {base_dir}")
    print(f"  To clean up, delete the directory: rm -rf {base_dir}")


def demo_resume_functionality():
    """
    Demonstrate resuming from a saved state file.
    This would be used when a batch job is interrupted.
    """
    print("\nResume Functionality Demo")
    print("-" * 30)
    
    state_file = "./examples/batch_example_dir/invoice_processing.json"
    if os.path.exists(state_file):
        print(f"Found existing state file: {state_file}")
        
        # Create BatchManager from existing state
        # The original files parameter is required for file-based batches
        try:
            manager = BatchManager(
                files=["dummy.pdf"],  # Required but will be overridden by state
                prompt="dummy",  # Required but will be overridden by state  
                model="claude-3-5-sonnet-20241022",
                state_path=state_file
            )
            
            print("Resuming from saved state...")
            summary = manager.run(print_progress=True)
            print(f"Resume completed: {summary.get('completed_items', 0)} total items completed")
        except Exception as e:
            print(f"Resume failed: {e}")
            print("This is expected if the original invoice files are not available.")
    else:
        print(f"No existing state file found at {state_file}")
        print("Run the main example first to create a state file, then interrupt it to test resume.")


if __name__ == "__main__":
    main()
    demo_resume_functionality()