"""
End-to-end integration tests for BatchManager.

These tests require a real API key and make actual calls to Anthropic's API.
They test the full BatchManager functionality including state persistence,
cost tracking, results saving, and complex nested Pydantic models with citations.
"""

import json
import os
import tempfile
from typing import List
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from batchata.batch_manager import BatchManager
from tests.utils.pdf_utils import create_pdf


class Invoice(BaseModel):
    """Simple invoice structure"""
    invoice_number: str
    company_name: str
    total_amount: float


def create_realistic_invoice_pdfs(num_invoices: int = 3) -> List[bytes]:
    """Create realistic invoice PDFs with line items for testing."""
    import random
    
    invoice_pdfs = []
    for i in range(1, num_invoices + 1):
        invoice_number = f"INV-2024-{i:04d}"
        company_name = f"TestCorp {chr(65 + i)} Inc."
        
        # Create realistic line items
        num_items = random.randint(2, 4)
        line_items = []
        subtotal = 0
        
        for j in range(num_items):
            qty = random.randint(1, 5)
            price = random.uniform(50, 500)
            item_total = qty * price
            subtotal += item_total
            line_items.append(f"Item {j+1}: Professional Services - Qty: {qty} @ ${price:.2f} = ${item_total:.2f}")
        
        invoice_content = f"""INVOICE

{company_name}
123 Business Drive, Suite {100 + i}
Tech City, CA 9410{i}

Invoice Number: {invoice_number}
Date: 2024-01-{(i * 3) % 28 + 1:02d}

Bill To: Client Company {i}
456 Customer Avenue
New York, NY 10001

LINE ITEMS:
{chr(10).join(line_items)}

SUBTOTAL: ${subtotal:.2f}
TAX (8.5%): ${subtotal * 0.085:.2f}
TOTAL: ${subtotal * 1.085:.2f}

Payment Terms: Net 30 days
Thank you for your business!"""
        
        pdf_bytes = create_pdf([invoice_content])
        invoice_pdfs.append(pdf_bytes)
    
    return invoice_pdfs


def test_batch_manager_e2e_invoice_processing():
    """
    Comprehensive E2E test for BatchManager with real API calls.
    Tests: nested Pydantic models, citations, state persistence, results saving, cost tracking.
    """
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic invoice PDFs
        num_invoices = 6  # Enough to create multiple jobs
        invoice_pdfs = create_realistic_invoice_pdfs(num_invoices)
        
        # Save PDFs to temporary files
        invoice_files = []
        for i, pdf_bytes in enumerate(invoice_pdfs):
            invoice_path = os.path.join(temp_dir, f"invoice_{i+1:03d}.pdf")
            with open(invoice_path, 'wb') as f:
                f.write(pdf_bytes)
            invoice_files.append(invoice_path)
        
        # Set up BatchManager paths
        state_file = os.path.join(temp_dir, "batch_state.json")
        results_dir = os.path.join(temp_dir, "results")
        
        # Initialize BatchManager with settings that will create multiple jobs
        manager = BatchManager(
            files=invoice_files,
            prompt="Extract detailed invoice data.",
            model="claude-3-5-sonnet-20241022",
            response_model=Invoice,
            enable_citations=True,
            items_per_job=2,  # Create 3 jobs for 6 invoices
            max_parallel_jobs=2,  # Test parallel processing
            max_cost=20.0,  # High enough limit for testing
            max_wait_time=600,  # 10 minutes timeout
            state_path=state_file,
            results_dir=results_dir
        )
        
        # Verify initial state
        initial_stats = manager.stats
        assert initial_stats['total_items'] == num_invoices
        assert initial_stats['completed_items'] == 0
        assert initial_stats['failed_items'] == 0
        assert initial_stats['total_cost'] == 0.0
        
        # Verify state file was created
        assert os.path.exists(state_file)
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        assert state_data['model'] == "claude-3-5-sonnet-20241022"
        assert state_data['items_per_job'] == 2
        assert len(state_data['jobs']) == 3  # 6 invoices / 2 per job
        
        # Run processing
        print("🚀 Starting BatchManager E2E test processing...")
        summary = manager.run(print_progress=True)
        
        # Verify processing completed successfully
        assert summary['total_items'] == num_invoices
        assert summary['completed_items'] > 0, "At least some items should have completed"
        assert summary['total_cost'] > 0.0, "Should have incurred some cost"
        
        # Verify final stats
        final_stats = manager.stats
        print(f"📊 Final stats: {final_stats}")
        
        # Verify state persistence
        assert os.path.exists(state_file)
        with open(state_file, 'r') as f:
            final_state = json.load(f)
        assert final_state['total_cost'] > 0
        
        # Verify results were saved to disk
        if summary['completed_items'] > 0:
            assert os.path.exists(results_dir)
            processed_dir = os.path.join(results_dir, "processed")
            assert os.path.exists(processed_dir)
            
            # Check that result files were created
            result_files = [f for f in os.listdir(processed_dir) if f.endswith('_results.json')]
            assert len(result_files) > 0, "Should have created result files"
            
            # Verify structure of saved results
            sample_result_file = os.path.join(processed_dir, result_files[0])
            with open(sample_result_file, 'r') as f:
                saved_results = json.load(f)
            
            assert isinstance(saved_results, list)
            if len(saved_results) > 0:
                sample_result = saved_results[0]
                assert "result" in sample_result
                assert "citations" in sample_result
                
                # Verify the nested structure
                result = sample_result["result"]
                assert "invoice_number" in result
                assert "company_name" in result
                assert "total_amount" in result
                
                # Verify citations structure
                citations = sample_result["citations"]
                assert isinstance(citations, dict)
                # Should have citations for at least some fields
                assert len(citations) > 0
        
        # Test results loading functionality
        if summary['completed_items'] > 0:
            loaded_results = manager.results()
            assert isinstance(loaded_results, list)
            assert len(loaded_results) > 0  # Should have some results
            
            # Check structure of loaded results
            for i, result_entry in enumerate(loaded_results):
                assert "result" in result_entry
                assert "citations" in result_entry
                
                # Verify the nested Invoice structure
                result = result_entry["result"]
                assert hasattr(result, "invoice_number")
                assert hasattr(result, "company_name")
                assert hasattr(result, "total_amount")


