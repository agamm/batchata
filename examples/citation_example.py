"""
Citation Example

Demonstrates how to use citation support with PDF processing.
"""

import time
from batchata import batch
from batchata.citations import Citation
from tests.utils.pdf_utils import create_pdf


def main():
    """Demonstrate citation extraction from PDFs."""
    
    print("Citation Example")
    print("=" * 50)
    
    # Create sample research papers
    paper1 = create_pdf([
        "Climate Change Impact Study\n\nGlobal temperatures have risen by 1.1 degrees Celsius since pre-industrial times. This warming is primarily caused by human activities."
    ])
    
    paper2 = create_pdf([
        "Renewable Energy Solutions\n\nSolar power efficiency has increased by 40% in the last decade. Wind energy now provides 10% of global electricity generation."
    ])
    
    try:
        # Process PDFs with citations enabled
        job = batch(
            files=[paper1, paper2],
            prompt="Summarize the key findings from these research papers, citing specific claims.",
            model="claude-3-5-sonnet-20241022",
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
        
        # Display results with citations
        for i, result_entry in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            
            result_text = result_entry['result']
            citations = result_entry['citations']
            
            print(f"Summary: {result_text}")
            
            if citations:
                print(f"\nCitations ({len(citations)}):")
                for j, citation in enumerate(citations, 1):
                    print(f"\n  [{j}] {citation.cited_text}")
                    if hasattr(citation, 'document_title'):
                        print(f"      From: {citation.document_title}")
                    if hasattr(citation, 'start_page'):
                        print(f"      Page: {citation.start_page}")
            else:
                print("No citations found")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()