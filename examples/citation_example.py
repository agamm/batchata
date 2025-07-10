"""
Citation Example

Demonstrates how to use citation support with PDF processing.
"""

from src import batch, Citation, BatchJob
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
            import time
            time.sleep(5)
            job.stats(print_stats=True)
        
        results = job.results()
        print(f"\nProcessing complete! Got {len(results)} results.")
        
        # Display results with citations
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            
            if isinstance(result, CitedText):
                print(f"Summary: {result.text}")
                print(f"\nCitations ({len(result.citations)}):")
                for j, citation in enumerate(result.citations, 1):
                    print(f"\n  [{j}] {citation.cited_text}")
                    print(f"      From: {citation.document_title}")
                    print(f"      Type: {citation.type}")
                    if citation.start_page_number:
                        print(f"      Pages: {citation.start_page_number}-{citation.end_page_number}")
                    elif citation.start_char_index:
                        print(f"      Characters: {citation.start_char_index}-{citation.end_char_index}")
            else:
                print(f"Text: {result}")
                print("No citations found")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()