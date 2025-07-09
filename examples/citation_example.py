"""
Citation Example

Demonstrates how to use citation support with PDF processing.
"""

from src import batch_files, CitedText, BatchJob


def create_research_paper_pdf(title: str, content: str) -> bytes:
    """Create a minimal PDF that looks like a research paper."""
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
<< /Length {len(title) + len(content) + 100} >>
stream
BT
/F1 16 Tf
72 720 Td
({title}) Tj
0 -40 Td
/F1 12 Tf
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
{500 + len(title) + len(content)}
%%EOF"""
    return pdf_content.encode('latin-1')


def main():
    """Demonstrate citation extraction from PDFs."""
    
    print("Citation Example")
    print("=" * 50)
    
    # Create sample research papers
    paper1 = create_research_paper_pdf(
        "Climate Change Impact Study",
        "Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times. This warming is primarily caused by human activities."
    )
    
    paper2 = create_research_paper_pdf(
        "Renewable Energy Solutions", 
        "Solar power efficiency has increased by 40% in the last decade. Wind energy now provides 10% of global electricity generation."
    )
    
    try:
        # Process PDFs with citations enabled
        job = batch_files(
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