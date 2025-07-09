"""
PDF Utilities Module

Provides utility functions for creating test PDFs.
"""

from typing import List


def create_pdf(pages: List[str]) -> bytes:
    """
    Create a PDF with the given pages.
    
    Args:
        pages: List of text content for each page
        
    Returns:
        PDF file as bytes
    """
    if not pages:
        raise ValueError("At least one page is required")
    
    num_pages = len(pages)
    
    # Build page objects
    page_objects = []
    content_objects = []
    
    for i, page_content in enumerate(pages):
        page_num = i + 3  # Pages start from object 3
        content_num = page_num + num_pages  # Content objects after page objects
        
        page_objects.append(f"{page_num} 0 obj")
        page_objects.append(f"<< /Type /Page /Parent 2 0 R /Resources {2 + num_pages + num_pages + 1} 0 R /MediaBox [0 0 612 792] /Contents {content_num} 0 R >>")
        page_objects.append("endobj")
        
        stream_content = f"""BT
/F1 12 Tf
72 720 Td
({page_content}) Tj
ET"""
        
        content_objects.append(f"{content_num} 0 obj")
        content_objects.append(f"<< /Length {len(stream_content)} >>")
        content_objects.append("stream")
        content_objects.append(stream_content)
        content_objects.append("endstream")
        content_objects.append("endobj")
    
    # Build page references for Pages object
    page_refs = " ".join([f"{i + 3} 0 R" for i in range(num_pages)])
    
    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [{page_refs}] /Count {num_pages} >>
endobj
{chr(10).join(page_objects)}
{2 + num_pages + num_pages + 1} 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
{chr(10).join(content_objects)}
xref
0 {2 + num_pages + num_pages + 2}
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n"""

    # Add xref entries (simplified)
    for i in range(num_pages + num_pages + 1):
        pdf_content += f"\n{1000 + i * 100:010d} 00000 n"
    
    pdf_content += f"""
trailer
<< /Size {2 + num_pages + num_pages + 2} /Root 1 0 R >>
startxref
{5000 + sum(len(p) for p in pages)}
%%EOF"""
    
    return pdf_content.encode('latin-1')