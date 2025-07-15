"""
PDF Utilities Module

Provides utility functions for creating test PDFs.
"""

import re
from pathlib import Path
from typing import List

import pypdf


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
        
        # Split content into lines and position each line separately
        lines = page_content.split('\n')
        line_commands = []
        for i, line in enumerate(lines):
            if i == 0:
                line_commands.append(f"72 720 Td")
            else:
                line_commands.append(f"0 -15 Td")  # Move down 15 points for each line
            line_commands.append(f"({line}) Tj")
        
        stream_content = f"""BT
/F1 12 Tf
{chr(10).join(line_commands)}
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


def is_textual_pdf(
    path: str | Path,
    text_page_thresh: float = 0.2,   # ≤20% blank pages ⇒ treat as textual
    min_chars_per_page: int = 20     # Minimum characters per page to consider it textual
) -> float:
    """
    Classify a PDF as textual (machine‑readable) using pypdf.
    
    Returns:
        float: textual score from 0.0 (no text) to 1.0 (fully textual)
               0.8+ is pretty textual, <0.1 shows warning, 0.0 raises error with citations
    
    Args:
        text_page_thresh: max fraction of pages allowed to lack text
        min_chars_per_page: minimum characters per page to consider it textual
    """
    try:
        reader = pypdf.PdfReader(str(path))
        
        if not reader.pages:
            return 0.0
        
        pages_with_text = 0
        total_pages = len(reader.pages)
        
        for page in reader.pages:
            try:
                # Extract text from the page
                text = page.extract_text()
                
                # Remove whitespace and count actual characters
                cleaned_text = ''.join(text.split())
                
                # Check if page has substantial text
                if len(cleaned_text) >= min_chars_per_page:
                    pages_with_text += 1
                    
            except Exception:
                # If text extraction fails, consider page as non-textual
                continue
        
        # Calculate score based on pages with actual text
        if total_pages == 0:
            return 0.0
            
        textual_ratio = pages_with_text / total_pages
        
        # Apply threshold - if too many pages lack text, score drops
        textless_ratio = 1 - textual_ratio
        if textless_ratio > text_page_thresh:
            # Too many pages without text
            return textual_ratio * 0.5  # Penalize score
        
        return textual_ratio
        
    except Exception:
        # If PDF can't be read, assume it's not textual
        return 0.0