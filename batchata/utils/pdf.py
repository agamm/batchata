"""
PDF Utilities Module

Provides utility functions for creating test PDFs and extracting text.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional

import pypdf
from ..utils import get_logger

logger = get_logger(__name__)


def create_pdf(pages: List[str]) -> bytes:
    """
    Create a PDF with the given pages using pypdf.PdfWriter.
    
    Args:
        pages: List of text content for each page
        
    Returns:
        PDF file as bytes
    """
    if not pages:
        raise ValueError("At least one page is required")
    
    import io
    from pypdf import PdfWriter, PageObject
    from pypdf.generic import StreamObject, DictionaryObject, NameObject
    
    writer = PdfWriter()
    
    for page_content in pages:
        # Create a blank page
        page = PageObject.create_blank_page(width=612, height=792)  # Letter size
        
        # Create text content stream
        lines = page_content.split('\n')
        text_commands = []
        text_commands.append("BT")  # Begin text
        text_commands.append("/F1 12 Tf")  # Set font
        text_commands.append("72 720 Td")  # Position at top-left with margin
        
        for i, line in enumerate(lines):
            if i > 0:
                text_commands.append("0 -15 Td")  # Move down for next line
            # Escape parentheses and backslashes in text
            escaped_line = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            text_commands.append(f"({escaped_line}) Tj")  # Show text
        
        text_commands.append("ET")  # End text
        content_stream = "\n".join(text_commands)
        
        # Create StreamObject properly as shown in pypdf tests
        stream_object = StreamObject()
        stream_object[NameObject("/Type")] = NameObject("/Text")
        stream_object._data = content_stream.encode()
        
        # Add the content stream to the page
        page[NameObject("/Contents")] = stream_object
        
        # Add basic font resources
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica")
                })
            })
        })
        
        writer.add_page(page)
    
    # Write to bytes
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def is_textual_pdf(
    path: str | Path,
    text_page_thresh: float = 0.2,   # ≤20% blank pages ⇒ treat as textual
    min_chars_per_page: int = 20     # Minimum characters per page to consider it textual
) -> float:
    """
    Classify a PDF as textual (machine‑readable) using pypdf.
    
    Args:
        path: Path to the PDF file to analyze
        text_page_thresh: max fraction of pages allowed to lack text
        min_chars_per_page: minimum characters per page to consider it textual
        
    Returns:
        float: textual score from 0.0 (no text) to 1.0 (fully textual)
               0.8+ is pretty textual, <0.1 shows warning, 0.0 raises error with citations
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


def extract_text_from_pdf(path: str | Path) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        reader = pypdf.PdfReader(str(path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
            except Exception as e:
                logger.debug(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF {path}: {e}")
        return ""


def get_pdf_info(path: str | Path) -> Tuple[int, bool, Optional[str]]:
    """
    Get information about a PDF file for cost estimation.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Tuple of (page_count, is_textual, extracted_text)
        - page_count: Number of pages in the PDF
        - is_textual: Whether the PDF has extractable text
        - extracted_text: Text content if textual, None otherwise
    """
    try:
        reader = pypdf.PdfReader(str(path))
        page_count = len(reader.pages)
        
        # Check if PDF is textual
        textual_score = is_textual_pdf(path)
        is_textual = textual_score > 0.5  # Consider textual if >50% pages have text
        
        # Extract text if textual
        extracted_text = None
        if is_textual:
            extracted_text = extract_text_from_pdf(path)
            if not extracted_text.strip():
                is_textual = False
                extracted_text = None
        
        logger.debug(f"PDF info for {path}: {page_count} pages, textual={is_textual}, "
                    f"text_length={len(extracted_text) if extracted_text else 0}")
        
        return page_count, is_textual, extracted_text
        
    except Exception as e:
        logger.error(f"Failed to get PDF info for {path}: {e}")
        return 0, False, None


def estimate_pdf_tokens(path: str | Path, prompt: Optional[str] = None, 
                       pdf_token_multiplier: float = 1.5,
                       tokens_per_page: int = 2000) -> int:
    """
    Estimate token count for a PDF file.
    
    Provider-specific tokens per page estimates:
    - Anthropic: 1,500-3,000 tokens/page (default: 2000)
    - Gemini: ~258 tokens/page
    - OpenAI: 300-1,280 tokens/page (use: 1000)
    
    Args:
        path: Path to the PDF file
        prompt: Optional prompt to include in token count
        pdf_token_multiplier: Deprecated, kept for compatibility
        tokens_per_page: Tokens per page estimate (default: 2000 for Anthropic)
        
    Returns:
        Estimated token count
    """
    from .llm import token_count_simple
    
    try:
        # Get page count
        reader = pypdf.PdfReader(str(path))
        page_count = len(reader.pages)
        
        # Use provider-specific tokens per page estimate
        pdf_tokens = page_count * tokens_per_page
        
        # Add prompt tokens
        prompt_tokens = token_count_simple(prompt) if prompt else 0
        
        # Add minimal overhead for PDF processing
        PDF_TOKEN_OVERHEAD = 100  # tokens
        overhead_tokens = PDF_TOKEN_OVERHEAD
        
        total_tokens = pdf_tokens + prompt_tokens + overhead_tokens
        
        logger.debug(
            f"PDF {path}: {page_count} pages × {tokens_per_page} = {pdf_tokens} tokens, "
            f"prompt: {prompt_tokens}, total: {total_tokens}"
        )
        
        return total_tokens
        
    except Exception as e:
        logger.warning(f"Failed to estimate PDF tokens: {e}")
        return 0