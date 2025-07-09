import pytest
import base64
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from src import batch


class DocumentInfo(BaseModel):
    title: str
    content: str
    page_count: int


def create_test_pdf(content: str) -> bytes:
    """Create a minimal valid PDF with given content.
    
    This creates a basic but valid PDF structure that should be
    readable by PDF viewers and Claude's API.
    """
    # Create a minimal but valid PDF structure
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
<< /Length {len(content) + 60} >>
stream
BT
/F1 12 Tf
72 720 Td
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
{400 + len(content)}
%%EOF"""
    return pdf_content.encode('latin-1')


def test_pdf_to_document_block():
    """Test converting PDF to document content block."""
    from src.file_processing import pdf_to_document_block
    
    pdf_content = create_test_pdf("Test PDF Content")
    doc_block = pdf_to_document_block(pdf_content)
    
    assert doc_block["type"] == "document"
    assert doc_block["source"]["type"] == "base64"
    assert doc_block["source"]["media_type"] == "application/pdf"
    assert doc_block["source"]["data"] == base64.b64encode(pdf_content).decode('utf-8')


def test_batch_with_single_pdf():
    """Test processing a single PDF file."""
    pdf_content = create_test_pdf("Invoice #12345")
    
    with patch('src.core.AnthropicBatchProvider') as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        mock_provider.validate_batch.return_value = None
        mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
        mock_provider.create_batch.return_value = "batch_123"
        mock_provider.has_citations_enabled.return_value = False
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []
        mock_provider.parse_results.return_value = ([
            DocumentInfo(title="Invoice", content="Invoice #12345", page_count=1)
        ], None)
        
        messages = [[
            {"role": "user", "content": [
                {"type": "text", "text": "Extract information from this PDF"},
                {"type": "document", "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(pdf_content).decode('utf-8')
                }}
            ]}
        ]]
        
        job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=DocumentInfo
        )
        
        results = job.results()
        assert len(results) == 1
        assert results[0].title == "Invoice"
        assert results[0].content == "Invoice #12345"


def test_batch_with_multiple_pdfs():
    """Test processing multiple PDFs in a batch."""
    pdf1 = create_test_pdf("Document 1")
    pdf2 = create_test_pdf("Document 2")
    pdf3 = create_test_pdf("Document 3")
    
    with patch('src.core.AnthropicBatchProvider') as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        mock_provider.validate_batch.return_value = None
        mock_provider.prepare_batch_requests.return_value = [
            {'custom_id': 'request_0', 'params': {}},
            {'custom_id': 'request_1', 'params': {}},
            {'custom_id': 'request_2', 'params': {}}
        ]
        mock_provider.create_batch.return_value = "batch_123"
        mock_provider.has_citations_enabled.return_value = False
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []
        mock_provider.parse_results.return_value = ([
            DocumentInfo(title="Doc1", content="Document 1", page_count=1),
            DocumentInfo(title="Doc2", content="Document 2", page_count=1),
            DocumentInfo(title="Doc3", content="Document 3", page_count=1)
        ], None)
        
        messages = []
        for i, pdf_content in enumerate([pdf1, pdf2, pdf3], 1):
            messages.append([{
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Extract info from document {i}"},
                    {"type": "document", "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(pdf_content).decode('utf-8')
                    }}
                ]
            }])
        
        job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=DocumentInfo
        )
        
        results = job.results()
        assert len(results) == 3
        assert results[0].title == "Doc1"
        assert results[1].title == "Doc2"
        assert results[2].title == "Doc3"


def test_batch_with_file_paths():
    """Test processing PDFs from file paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_files = []
        for i in range(3):
            pdf_path = Path(temp_dir) / f"test_{i}.pdf"
            pdf_content = create_test_pdf(f"Test Document {i}")
            pdf_path.write_bytes(pdf_content)
            pdf_files.append(pdf_path)
        
        with patch('src.core.AnthropicBatchProvider') as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [
                {'custom_id': f'request_{i}', 'params': {}} for i in range(3)
            ]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = False
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = ([
                DocumentInfo(title=f"Doc{i}", content=f"Test Document {i}", page_count=1)
                for i in range(3)
            ], None)
            
            from src.file_processing import batch_files
            
            job = batch_files(
                files=pdf_files,
                prompt="Extract document information",
                model="claude-3-haiku-20240307",
                response_model=DocumentInfo
            )
            
            results = job.results()
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.title == f"Doc{i}"
                assert result.content == f"Test Document {i}"