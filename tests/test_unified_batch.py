"""
Tests for unified batch() function that handles both messages and files.
"""

import pytest
import base64
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import BaseModel


class ResultModel(BaseModel):
    content: str
    status: str


class DocumentInfo(BaseModel):
    title: str
    content: str


def create_test_pdf(content: str) -> bytes:
    """Create a minimal valid PDF with given content."""
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


class TestUnifiedBatch:
    """Test the unified batch() function with either messages or files."""
    
    def test_batch_with_messages_only(self):
        """Test batch() with messages parameter (existing behavior)."""
        from src import batch
        
        messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}]
        ]
        
        with patch('src.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [
                {'custom_id': 'request_0', 'params': {}},
                {'custom_id': 'request_1', 'params': {}}
            ]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = False
            mock_provider._is_batch_completed.return_value = True
            mock_provider.get_results.return_value = []
            mock_provider.parse_results.return_value = [
                {"result": ResultModel(content="Response 1", status="ok"), "citations": None},
                {"result": ResultModel(content="Response 2", status="ok"), "citations": None}
            ]
            
            job = batch(
                messages=messages,
                model="claude-3-haiku-20240307",
                response_model=ResultModel
            )
            
            results = job.results()
            assert len(results) == 2
            assert results[0]["result"].content == "Response 1"
            assert results[1]["result"].content == "Response 2"
    
    def test_batch_with_files_and_prompt(self):
        """Test batch() with files parameter (batch_files behavior)."""
        from src import batch
        
        pdf1 = create_test_pdf("Document 1")
        pdf2 = create_test_pdf("Document 2")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "doc1.pdf"
            file2 = Path(temp_dir) / "doc2.pdf"
            file1.write_bytes(pdf1)
            file2.write_bytes(pdf2)
            
            with patch('src.core.get_provider_for_model') as mock_provider_func:
                mock_provider = MagicMock()
                mock_provider_func.return_value = mock_provider
                mock_provider.validate_batch.return_value = None
                mock_provider.prepare_batch_requests.return_value = [
                    {'custom_id': 'request_0', 'params': {}},
                    {'custom_id': 'request_1', 'params': {}}
                ]
                mock_provider.create_batch.return_value = "batch_123"
                mock_provider.has_citations_enabled.return_value = False
                mock_provider._is_batch_completed.return_value = True
                mock_provider.get_results.return_value = []
                mock_provider.parse_results.return_value = [
                    {"result": DocumentInfo(title="Doc1", content="Content 1"), "citations": None},
                    {"result": DocumentInfo(title="Doc2", content="Content 2"), "citations": None}
                ]
                
                job = batch(
                    files=[str(file1), str(file2)],
                    prompt="Extract document info",
                    model="claude-3-haiku-20240307",
                    response_model=DocumentInfo
                )
                
                results = job.results()
                assert len(results) == 2
                assert results[0]["result"].title == "Doc1"
                assert results[1]["result"].title == "Doc2"
    
    def test_batch_with_both_messages_and_files_raises_error(self):
        """Test that providing both messages and files raises ValueError."""
        from src import batch
        
        with pytest.raises(ValueError, match="Cannot provide both messages and files"):
            batch(
                messages=[[{"role": "user", "content": "Hello"}]],
                files=["test.pdf"],
                prompt="Extract info",
                model="claude-3-haiku-20240307"
            )
    
    def test_batch_with_neither_messages_nor_files_raises_error(self):
        """Test that providing neither messages nor files raises ValueError."""
        from src import batch
        
        with pytest.raises(ValueError, match="Must provide either messages or files"):
            batch(
                model="claude-3-haiku-20240307"
            )
    
    def test_batch_with_files_but_no_prompt_raises_error(self):
        """Test that using files without prompt raises ValueError."""
        from src import batch
        
        with pytest.raises(ValueError, match="prompt is required when using files"):
            batch(
                files=["test.pdf"],
                model="claude-3-haiku-20240307"
            )
    
    def test_batch_with_different_file_types(self):
        """Test batch() with string paths, Path objects, and bytes."""
        from src import batch
        
        pdf_bytes = create_test_pdf("Test Document")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # String path
            str_path = str(Path(temp_dir) / "string.pdf")
            Path(str_path).write_bytes(pdf_bytes)
            
            # Path object
            path_obj = Path(temp_dir) / "path.pdf"
            path_obj.write_bytes(pdf_bytes)
            
            # Raw bytes
            raw_bytes = pdf_bytes
            
            with patch('src.core.get_provider_for_model') as mock_provider_func:
                mock_provider = MagicMock()
                mock_provider_func.return_value = mock_provider
                mock_provider.validate_batch.return_value = None
                mock_provider.prepare_batch_requests.return_value = [
                    {'custom_id': f'request_{i}', 'params': {}} for i in range(3)
                ]
                mock_provider.create_batch.return_value = "batch_123"
                mock_provider.has_citations_enabled.return_value = False
                mock_provider._is_batch_completed.return_value = True
                mock_provider.get_results.return_value = []
                mock_provider.parse_results.return_value = [
                    {"result": DocumentInfo(title=f"Doc{i}", content="Test Document"), "citations": None}
                    for i in range(3)
                ]
                
                job = batch(
                    files=[str_path, path_obj, raw_bytes],
                    prompt="Extract info",
                    model="claude-3-haiku-20240307",
                    response_model=DocumentInfo
                )
                
                results = job.results()
                assert len(results) == 3
                for result_entry in results:
                    assert result_entry["result"].content == "Test Document"
    
    def test_batch_with_empty_messages_list(self):
        """Test batch() with empty messages list."""
        from src import batch
        
        job = batch(
            messages=[],
            model="claude-3-haiku-20240307"
        )
        
        assert job.is_complete()
        assert job.results() == []
    
    def test_batch_with_empty_files_list(self):
        """Test batch() with empty files list."""
        from src import batch
        
        job = batch(
            messages=[],
            model="claude-3-haiku-20240307"
        )
        
        assert job.is_complete()
        assert job.results() == []
    
    def test_batch_with_files_and_citations(self):
        """Test batch() with files and citations enabled."""
        from src import batch
        from src import Citation
        
        pdf_bytes = create_test_pdf("Test Document")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_bytes(pdf_bytes)
            
            with patch('src.core.get_provider_for_model') as mock_provider_func:
                mock_provider = MagicMock()
                mock_provider_func.return_value = mock_provider
                mock_provider.validate_batch.return_value = None
                mock_provider.prepare_batch_requests.return_value = [
                    {'custom_id': 'request_0', 'params': {}}
                ]
                mock_provider.create_batch.return_value = "batch_123"
                mock_provider.has_citations_enabled.return_value = True
                mock_provider._is_batch_completed.return_value = True
                mock_provider.get_results.return_value = []
                
                # Mock citations
                mock_citations = {
                    "title": [Citation(
                        type="page_location",
                        cited_text="Test Document",
                        document_index=0,
                        document_title="test.pdf",
                        start_page_number=1,
                        end_page_number=1
                    )]
                }
                
                mock_provider.parse_results.return_value = [
                    {"result": DocumentInfo(title="Test", content="Test Document"), "citations": mock_citations}
                ]
                
                job = batch(
                    files=[str(pdf_path)],
                    prompt="Extract info",
                    model="claude-3-haiku-20240307",
                    response_model=DocumentInfo,
                    enable_citations=True
                )
                
                results = job.results()
                
                assert len(results) == 1
                assert results[0]["result"].title == "Test"
                citations = results[0]["citations"]
                assert "title" in citations
                assert len(citations["title"]) == 1
    
    def test_batch_messages_without_prompt_works(self):
        """Test that messages without prompt parameter works fine."""
        from src import batch
        
        messages = [[{"role": "user", "content": "Hello"}]]
        
        with patch('src.core.get_provider_for_model') as mock_provider_func:
            mock_provider = MagicMock()
            mock_provider_func.return_value = mock_provider
            mock_provider.validate_batch.return_value = None
            mock_provider.prepare_batch_requests.return_value = [
                {'custom_id': 'request_0', 'params': {}}
            ]
            mock_provider.create_batch.return_value = "batch_123"
            mock_provider.has_citations_enabled.return_value = False
            
            # No prompt parameter needed for messages
            job = batch(
                messages=messages,
                model="claude-3-haiku-20240307"
            )
            
            assert job is not None
            assert job._batch_id == "batch_123"