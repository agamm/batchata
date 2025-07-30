"""Test PDF token estimation functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from batchata.utils.pdf import extract_text_from_pdf, get_pdf_info, estimate_pdf_tokens
from batchata.providers.anthropic.anthropic import AnthropicProvider
from batchata.core.job import Job


class TestPDFTextExtraction:
    """Test PDF text extraction utilities."""
    
    def test_extract_text_from_pdf_success(self):
        """Test successful text extraction from PDF."""
        # Mock pypdf reader
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        
        with patch('pypdf.PdfReader', return_value=mock_reader):
            text = extract_text_from_pdf("test.pdf")
            
            assert text == "Page 1 content\n\nPage 2 content"
            assert mock_page1.extract_text.called
            assert mock_page2.extract_text.called
    
    def test_extract_text_from_pdf_empty_pages(self):
        """Test extraction with empty pages."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "  "  # Whitespace only
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Valid content"
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        
        with patch('pypdf.PdfReader', return_value=mock_reader):
            text = extract_text_from_pdf("test.pdf")
            
            # Should only include non-empty pages
            assert text == "Valid content"
    
    def test_extract_text_from_pdf_failure(self):
        """Test handling of extraction failure."""
        with patch('pypdf.PdfReader', side_effect=Exception("PDF read error")):
            text = extract_text_from_pdf("test.pdf")
            
            assert text == ""  # Should return empty string on failure
    
    def test_get_pdf_info_textual(self):
        """Test getting info from textual PDF."""
        # Mock reader with textual pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "This is a long text content with many words"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Another page with substantial text content"
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        
        with patch('pypdf.PdfReader', return_value=mock_reader):
            page_count, is_textual, extracted_text = get_pdf_info("test.pdf")
            
            assert page_count == 2
            assert is_textual is True
            assert "long text content" in extracted_text
            assert "Another page" in extracted_text
    
    def test_get_pdf_info_image_based(self):
        """Test getting info from image-based PDF."""
        # Mock reader with minimal text
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "1"  # Very minimal text
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = ""
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        
        with patch('pypdf.PdfReader', return_value=mock_reader):
            page_count, is_textual, extracted_text = get_pdf_info("test.pdf")
            
            assert page_count == 2
            assert is_textual is False
            assert extracted_text is None


class TestPDFTokenEstimation:
    """Test the standalone PDF token estimation utility."""
    
    def test_estimate_pdf_tokens_textual(self):
        """Test token estimation for textual PDF."""
        with patch('batchata.utils.pdf.get_pdf_info') as mock_info:
            mock_info.return_value = (5, True, "Sample text content " * 100)  # Textual PDF
            
            tokens = estimate_pdf_tokens(Path("test.pdf"), prompt="Analyze this")
            
            # Should use extracted text with 1.5x multiplier
            mock_info.assert_called_once_with(Path("test.pdf"))
            assert tokens > 0
    
    def test_estimate_pdf_tokens_image_based(self):
        """Test token estimation for image-based PDF."""
        with patch('batchata.utils.pdf.get_pdf_info') as mock_info:
            mock_info.return_value = (10, False, None)  # Image-based PDF, 10 pages
            
            tokens = estimate_pdf_tokens(Path("test.pdf"), prompt="Extract text")
            
            # Should use page count estimation (10 * 2000 = 20000) plus prompt tokens
            mock_info.assert_called_once_with(Path("test.pdf"))
            assert tokens >= 20000  # At least 20k tokens for 10 pages


class TestAnthropicPDFCostEstimation:
    """Test Anthropic provider's PDF cost estimation."""
    
    def test_estimate_cost_with_textual_pdf(self):
        """Test cost estimation for textual PDF."""
        # Mock the API key requirement
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(auto_register=False)
        
        # Create a job with PDF file
        pdf_path = Path("test_document.pdf")
        job = Job(
            id="test-job",
            file=pdf_path,
            prompt="Analyze this document",
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            enable_citations=True
        )
        
        # Mock tokencost and just test the regular flow
        with patch('tokencost.calculate_cost_by_tokens') as mock_calc:
            mock_calc.side_effect = [0.05, 0.01]  # Input cost, output cost
            
            # Mock the prepare_messages function to avoid file operations
            with patch('batchata.providers.anthropic.anthropic.prepare_messages') as mock_prepare:
                mock_prepare.return_value = ([{"role": "user", "content": "test"}], "system prompt")
                
                cost = provider.estimate_cost([job])
                
                # Should call prepare_messages
                mock_prepare.assert_called_once_with(job)
                # Cost should be calculated
                assert cost > 0
    
    def test_estimate_cost_with_image_pdf(self):
        """Test cost estimation for image-based PDF."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(auto_register=False)
        
        # Create a job with PDF file
        pdf_path = Path("scanned_document.pdf")
        job = Job(
            id="test-job",
            file=pdf_path,
            prompt="Extract text from this scan",
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            enable_citations=True
        )
        
        # Mock tokencost
        with patch('tokencost.calculate_cost_by_tokens') as mock_calc:
            mock_calc.side_effect = [1.2, 0.01]  # Higher input cost for more tokens
            
            # Mock the prepare_messages function to avoid file operations
            with patch('batchata.providers.anthropic.anthropic.prepare_messages') as mock_prepare:
                mock_prepare.return_value = ([{"role": "user", "content": "test"}], "system prompt")
                
                cost = provider.estimate_cost([job])
                
                # Should call prepare_messages and calculate cost
                mock_prepare.assert_called_once_with(job)
                assert cost > 0
    
    def test_estimate_cost_with_regular_messages(self):
        """Test that regular message estimation still works."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(auto_register=False)
        
        # Create a job with regular messages
        job = Job(
            id="test-job",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            temperature=0.7
        )
        
        # Mock tokencost
        with patch('tokencost.calculate_cost_by_tokens') as mock_calc:
            mock_calc.side_effect = [0.001, 0.0001]
            
            cost = provider.estimate_cost([job])
            
            # Should not call get_pdf_info
            assert mock_calc.call_count == 2
            assert cost == pytest.approx(0.00055, rel=0.01)  # (0.001 + 0.0001) * 0.5
    
    def test_estimate_cost_mixed_jobs(self):
        """Test cost estimation with mix of PDFs and messages."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(auto_register=False)
        
        jobs = [
            Job(
                id="pdf-job",
                file=Path("doc.pdf"),
                prompt="Summarize",
                model="claude-3-sonnet-20240229",
                max_tokens=500
            ),
            Job(
                id="msg-job",
                messages=[{"role": "user", "content": "Hi"}],
                model="claude-3-sonnet-20240229",
                max_tokens=100
            )
        ]
        
        with patch('tokencost.calculate_cost_by_tokens') as mock_calc:
            # PDF job costs + message job costs
            mock_calc.side_effect = [0.02, 0.005, 0.001, 0.0001]
            
            # Mock prepare_messages for the PDF job to avoid file operations
            with patch('batchata.providers.anthropic.anthropic.prepare_messages') as mock_prepare:
                mock_prepare.return_value = ([{"role": "user", "content": "test"}], "system prompt")
                
                total_cost = provider.estimate_cost(jobs)
                
                # Should handle both job types
                assert mock_calc.call_count == 4  # 2 calls per job
                assert total_cost > 0