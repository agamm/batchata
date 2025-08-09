"""Tests for PDF utilities."""

import pytest
from pathlib import Path
import tempfile

from batchata.utils.pdf import create_pdf, is_textual_pdf, estimate_pdf_tokens


class TestCreatePdf:
    """Test PDF creation functionality."""
    
    def test_create_single_page(self):
        """Test creating a single page PDF."""
        pages = ["Hello World"]
        pdf_bytes = create_pdf(pages)
        
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b'%PDF-1.3')
        assert pdf_bytes.endswith(b'%%EOF\n')
        assert b'Hello World' in pdf_bytes
    
    def test_create_multi_page(self):
        """Test creating a multi-page PDF."""
        pages = ["Page 1", "Page 2", "Page 3"]
        pdf_bytes = create_pdf(pages)
        
        assert isinstance(pdf_bytes, bytes)
        assert b'Page 1' in pdf_bytes
        assert b'Page 2' in pdf_bytes
        assert b'Page 3' in pdf_bytes
    
    def test_empty_pages_raises_error(self):
        """Test that empty pages list raises ValueError."""
        with pytest.raises(ValueError, match="At least one page is required"):
            create_pdf([])


class TestIsTextualPdf:
    """Test PDF textual score detection functionality."""
    
    def test_textual_pdf_high_score(self):
        """Test that textual PDFs get high textual scores."""
        # Create a text-heavy PDF
        pages = [
            "This is a text-based PDF with lots of content",
            "Page 2 has even more text content",
            "Page 3 continues with readable text"
        ]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Should get high textual score
            score = is_textual_pdf(tmp.name)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert score > 0.5  # Should be reasonably textual
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_empty_pdf_zero_score(self):
        """Test that empty/malformed PDFs get zero score."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"not a real pdf")
            tmp.flush()
            
            # Should get zero score
            score = is_textual_pdf(tmp.name)
            assert score == 0.0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_threshold_parameters(self):
        """Test that threshold parameters affect scoring."""
        pages = ["Some text content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Different thresholds
            score_default = is_textual_pdf(tmp.name)
            score_strict = is_textual_pdf(
                tmp.name, 
                text_page_thresh=0.01,  # Very strict
            )
            
            # Both should be valid scores
            assert isinstance(score_default, float)
            assert isinstance(score_strict, float)
            assert 0.0 <= score_default <= 1.0
            assert 0.0 <= score_strict <= 1.0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_path_types(self):
        """Test that function accepts both string and Path objects."""
        pages = ["Test content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with string path
            score_str = is_textual_pdf(tmp.name)
            
            # Test with Path object
            score_path = is_textual_pdf(Path(tmp.name))
            
            # Results should be identical
            assert score_str == score_path
            assert isinstance(score_str, float)
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_score_ranges(self):
        """Test score interpretation ranges."""
        pages = ["Some text content for testing"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            score = is_textual_pdf(tmp.name)
            
            # Verify score is in valid range
            assert 0.0 <= score <= 1.0
            
            # For our simple text PDF, should be reasonably high
            assert score > 0.3  # Should detect some text content
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_nonexistent_file_returns_zero(self):
        """Test that nonexistent files return 0.0 score."""
        score = is_textual_pdf("/nonexistent/path/file.pdf")
        assert score == 0.0


class TestEstimatePdfTokens:
    """Test PDF token estimation functionality."""
    
    def test_basic_token_estimation(self):
        """Test basic token estimation for a simple PDF."""
        # Create a 3-page test PDF
        pages = [
            "Page 1: This is some content for testing token estimation.",
            "Page 2: More content here to make it realistic for the test.",
            "Page 3: Final page with conclusion and summary text."
        ]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test basic estimation
            tokens = estimate_pdf_tokens(tmp.name)
            
            # Should return reasonable token count
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Should be roughly pages × default_tokens_per_page + overhead
            # Default is 2000 tokens per page + 100 overhead = 3 × 2000 + 100 = 6100
            expected_range = (6000, 6200)  # Allow small variance
            assert expected_range[0] <= tokens <= expected_range[1]
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_token_estimation_with_prompt(self):
        """Test token estimation includes prompt tokens."""
        pages = ["Single page content"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with prompt
            prompt = "Analyze this document and extract key information please."
            tokens_with_prompt = estimate_pdf_tokens(tmp.name, prompt)
            
            # Test without prompt
            tokens_without_prompt = estimate_pdf_tokens(tmp.name)
            
            # Should include additional tokens for the prompt
            assert tokens_with_prompt > tokens_without_prompt
            
            # Difference should be reasonable for the prompt length
            prompt_tokens = tokens_with_prompt - tokens_without_prompt
            assert 10 <= prompt_tokens <= 30  # Reasonable range for prompt
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_custom_tokens_per_page(self):
        """Test custom tokens_per_page parameter."""
        pages = ["Page 1", "Page 2"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with different tokens_per_page values
            tokens_default = estimate_pdf_tokens(tmp.name)
            tokens_high = estimate_pdf_tokens(tmp.name, tokens_per_page=3000)
            tokens_low = estimate_pdf_tokens(tmp.name, tokens_per_page=1000)
            
            # Higher tokens_per_page should give higher estimate
            assert tokens_high > tokens_default
            assert tokens_default > tokens_low
            
            # Specific calculations (2 pages + 100 overhead)
            assert tokens_high == 2 * 3000 + 100  # 6100
            assert tokens_low == 2 * 1000 + 100   # 2100
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_path_types_accepted(self):
        """Test that function accepts both string and Path objects."""
        pages = ["Test content for path types"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with string path
            tokens_str = estimate_pdf_tokens(tmp.name)
            
            # Test with Path object  
            tokens_path = estimate_pdf_tokens(Path(tmp.name))
            
            # Results should be identical
            assert tokens_str == tokens_path
            assert isinstance(tokens_str, int)
            assert tokens_str > 0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_page_count_scaling(self):
        """Test that token estimation scales with page count."""
        # Create PDFs with different page counts
        single_page = create_pdf(["Single page"])
        multi_page = create_pdf(["Page 1", "Page 2", "Page 3", "Page 4", "Page 5"])
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp2:
            
            tmp1.write(single_page)
            tmp2.write(multi_page)
            tmp1.flush()
            tmp2.flush()
            
            tokens_1_page = estimate_pdf_tokens(tmp1.name)
            tokens_5_pages = estimate_pdf_tokens(tmp2.name)
            
            # 5-page PDF should have roughly 5x more tokens than 1-page
            # (accounting for fixed overhead)
            expected_ratio = 5  # 5 pages vs 1 page
            actual_ratio = (tokens_5_pages - 100) / (tokens_1_page - 100)  # Remove overhead
            
            # Should be close to expected ratio
            assert abs(actual_ratio - expected_ratio) < 0.1
            
            # Clean up
            Path(tmp1.name).unlink()
            Path(tmp2.name).unlink()
    
    def test_nonexistent_file_returns_zero(self):
        """Test that nonexistent files return 0 tokens."""
        tokens = estimate_pdf_tokens("/nonexistent/path/file.pdf")
        assert tokens == 0
    
    def test_deprecated_parameters_still_work(self):
        """Test that deprecated parameters don't break the function."""
        pages = ["Test page for deprecated params"]
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test with deprecated pdf_token_multiplier parameter
            tokens = estimate_pdf_tokens(
                tmp.name, 
                pdf_token_multiplier=2.0  # Should be ignored now
            )
            
            # Should still work and return reasonable tokens
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Should be same as default (parameter is ignored)
            tokens_default = estimate_pdf_tokens(tmp.name)
            assert tokens == tokens_default
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_realistic_token_ranges(self):
        """Test that token estimates are in realistic ranges."""
        # Test different PDF sizes
        test_cases = [
            (["Short page"], "small PDF", (2000, 2200)),
            (["Page 1", "Page 2"], "medium PDF", (4000, 4200)), 
            (["P1", "P2", "P3", "P4", "P5"], "large PDF", (10000, 10200))
        ]
        
        for pages, description, expected_range in test_cases:
            pdf_bytes = create_pdf(pages)
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp.flush()
                
                tokens = estimate_pdf_tokens(tmp.name)
                
                # Should be in expected range
                assert expected_range[0] <= tokens <= expected_range[1], \
                    f"{description} tokens {tokens} not in range {expected_range}"
                
                # Clean up
                Path(tmp.name).unlink()
    
    def test_provider_specific_token_estimates(self):
        """Test that different providers get appropriate token estimates."""
        pages = ["Test page 1", "Test page 2", "Test page 3"]  # 3 pages
        pdf_bytes = create_pdf(pages)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            
            # Test different provider estimates
            anthropic_tokens = estimate_pdf_tokens(tmp.name, tokens_per_page=2000)  # Anthropic
            gemini_tokens = estimate_pdf_tokens(tmp.name, tokens_per_page=258)      # Gemini  
            openai_tokens = estimate_pdf_tokens(tmp.name, tokens_per_page=1000)     # OpenAI
            
            # Should reflect provider differences (3 pages + 100 overhead)
            assert anthropic_tokens == 3 * 2000 + 100  # 6100
            assert gemini_tokens == 3 * 258 + 100      # 874
            assert openai_tokens == 3 * 1000 + 100     # 3100
            
            # Verify ordering: Anthropic > OpenAI > Gemini
            assert anthropic_tokens > openai_tokens > gemini_tokens
            
            # Clean up
            Path(tmp.name).unlink()