"""Tests for LLM utilities.

Testing:
1. Simple token counting functionality
2. Different character types (Latin, CJK, emoji)
3. Conservative estimation approach
"""

import pytest

from batchata.utils.llm import token_count_simple


class TestLLMUtils:
    """Test LLM utility functions."""
    
    def test_token_count_empty_text(self):
        """Test token counting for empty text."""
        assert token_count_simple("") == 0
        assert token_count_simple(None) == 0
    
    def test_token_count_latin_text(self):
        """Test token counting for Latin character text."""
        # Simple English text
        text = "Hello world, this is a test."
        count = token_count_simple(text)
        
        # Should be conservative (overestimate)
        # 7 words * 1.6 + 5 = ~16, plus character-based estimate
        assert count > 15
        assert count < 50  # But not too excessive
        
        # Longer text
        long_text = "The quick brown fox jumps over the lazy dog. " * 5
        count = token_count_simple(long_text)
        # 45 words * 1.6 + 5 = ~77
        assert count > 70
        assert count < 150
    
    @pytest.mark.parametrize("text,min_tokens", [
        ("你好世界", 8),  # CJK: 4 chars * 2 tokens each
        ("こんにちは", 10),  # Japanese: 5 chars * 2 tokens
        ("안녕하세요", 10),  # Korean: 5 chars * 2 tokens
        ("你好 Hello", 12),  # Mixed CJK and Latin
    ])
    def test_token_count_cjk_text(self, text, min_tokens):
        """Test token counting for CJK characters."""
        count = token_count_simple(text)
        # CJK characters count as 2 tokens each in conservative estimate
        assert count >= min_tokens
        
    def test_token_count_emoji_text(self):
        """Test token counting for emoji characters."""
        # Emojis should count as 3 tokens each (conservative)
        text = "Hello 😀🎉"
        count = token_count_simple(text)
        
        # "Hello" (1 word) + 2 emojis * 3 tokens
        # Word-based: 1 * 1.6 + 5 = ~7
        # Char-based includes 6 tokens for emojis
        assert count >= 15
        
        # Multiple emojis
        emoji_text = "🚀🌟💡🎯"
        count = token_count_simple(emoji_text)
        # 4 emojis * 3 tokens = 12, plus overhead
        assert count >= 12
    
    def test_token_count_mixed_content(self):
        """Test token counting for mixed content types."""
        # Mix of Latin, CJK, emoji, and special characters
        text = "Hello 世界! 🌍 Testing... #hashtag @mention"
        count = token_count_simple(text)
        
        # Should handle all types conservatively
        assert count > 20
        
        # Complex mixed text
        complex_text = """
        English text here.
        中文文本在这里。
        日本語のテキスト。
        Emojis: 🎉🎊🎈
        Special: ©®™
        """
        count = token_count_simple(complex_text)
        assert count > 50  # Conservative estimate for complex content