"""Tests for Gemini model configurations."""

import pytest
from batchata.providers.gemini.models import GEMINI_MODELS


class TestGeminiModels:
    """Test Gemini model configurations."""
    
    def test_models_exist(self):
        """Test that Gemini models are defined."""
        assert len(GEMINI_MODELS) > 0
        assert "gemini-2.5-flash" in GEMINI_MODELS
        assert "gemini-2.5-pro" in GEMINI_MODELS
    
    def test_model_configurations(self):
        """Test model configurations are valid."""
        for model_name, config in GEMINI_MODELS.items():
            assert config.name == model_name
            assert config.max_input_tokens > 0
            assert config.max_output_tokens > 0
            assert 0 <= config.batch_discount <= 1
            assert isinstance(config.supports_images, bool)
            assert isinstance(config.supports_files, bool)
            assert isinstance(config.supports_structured_output, bool)
            assert isinstance(config.file_types, list)
    
    def test_batch_discount(self):
        """Test batch discount is set correctly."""
        for config in GEMINI_MODELS.values():
            assert config.batch_discount == 0.5  # 50% discount confirmed in docs
    
    def test_file_support(self):
        """Test file support configurations."""
        for config in GEMINI_MODELS.values():
            assert config.supports_files is True
            assert config.supports_images is True
            # Should not include .docx anymore
            assert ".docx" not in config.file_types
            # Should include these formats
            assert ".pdf" in config.file_types
            assert ".txt" in config.file_types
            assert ".jpg" in config.file_types
            assert ".png" in config.file_types
    
    def test_structured_output_support(self):
        """Test structured output is supported."""
        for config in GEMINI_MODELS.values():
            assert config.supports_structured_output is True
    
    def test_context_windows(self):
        """Test context window sizes are reasonable."""
        # Pro models should have larger context
        pro_config = GEMINI_MODELS["gemini-2.5-pro"]
        flash_config = GEMINI_MODELS["gemini-2.5-flash"]
        
        assert pro_config.max_input_tokens >= flash_config.max_input_tokens
        assert pro_config.max_input_tokens == 2097152  # 2M tokens
        assert flash_config.max_input_tokens == 1048576  # 1M tokens