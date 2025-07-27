"""Tests for Gemini model configurations."""

import pytest
from batchata.providers.gemini.models import GEMINI_MODELS


class TestGeminiModels:
    """Test Gemini model configurations."""
    
    def test_models_exist(self):
        """Test that Gemini models are defined."""
        assert len(GEMINI_MODELS) > 0
    
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
    
