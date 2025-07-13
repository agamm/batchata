"""Tests for ModelConfig data class."""

import pytest

from batchata.providers import ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_model_config_basic(self):
        """Test basic model configuration."""
        config = ModelConfig(
            name="test-model",
            max_input_tokens=100000,
            max_output_tokens=4096,
            batch_discount=0.5
        )
        assert config.name == "test-model"
        assert config.max_input_tokens == 100000
        assert config.max_output_tokens == 4096
        assert config.batch_discount == 0.5
        assert config.supports_images is False
        assert config.supports_files is False
        assert config.file_types == []
    
    def test_model_config_advanced(self):
        """Test advanced model configuration."""
        config = ModelConfig(
            name="advanced-model",
            max_input_tokens=200000,
            max_output_tokens=8192,
            batch_discount=0.6,
            supports_images=True,
            supports_files=True,
            supports_citations=True,
            file_types=[".pdf", ".docx", ".txt"]
        )
        assert config.supports_images is True
        assert config.supports_files is True
        assert config.supports_citations is True
        assert len(config.file_types) == 3