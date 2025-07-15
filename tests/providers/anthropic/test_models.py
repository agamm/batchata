"""Tests for Anthropic model definitions.

Testing:
1. Model configuration correctness
2. Model capability flags
3. Supported file types
"""

import pytest

from batchata.providers.anthropic.models import ANTHROPIC_MODELS


class TestAnthropicModels:
    """Test Anthropic model definitions."""
    
    def test_model_definitions_exist(self):
        """Test that all expected Anthropic models are defined."""
        expected_models = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022", 
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229"
        ]
        
        for model_name in expected_models:
            assert model_name in ANTHROPIC_MODELS
            model = ANTHROPIC_MODELS[model_name]
            assert model.name == model_name
            assert model.max_input_tokens > 0
            assert model.max_output_tokens > 0
            assert 0 < model.batch_discount <= 1.0
    
    def test_model_capabilities_consistency(self):
        """Test model capabilities are consistent."""
        for model_name, config in ANTHROPIC_MODELS.items():
            # All models should support structured output
            assert config.supports_structured_output is True
            
            # If supports files, should have file types
            if config.supports_files:
                assert len(config.file_types) > 0
                assert all(ext.startswith('.') for ext in config.file_types)
            else:
                assert len(config.file_types) == 0
            
            # Newer models should support images
            if "3-5" in model_name:  # Claude 3.5 models
                assert config.supports_images is True
            
            # All should have reasonable token limits
            assert config.max_input_tokens >= 100000
            assert config.max_output_tokens >= 4096
            
            # Batch discount should be reasonable
            assert 0.3 <= config.batch_discount <= 0.5
    
    @pytest.mark.parametrize("model_name,expected_features", [
        ("claude-3-5-sonnet-20241022", {
            "supports_images": True,
            "supports_files": True,
            "supports_citations": True,
            "max_input_tokens": 200000
        }),
        ("claude-3-5-haiku-20241022", {
            "supports_images": True,
            "supports_files": True,
            "supports_citations": True,
            "max_input_tokens": 200000
        }),
        ("claude-3-opus-20240229", {
            "supports_images": True,
            "supports_files": False,
            "supports_citations": True,
            "max_input_tokens": 200000
        })
    ])
    def test_specific_model_features(self, model_name, expected_features):
        """Test specific features for each model."""
        model = ANTHROPIC_MODELS[model_name]
        
        for feature, expected_value in expected_features.items():
            actual_value = getattr(model, feature)
            assert actual_value == expected_value, f"{model_name}.{feature} should be {expected_value}, got {actual_value}"