"""Tests for OpenAI model definitions.

Testing:
1. Model configuration correctness
2. Model capability flags consistency  
3. Specific model features validation
"""

import pytest

from batchata.providers.openai.models import OPENAI_MODELS


class TestOpenAIModels:
    """Test OpenAI model definitions."""
    
    def test_model_definitions_exist(self):
        """Test that all expected OpenAI models are defined with valid configs."""
        expected_models = [
            "gpt-4.1-2025-04-14",
            "o4-mini-2025-04-16", 
            "o3-2025-04-16",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o-mini-2024-07-18"
        ]
        
        for model_name in expected_models:
            assert model_name in OPENAI_MODELS
            model = OPENAI_MODELS[model_name]
            assert model.name == model_name
            assert model.max_input_tokens > 0
            assert model.max_output_tokens > 0
            assert 0 < model.batch_discount <= 1.0