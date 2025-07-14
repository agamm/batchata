"""Tests for ModelConfig class.

Testing:
1. Model configuration attributes and capabilities
2. Batch discount application
3. File type support validation
"""

import pytest

from batchata.providers.model_config import ModelConfig


class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_model_configuration_creation(self):
        """Test creating model configurations with different capabilities."""
        # Basic model
        basic_model = ModelConfig(
            name="test-basic",
            max_input_tokens=100000,
            max_output_tokens=4096,
            batch_discount=0.5,  # 50% discount
            supports_citations=False,
            supports_images=False,
            supports_files=False
        )
        
        assert basic_model.name == "test-basic"
        assert basic_model.max_input_tokens == 100000
        assert basic_model.max_output_tokens == 4096
        assert basic_model.batch_discount == 0.5
        assert basic_model.supports_citations is False
        assert basic_model.supports_images is False
        assert basic_model.supports_files is False
        assert basic_model.supports_structured_output is True  # Default
        assert basic_model.file_types == []  # Default empty list
        
        # Advanced model with all features
        advanced_model = ModelConfig(
            name="test-advanced",
            max_input_tokens=200000,
            max_output_tokens=8192,
            batch_discount=0.3,  # 70% discount (pay 30%)
            supports_citations=True,
            supports_images=True,
            supports_files=True,
            supports_structured_output=True,
            file_types=[".pdf", ".docx", ".txt"]
        )
        
        assert advanced_model.supports_citations is True
        assert advanced_model.supports_images is True
        assert advanced_model.supports_files is True
        assert advanced_model.max_input_tokens == 200000
        assert len(advanced_model.file_types) == 3
        assert ".pdf" in advanced_model.file_types
    
    def test_batch_discount_scenarios(self):
        """Test different batch discount configurations."""
        # No discount (full price)
        full_price = ModelConfig(
            name="no-discount",
            max_input_tokens=10000,
            max_output_tokens=1000,
            batch_discount=1.0  # Pay 100%
        )
        assert full_price.batch_discount == 1.0
        
        # Half price
        half_price = ModelConfig(
            name="half-price",
            max_input_tokens=10000,
            max_output_tokens=1000,
            batch_discount=0.5  # Pay 50%
        )
        assert half_price.batch_discount == 0.5
        
        # Heavy discount
        heavy_discount = ModelConfig(
            name="heavy-discount",
            max_input_tokens=10000,
            max_output_tokens=1000,
            batch_discount=0.1  # Pay only 10%
        )
        assert heavy_discount.batch_discount == 0.1
    
    @pytest.mark.parametrize("file_ext,file_types,expected", [
        (".pdf", [".pdf", ".docx"], True),
        (".txt", [".pdf", ".docx"], False),
        (".PDF", [".pdf"], False),  # Case sensitive
        (".docx", [".pdf", ".docx", ".txt"], True),
        ("", [], False),
    ])
    def test_file_type_support(self, file_ext, file_types, expected):
        """Test checking file type support."""
        model = ModelConfig(
            name="test-model",
            max_input_tokens=10000,
            max_output_tokens=1000,
            batch_discount=0.5,
            supports_files=bool(file_types),
            file_types=file_types
        )
        
        # Check if file type is supported
        is_supported = file_ext in model.file_types
        assert is_supported == expected
        
        # Verify supports_files flag consistency
        if file_types:
            assert model.supports_files is True