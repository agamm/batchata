"""Tests for base Provider class.

Testing:
1. Abstract method enforcement
2. Parameter validation with Pydantic schemas
3. Model support and configuration
"""

import pytest
from typing import List, Optional, Dict
from unittest.mock import MagicMock

from batchata.providers import Provider
from batchata.providers.model_config import ModelConfig
from batchata.core.job import Job
from batchata.core.job_result import JobResult
from batchata.exceptions import ValidationError


class ConcreteProvider(Provider):
    """Concrete implementation for testing."""
    
    def __init__(self):
        super().__init__()
        self.models = {
            "test-model": ModelConfig(
                name="test-model",
                max_input_tokens=1000,
                max_output_tokens=500,
                batch_discount=0.5,
                supports_citations=True
            )
        }
    
    def validate_job(self, job: Job) -> None:
        pass
    
    def create_batch(self, jobs: List[Job]) -> str:
        return "test-batch-id"
    
    def get_batch_status(self, batch_id: str) -> tuple[str, Optional[Dict]]:
        return "complete", None
    
    def get_batch_results(self, batch_id: str, raw_responses_dir: Optional[str] = None) -> List[JobResult]:
        return []
    
    def cancel_batch(self, batch_id: str) -> bool:
        return True
    
    def estimate_cost(self, jobs: List[Job]) -> float:
        return 0.01


class TestProvider:
    """Test base Provider functionality."""
    
    def test_parameter_validation_schema(self):
        """Test Pydantic parameter validation schemas."""
        provider = ConcreteProvider()
        
        # Get schema for valid model
        schema = provider.get_param_schema("test-model")
        
        # Valid parameters
        valid_params = schema(temperature=0.5, max_tokens=100, enable_citations=True)
        assert valid_params.temperature == 0.5
        assert valid_params.max_tokens == 100
        assert valid_params.enable_citations is True
        
        # Invalid temperature
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            schema(temperature=1.5)
        
        # Invalid max_tokens
        with pytest.raises(ValueError, match="must be positive"):
            schema(max_tokens=-100)
        
        # Citations not supported for model that doesn't support them
        provider.models["test-model"].supports_citations = False
        schema = provider.get_param_schema("test-model")
        with pytest.raises(ValueError, match="does not support citations"):
            schema(enable_citations=True)
    
    def test_model_support_checking(self):
        """Test checking model support and configuration."""
        provider = ConcreteProvider()
        
        # Supported model
        assert provider.supports_model("test-model") is True
        config = provider.get_model_config("test-model")
        assert config is not None
        assert config.name == "test-model"
        assert config.max_input_tokens == 1000
        assert config.max_output_tokens == 500
        assert config.batch_discount == 0.5
        
        # Unsupported model
        assert provider.supports_model("unknown-model") is False
        assert provider.get_model_config("unknown-model") is None
        
        # Add another model
        provider.models["test-model-2"] = ModelConfig(
            name="test-model-2",
            max_input_tokens=2000,
            max_output_tokens=1000,
            batch_discount=0.3,
            supports_citations=False
        )
        
        assert provider.supports_model("test-model-2") is True
        assert len(provider.models) == 2
    
    def test_validate_params_method(self):
        """Test the validate_params convenience method."""
        provider = ConcreteProvider()
        
        # Valid params - should not raise
        provider.validate_params("test-model", temperature=0.7, max_tokens=200)
        
        # Invalid model
        with pytest.raises(ValidationError, match="Unknown model"):
            provider.validate_params("invalid-model", temperature=0.5)
        
        # Invalid param value
        with pytest.raises(ValueError):
            provider.validate_params("test-model", temperature=2.0)
        
        # Extra params should be ignored by default Pydantic behavior
        provider.validate_params("test-model", temperature=0.5, unknown_param="test")