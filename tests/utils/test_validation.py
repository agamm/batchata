"""Tests for validation utilities."""

import pytest
from pydantic import BaseModel
from typing import Optional, List

from batchata.utils.validation import validate_flat_model


class FlatModel(BaseModel):
    """A flat model with only primitive types."""
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []


class NestedAddress(BaseModel):
    """A nested model."""
    street: str
    city: str


class ModelWithNested(BaseModel):
    """A model with nested BaseModel field."""
    name: str
    address: NestedAddress  # This is nested


class ModelWithOptionalNested(BaseModel):
    """A model with optional nested field."""
    name: str
    address: Optional[NestedAddress] = None


class TestValidateFlatModel:
    """Test model validation for citation compatibility."""
    
    def test_flat_model_passes(self):
        """Test that flat models pass validation."""
        # Should not raise
        validate_flat_model(FlatModel)
    
    def test_nested_model_fails(self):
        """Test that nested models fail validation."""
        with pytest.raises(ValueError) as exc_info:
            validate_flat_model(ModelWithNested)
        
        assert "Citations with response_model require flat models" in str(exc_info.value)
        assert "Field 'address' is a nested model" in str(exc_info.value)
    
    def test_optional_nested_fails(self):
        """Test that optional nested models also fail."""
        with pytest.raises(ValueError) as exc_info:
            validate_flat_model(ModelWithOptionalNested)
        
        assert "Field 'address' is a nested model" in str(exc_info.value)