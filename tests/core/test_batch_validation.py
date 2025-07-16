"""Tests for batch validation with citations."""

import pytest
from pydantic import BaseModel
from typing import Optional

from batchata import Batch


class FlatInvoice(BaseModel):
    """A flat model - should work with citations."""
    invoice_number: str
    total_amount: float
    vendor: str


class Address(BaseModel):
    """Nested model."""
    street: str
    city: str


class NestedInvoice(BaseModel):
    """Model with nested field - should fail with citations."""
    invoice_number: str
    billing_address: Address  # Nested


class TestBatchCitationValidation:
    """Test early validation of citation compatibility."""
    
    def test_flat_model_with_citations_allowed(self):
        """Test that flat models work with citations."""
        batch = Batch("./results").set_state(file="./state")
        
        # Should not raise
        batch.add_job(
            messages=[{"role": "user", "content": "Extract invoice"}],
            model="claude-3-5-sonnet-20241022",
            response_model=FlatInvoice,
            enable_citations=True
        )
        
        assert len(batch.jobs) == 1
    
    def test_nested_model_with_citations_fails(self):
        """Test that nested models fail early with citations."""
        batch = Batch("./results").set_state(file="./state")
        
        with pytest.raises(ValueError) as exc_info:
            batch.add_job(
                messages=[{"role": "user", "content": "Extract invoice"}],
                model="claude-3-5-sonnet-20241022",
                response_model=NestedInvoice,
                enable_citations=True
            )
        
        assert "Citations with response_model require flat models" in str(exc_info.value)
        assert "billing_address" in str(exc_info.value)
        assert len(batch.jobs) == 0  # Job not added
    
    def test_nested_model_without_citations_allowed(self):
        """Test that nested models work when citations are disabled."""
        batch = Batch("./results").set_state(file="./state")
        
        # Should not raise when citations disabled
        batch.add_job(
            messages=[{"role": "user", "content": "Extract invoice"}],
            model="claude-3-5-sonnet-20241022",
            response_model=NestedInvoice,
            enable_citations=False  # Citations disabled
        )
        
        assert len(batch.jobs) == 1
    
    def test_no_model_with_citations_allowed(self):
        """Test that citations work without response_model."""
        batch = Batch("./results").set_state(file="./state")
        
        # Should not raise
        batch.add_job(
            messages=[{"role": "user", "content": "Extract info"}],
            model="claude-3-5-sonnet-20241022",
            response_model=None,  # No model
            enable_citations=True
        )
        
        assert len(batch.jobs) == 1