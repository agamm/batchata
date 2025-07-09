#!/usr/bin/env python3
"""Test script to verify nested model validation works."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import batch_files
from tests.utils import create_pdf
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str


class PersonWithAddress(BaseModel):
    name: str
    age: int
    address: Address  # Nested model


class PersonFlat(BaseModel):
    name: str
    age: int
    street: str
    city: str


def test_nested_model_error():
    """Test that nested models with citations raise an error."""
    print("Testing nested model with citations (should fail)...")
    
    test_pdf = create_pdf(["Test document with person information"])
    
    try:
        job = batch_files(
            files=[test_pdf],
            prompt="Extract person information",
            model="claude-3-haiku-20240307",
            response_model=PersonWithAddress,
            enable_citations=True
        )
        print("❌ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✅ Got expected error: {e}")
        assert "Citations are only supported" in str(e)
        assert "flat Pydantic models" in str(e)
        assert "address" in str(e)  # Should mention the nested field


def test_flat_model_works():
    """Test that flat models work fine."""
    print("\nTesting flat model with citations (should work)...")
    
    test_pdf = create_pdf(["Test document with person information"])
    
    try:
        job = batch_files(
            files=[test_pdf],
            prompt="Extract person information",
            model="claude-3-haiku-20240307",
            response_model=PersonFlat,
            enable_citations=True
        )
        print(f"✅ Batch created successfully: {job._batch_id}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def test_nested_without_citations():
    """Test that nested models work without citations."""
    print("\nTesting nested model without citations (should work)...")
    
    test_pdf = create_pdf(["Test document with person information"])
    
    try:
        job = batch_files(
            files=[test_pdf],
            prompt="Extract person information",
            model="claude-3-haiku-20240307",
            response_model=PersonWithAddress,
            enable_citations=False  # No citations = should work
        )
        print(f"✅ Batch created successfully: {job._batch_id}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    test_nested_model_error()
    test_flat_model_works()
    test_nested_without_citations()
    print("\n🎉 All validation tests passed!")