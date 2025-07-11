"""Test nested Pydantic model validation with citations."""

import pytest
from typing import List, Optional, Union
from pydantic import BaseModel
from batchata.batch_manager import BatchManager


class SimpleModel(BaseModel):
    name: str


class DirectNestedModel(BaseModel):
    name: str
    nested: SimpleModel


class ListNestedModel(BaseModel):
    name: str
    items: List[SimpleModel]


class OptionalNestedModel(BaseModel):
    name: str
    optional_item: Optional[SimpleModel]


class UnionNestedModel(BaseModel):
    name: str
    union_item: Union[SimpleModel, str]


def test_direct_nested_model_validation():
    """Direct nested models should be rejected"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models.*contains nested model"):
        BatchManager(
            files=["test.txt"],
            prompt="test",
            model="claude-3-5-sonnet-20241022",
            response_model=DirectNestedModel,
            enable_citations=True
        )


def test_list_nested_model_validation():
    """List[BaseModel] should be rejected"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models.*contains nested model"):
        BatchManager(
            files=["test.txt"],
            prompt="test",
            model="claude-3-5-sonnet-20241022",
            response_model=ListNestedModel,
            enable_citations=True
        )


def test_optional_nested_model_validation():
    """Optional[BaseModel] should be rejected"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models.*contains nested model"):
        BatchManager(
            files=["test.txt"],
            prompt="test",
            model="claude-3-5-sonnet-20241022",
            response_model=OptionalNestedModel,
            enable_citations=True
        )


def test_union_nested_model_validation():
    """Union[BaseModel, ...] should be rejected"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models.*contains nested model"):
        BatchManager(
            files=["test.txt"],
            prompt="test",
            model="claude-3-5-sonnet-20241022",
            response_model=UnionNestedModel,
            enable_citations=True
        )


def test_flat_model_passes_validation():
    """Flat models should pass validation"""
    manager = BatchManager(
        files=["test.txt"],
        prompt="test",
        model="claude-3-5-sonnet-20241022",
        response_model=SimpleModel,
        enable_citations=True
    )
    assert manager._response_model == SimpleModel
    assert manager.batch_kwargs.get("enable_citations") == True


def test_no_citations_allows_nested_models():
    """Nested models should be allowed when citations are disabled"""
    manager = BatchManager(
        files=["test.txt"],
        prompt="test",
        model="claude-3-5-sonnet-20241022",
        response_model=ListNestedModel,
        enable_citations=False
    )
    assert manager._response_model == ListNestedModel