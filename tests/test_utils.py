"""Test utility functions."""

import pytest
from typing import List, Optional, Union
from pydantic import BaseModel
from src.utils import check_flat_model_for_citation_mapping


class FlatModel(BaseModel):
    name: str
    count: int


class NestedModel(BaseModel):
    name: str
    nested: FlatModel


class ListModel(BaseModel):
    name: str
    items: List[FlatModel]


class OptionalModel(BaseModel):
    name: str
    optional_item: Optional[FlatModel]


class UnionModel(BaseModel):
    name: str
    union_item: Union[FlatModel, str]


def test_flat_model_with_citations_passes():
    """Flat models should pass validation with citations enabled"""
    check_flat_model_for_citation_mapping(FlatModel, True)  # Should not raise


def test_flat_model_without_citations_passes():
    """Flat models should pass validation with citations disabled"""
    check_flat_model_for_citation_mapping(FlatModel, False)  # Should not raise


def test_nested_model_without_citations_passes():
    """Nested models should pass validation when citations are disabled"""
    check_flat_model_for_citation_mapping(NestedModel, False)  # Should not raise


def test_no_model_passes():
    """No model should pass validation"""
    check_flat_model_for_citation_mapping(None, True)  # Should not raise


def test_direct_nested_model_with_citations_fails():
    """Direct nested models should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(NestedModel, True)


def test_list_nested_model_with_citations_fails():
    """List[BaseModel] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(ListModel, True)


def test_optional_nested_model_with_citations_fails():
    """Optional[BaseModel] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(OptionalModel, True)


def test_union_nested_model_with_citations_fails():
    """Union[BaseModel, ...] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(UnionModel, True)


def test_error_message_contains_field_name():
    """Error message should contain the problematic field name"""
    with pytest.raises(ValueError, match="Field 'nested' contains nested model"):
        check_flat_model_for_citation_mapping(NestedModel, True)
    
    with pytest.raises(ValueError, match="Field 'items' contains nested model"):
        check_flat_model_for_citation_mapping(ListModel, True)