"""Utility functions for ai-batch."""

from typing import Type, Optional, get_origin, get_args
from pydantic import BaseModel


def check_flat_model_for_citation_mapping(response_model: Optional[Type[BaseModel]], enable_citations: bool) -> None:
    """
    Validate that response model is flat when citation mapping is enabled.
    Citation mapping only works with flat Pydantic models, not nested ones.
    
    Raises ValueError if nested models are used with citations enabled.
    """
    if not (response_model and enable_citations):
        return
    
    def has_nested_model(field_type: Type) -> bool:
        # Direct BaseModel check
        if (hasattr(field_type, '__mro__') and 
            BaseModel in field_type.__mro__ and 
            field_type != BaseModel):
            return True
        
        # Check generic types (List[Model], Optional[Model], etc.)
        args = get_args(field_type)
        return args and any(has_nested_model(arg) for arg in args)
    
    for field_name, field_info in response_model.model_fields.items():
        if has_nested_model(field_info.annotation):
            raise ValueError(
                f"Citation mapping requires flat Pydantic models. "
                f"Field '{field_name}' contains nested model(s). "
                f"Please flatten your model structure when using citations."
            )