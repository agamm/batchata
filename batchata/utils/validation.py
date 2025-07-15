"""Validation utilities."""

from typing import Type
from pydantic import BaseModel


def validate_flat_model(model: Type[BaseModel]) -> None:
    """Validate that a Pydantic model is flat (no nested BaseModel fields).
    
    Args:
        model: The Pydantic model class to validate
        
    Raises:
        ValueError: If model contains nested BaseModel fields
    """
    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        
        # Handle Optional, List, etc.
        if hasattr(field_type, '__origin__'):
            # Get the first type arg (e.g., for Optional[X], get X)
            args = getattr(field_type, '__args__', [])
            if args:
                field_type = args[0]
        
        # Check if field is a BaseModel subclass
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            raise ValueError(
                f"Citations with response_model require flat models. "
                f"Field '{field_name}' is a nested model. "
                f"Consider flattening your model or disabling citations."
            )