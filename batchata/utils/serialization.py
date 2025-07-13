"""Serialization utilities."""

from typing import Any


def to_dict(obj: Any) -> Any:
    """Convert object to dictionary recursively."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    else:
        return obj