"""Custom JSON encoder for batchata objects."""

import json
from dataclasses import asdict, is_dataclass
from typing import Any
from pydantic import BaseModel

from ..types import Citation


class BatchataJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles batchata objects like Citation.
    
    This encoder automatically converts Citation objects and other dataclasses
    to dictionaries for JSON serialization.
    
    Usage:
        ```python
        import json
        from batchata.utils import BatchataJSONEncoder
        
        # Now JobResult objects with Citation objects can be serialized directly
        json.dump(job_result, f, cls=BatchataJSONEncoder, indent=2)
        ```
    """
    
    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        # Handle Citation objects specifically
        if isinstance(obj, Citation):
            return {
                'text': obj.text,
                'source': obj.source,
                'page': obj.page,
                'metadata': obj.metadata
            }
        
        # Handle other dataclasses
        if is_dataclass(obj):
            return asdict(obj)
        
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        
        # Let the base class handle other types
        return super().default(obj)