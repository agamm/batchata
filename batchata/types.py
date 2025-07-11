"""
Type definitions for bachata.
"""

from typing import Any, Dict, List, Union, TypedDict


class BatchResult(TypedDict):
    """Unified result structure for batch processing.
    
    Fields:
        result: The parsed response model instance or plain text
        citations: Field citations dict, citation list, or None if not enabled
    """
    result: Any  # Can be BaseModel instance, dict, or str
    citations: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], None]