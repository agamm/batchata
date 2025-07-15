"""Citation mapping utilities for Anthropic responses.

Uses value-based reverse mapping: instead of parsing text patterns,
we search for known field values in citations and check field relevance.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from ...types import Citation
from ...utils import get_logger


logger = get_logger(__name__)


def map_citations_to_fields(
    citation_blocks: List[Tuple[str, Citation]], 
    parsed_response: BaseModel,
) -> Tuple[Dict[str, List[Citation]], Optional[str]]:
    """Map citations to model fields using value-based approach.
    
    Works backwards from known values in the parsed result. If a field's value
    exists in a citation and field-related words appear nearby, it's mapped.
    
    Args:
        citation_blocks: List of (block_text, citation) tuples
        parsed_response: The parsed Pydantic model with field values
        
    Returns:
        Tuple of:
        - Dict mapping field names to lists of citations
        - Optional warning message if many fields couldn't be mapped
    """
    if not citation_blocks or not parsed_response:
        return {}, None
    
    field_mappings = {}
    unmapped_fields = []
    
    # For each field and its value in the parsed response
    for field_name, field_value in parsed_response.model_dump().items():
        if _should_skip_field(field_value):
            continue
            
        mapped = False
        value_variants = _get_value_variants(field_value)
        
        # Check each citation for this field's value
        for citation_text, citation in citation_blocks:
            if _contains_value(citation_text, value_variants):
                if _is_field_relevant(citation_text, field_name, field_value):
                    if field_name not in field_mappings:
                        field_mappings[field_name] = []
                    # Avoid duplicate citations for the same field
                    if citation not in field_mappings[field_name]:
                        field_mappings[field_name].append(citation)
                    mapped = True
        
        if not mapped:
            unmapped_fields.append(field_name)
    
    # Generate warning if many fields unmapped
    warning = None
    total_mappable_fields = len([v for v in parsed_response.model_dump().values() 
                                if not _should_skip_field(v)])
    if unmapped_fields and len(unmapped_fields) > total_mappable_fields * 0.5:
        warning = f"Could not find citations for: {', '.join(unmapped_fields)}"
    
    return field_mappings, warning


def _should_skip_field(field_value: Any) -> bool:
    """Check if a field value should be skipped for citation mapping."""
    # Skip None, empty strings, and boolean values
    if field_value is None or field_value == "":
        return True
    
    # Skip boolean values - too risky (true/false appear everywhere)
    if isinstance(field_value, bool):
        return True
    
    # Skip complex types (lists, dicts) - not supported in flat models
    if isinstance(field_value, (list, dict)):
        return True
    
    return False


def _get_value_variants(value: Any) -> Set[str]:
    """Get all reasonable string representations of a value for searching."""
    variants = set()
    
    if isinstance(value, (int, float)):
        # Numeric value variants
        num = float(value)
        if num == int(num):  # Whole number
            variants.update([
                str(int(num)),
                f"${int(num)}",
                f"{int(num)}.00", 
                f"${int(num)}.00",
            ])
        else:
            variants.update([
                f"{num:.2f}",
                f"${num:.2f}",
            ])
    elif isinstance(value, str):
        # String value variants
        value = value.strip()
        if value:  # Non-empty string
            variants.add(value.lower())
            variants.add(value)  # Original case
            
            # Handle quoted values
            if value.startswith('"') and value.endswith('"'):
                unquoted = value[1:-1]
                variants.add(unquoted.lower())
                variants.add(unquoted)
            elif value.startswith("'") and value.endswith("'"):
                unquoted = value[1:-1]
                variants.add(unquoted.lower())
                variants.add(unquoted)
    
    return variants


def _contains_value(citation_text: str, value_variants: Set[str]) -> bool:
    """Check if any value variant exists in the citation text."""
    text_lower = citation_text.lower()
    
    for variant in value_variants:
        if variant.lower() in text_lower:
            return True
        # Also check for quoted versions
        if f'"{variant.lower()}"' in text_lower:
            return True
        if f"'{variant.lower()}'" in text_lower:
            return True
    
    return False


def _is_field_relevant(citation_text: str, field_name: str, field_value: Any) -> bool:
    """Check if field name is relevant to the citation containing the value.
    
    Uses a window around the value to check if field-related words appear nearby.
    For exact value matches, requires at least one field word.
    For fuzzy value matches, requires all field words.
    """
    citation_lower = citation_text.lower()
    value_variants = _get_value_variants(field_value)
    
    # Find where the value appears and check if it's an exact match
    value_position = None
    is_exact_match = False
    
    for variant in value_variants:
        pos = citation_lower.find(variant.lower())
        if pos != -1:
            value_position = pos
            # Check if this is the original value (exact match)
            if variant.lower() == str(field_value).lower():
                is_exact_match = True
            break
    
    if value_position is None:
        return False
    
    # Create a window around the value (50 chars before and after)
    window_size = 50
    start_pos = max(0, value_position - window_size)
    end_pos = min(len(citation_lower), value_position + window_size)
    text_window = citation_lower[start_pos:end_pos]
    
    # Get field words (split on underscores)
    field_words = [word for word in field_name.lower().split('_') if len(word) > 2]
    if not field_words:
        return True  # No words to check
    
    matched_words = 0
    
    # Check for direct word matches
    for field_word in field_words:
        if field_word in text_window:
            matched_words += 1
    
    # For exact value matches, just need at least one field word
    if is_exact_match and matched_words > 0:
        return True
    
    # For non-exact matches, check fuzzy matching and require all words
    if matched_words < len(field_words):
        window_words = re.findall(r'\b\w{3,}\b', text_window)
        for field_word in field_words:
            if field_word in text_window:
                continue  # Already matched
            # Check for fuzzy match
            for window_word in window_words:
                if _levenshtein_distance(window_word, field_word) <= 1:
                    matched_words += 1
                    break
    
    # For non-exact matches, require all field words
    return matched_words == len(field_words)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]