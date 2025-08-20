"""Citation mapping utilities for Anthropic responses.

Uses value-based reverse mapping: instead of parsing text patterns,
we search for known field values in citations and check field relevance.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import replace
from pydantic import BaseModel

from ...types import Citation
from ...utils import get_logger


logger = get_logger(__name__)


# Confidence scoring thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.3
MISSING_FIELDS_RATIO_THRESHOLD = 0.5

# Field matching parameters
MIN_FIELD_WORD_LENGTH = 2
MIN_FUZZY_MATCH_LENGTH = 3
FUZZY_EDIT_DISTANCE_THRESHOLD = 1

# Pre-compiled regex patterns for performance
_CITATION_WORDS_PATTERN = re.compile(rf'\b\w{{{MIN_FUZZY_MATCH_LENGTH},}}\b')
_MULTILINE_FLAGS = re.MULTILINE
_MULTILINE_IGNORECASE_FLAGS = re.MULTILINE | re.IGNORECASE


def map_citations_to_fields(
    citation_blocks: List[Tuple[str, Citation]], 
    parsed_response: BaseModel,
) -> Tuple[Dict[str, List[Citation]], Optional[str]]:
    """Map citations to model fields using systematic value-first approach.
    
    Uses systematic fallback algorithm:
    1. Find all citations containing field value
    2. Score citations by field name match quality (exact → partial → value-only)  
    3. Return mappings with confidence indicators
    
    Args:
        citation_blocks: List of (block_text, citation) tuples
        parsed_response: The parsed Pydantic model with field values
        
    Returns:
        Tuple of:
        - Dict mapping field names to lists of Citation objects with confidence fields populated
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
        
        # Step 1: Find ALL citations containing this field's value
        value_citations = _find_citations_with_value(citation_blocks, field_value)
        
        if not value_citations:
            unmapped_fields.append(field_name)
            continue
            
        # Step 2: Score citations by field name match quality
        scored_citations = []
        for citation_text, citation in value_citations:
            field_score = _calculate_field_match_score(citation_text, field_name)
            scored_citations.append((citation_text, citation, field_score))
        
        # Step 3: Apply systematic fallback with confidence scoring
        field_mappings[field_name] = []
        
        # Try exact field match first (score >= HIGH_CONFIDENCE_THRESHOLD)
        high_confidence = [(text, cit, score) for text, cit, score in scored_citations if score >= HIGH_CONFIDENCE_THRESHOLD]
        if high_confidence:
            for _, citation, score in high_confidence:
                citation_copy = replace(citation,
                    confidence="high",
                    match_reason=f"exact field match (score: {score:.2f})"
                )
                field_mappings[field_name].append(citation_copy)
            continue
            
        # Try partial field match (score >= MEDIUM_CONFIDENCE_THRESHOLD)  
        medium_confidence = [(text, cit, score) for text, cit, score in scored_citations if score >= MEDIUM_CONFIDENCE_THRESHOLD]
        if medium_confidence:
            for _, citation, score in medium_confidence:
                citation_copy = replace(citation,
                    confidence="medium",
                    match_reason=f"partial field match (score: {score:.2f})"
                )
                field_mappings[field_name].append(citation_copy)
            continue
            
        # Fall back to value-only match with low confidence
        for _, citation, score in scored_citations:
            citation_copy = replace(citation,
                confidence="low",
                match_reason=f"value-only match (score: {score:.2f})"
            )
            field_mappings[field_name].append(citation_copy)
    
    # Generate warning if many fields unmapped
    warning = None
    total_mappable_fields = len([v for v in parsed_response.model_dump().values() 
                                if not _should_skip_field(v)])
    if unmapped_fields and len(unmapped_fields) > total_mappable_fields * MISSING_FIELDS_RATIO_THRESHOLD:
        warning = f"Could not find citations for: {', '.join(unmapped_fields)}"
    
    return field_mappings, warning


def _should_skip_field(field_value: Any) -> bool:
    """Check if a field value should be skipped for citation mapping."""
    # Skip None, empty strings, and boolean values
    if field_value is None or field_value == "":
        return True
    
    # Skip complex types (lists, dicts) - not supported in flat models
    if isinstance(field_value, (list, dict)):
        return True
    
    return False


def _get_value_variants(value: Any) -> List[str]:
    """Get all reasonable string representations of a value for searching.
    
    Returns ordered list with exact matches first, then variants.
    """
    variants = []
    
    if isinstance(value, (int, float)):
        # Numeric value variants - exact first, then formatted variants
        num = float(value)
        if num == int(num):  # Whole number
            int_val = int(num)
            # Start with exact string representation
            variants.append(str(int_val))
            # Add formatted variants
            variants.extend([
                f"${int_val}",
                f"{int_val}.00", 
                f"${int_val}.00",
                f"{int_val:,}",  # Comma formatting: 292,585
                f"${int_val:,}",  # Dollar with comma: $292,585
            ])
        else:
            # For floats, preserve original precision first
            str_val = str(num)
            variants.append(str_val)  # Original precision: 0.917 (exact first)
            # Add formatted variants
            variants.extend([
                f"{num:.2f}",  # 2 decimal: 0.92
                f"{num:.3f}",  # 3 decimal: 0.917
                f"${num:.2f}",
                f"${num:.3f}",
            ])
    elif isinstance(value, str):
        # String value variants - exact case first, then lowercase
        value = value.strip()
        if value:  # Non-empty string
            variants.append(value)  # Original case (exact first)
            if value.lower() != value:  # Only add lowercase if different
                variants.append(value.lower())
            
            # Handle quoted values
            if value.startswith('"') and value.endswith('"'):
                unquoted = value[1:-1]
                variants.append(unquoted)  # Exact unquoted
                if unquoted.lower() != unquoted:
                    variants.append(unquoted.lower())
            elif value.startswith("'") and value.endswith("'"):
                unquoted = value[1:-1]
                variants.append(unquoted)  # Exact unquoted
                if unquoted.lower() != unquoted:
                    variants.append(unquoted.lower())
    elif hasattr(value, 'year') and hasattr(value, 'month') and hasattr(value, 'day'):
        # Date and datetime objects - start with ISO format (most precise)
        variants.append(value.strftime('%Y-%m-%d'))  # ISO format first
        
        # Natural format with full month name
        variants.extend([
            value.strftime('%B %d, %Y'),  # "October 20, 2023"
            value.strftime('%B %d %Y'),   # "October 20 2023"
        ])
        
        # Abbreviated month
        variants.extend([
            value.strftime('%b %d, %Y'),  # "Oct 20, 2023"
            value.strftime('%b %d %Y'),   # "Oct 20 2023"
        ])
        
        # Numeric formats
        variants.extend([
            value.strftime('%m/%d/%Y'),   # "10/20/2023"
            value.strftime('%d/%m/%Y'),   # "20/10/2023"
            value.strftime('%m-%d-%Y'),   # "10-20-2023"
            value.strftime('%d-%m-%Y'),   # "20-10-2023"
        ])
        
        # Compact formats
        variants.extend([
            value.strftime('%m/%d/%y'),   # "10/20/23"
            value.strftime('%d/%m/%y'),   # "20/10/23"
        ])
    
    return variants


def _contains_value(citation_text: str, value_variants: List[str]) -> bool:
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


def _find_citations_with_value(citation_blocks: List[Tuple[str, Citation]], field_value: Any) -> List[Tuple[str, Citation]]:
    """Find all citations containing the field value (any variant).
    
    Args:
        citation_blocks: List of (block_text, citation) tuples
        field_value: The field value to search for
        
    Returns:
        List of (block_text, citation) tuples that contain the value
    """
    value_variants = _get_value_variants(field_value)
    matching_citations = []
    
    for citation_text, citation in citation_blocks:
        if _contains_value(citation_text, value_variants):
            matching_citations.append((citation_text, citation))
    
    return matching_citations


def _calculate_field_match_score(citation_text: str, field_name: str) -> float:
    """Calculate field name match score from 0.0 to 1.0.
    
    Uses fuzzy matching and considers field patterns in citation context.
    
    Args:
        citation_text: The citation block text (with N-1, N+1 context)
        field_name: The field name to match against
        
    Returns:
        Float score from 0.0 (no match) to 1.0 (perfect match)
    """
    citation_lower = citation_text.lower()
    field_words = [word for word in field_name.lower().split('_') if len(word) > MIN_FIELD_WORD_LENGTH]
    
    if not field_words:
        return 0.5  # Default score for fields with no meaningful words
    
    # Check for exact field pattern matches first
    pattern_score = _check_field_patterns(citation_lower, field_name, field_words)
    if pattern_score > 0:
        return pattern_score
    
    # Fall back to fuzzy word matching
    return _calculate_fuzzy_word_score(citation_lower, field_words)


def _check_field_patterns(citation_lower: str, field_name: str, field_words: List[str]) -> float:
    """Check for structured field patterns in citation text.
    
    Returns:
        1.0 for markdown patterns, 0.9 for non-markdown patterns, 0.0 for no match
    """
    field_words_joined = " ".join(field_words)
    field_name_readable = field_name.replace("_", " ")
    
    # Check markdown patterns first (highest confidence)
    if _check_markdown_patterns(citation_lower, field_name, field_words_joined):
        return 1.0
    
    # Check non-markdown patterns (high confidence)
    if _check_non_markdown_patterns(citation_lower, field_words_joined, field_name_readable):
        return 0.9
    
    return 0.0


def _check_markdown_patterns(citation_lower: str, field_name: str, field_words_joined: str) -> bool:
    """Check for markdown field patterns like **field**: value."""
    # Dynamic patterns that need field-specific escaping
    markdown_patterns = [
        rf'\*\*[^*]*{re.escape(field_name.replace("_", "[\\s_]"))}\*\*\s*:',  # **field_name**:
        rf'\*\*[^*]*{re.escape(field_words_joined)}\*\*\s*:',  # **field words**:
        rf'^\\s*-\\s*\*\*[^*]*{re.escape(field_words_joined)}\*\*',  # - **field words**
    ]
    
    for pattern in markdown_patterns:
        if re.search(pattern, citation_lower, _MULTILINE_FLAGS):
            return True
    
    return False


def _check_non_markdown_patterns(citation_lower: str, field_words_joined: str, field_name_readable: str) -> bool:
    """Check for non-markdown field patterns like field: value."""
    # Dynamic patterns that need field-specific escaping
    non_markdown_patterns = [
        rf'\b{re.escape(field_words_joined)}\s*:\s*',  # Field words: value
        rf'\b{re.escape(field_name_readable)}\s*:\s*',  # Field name: value
        rf'^{re.escape(field_words_joined)}\s*-\s*',  # Field words - value (at line start)
        rf'\b{re.escape(field_words_joined)}\s+(?:is|are|was|were)\s+',  # Field words is/are value
    ]
    
    for pattern in non_markdown_patterns:
        if re.search(pattern, citation_lower, _MULTILINE_IGNORECASE_FLAGS):
            return True
    
    return False


def _calculate_fuzzy_word_score(citation_lower: str, field_words: List[str]) -> float:
    """Calculate score based on fuzzy word matching.
    
    Returns:
        Ratio of matched words to total words (0.0 to 1.0)
    """
    matched_words = 0
    total_words = len(field_words)
    
    for field_word in field_words:
        # Direct match
        if field_word in citation_lower:
            matched_words += 1
            continue
            
        # Fuzzy match with edit distance
        citation_words = _CITATION_WORDS_PATTERN.findall(citation_lower)
        found_fuzzy = False
        for citation_word in citation_words:
            if _levenshtein_distance(citation_word, field_word) <= FUZZY_EDIT_DISTANCE_THRESHOLD:
                matched_words += 1
                found_fuzzy = True
                break
        
        if found_fuzzy:
            continue
                
        # Common word transformations
        if _fuzzy_word_match(citation_lower, field_word):
            matched_words += 1
    
    return matched_words / total_words if matched_words > 0 else 0.0


def _fuzzy_word_match(text: str, word: str) -> bool:
    """Check for common word transformations and partial matches.
    
    Args:
        text: Text to search in
        word: Word to find matches for
        
    Returns:
        True if fuzzy match found
    """
    # Common transformations (ensure minimum length of 3 for meaningful matches)
    transformations = []
    
    # Plural forms
    transformations.append(word + 's')      # tax → taxes
    transformations.append(word + 'es')     # story → stories
    
    # Singular forms (only if result is at least 3 chars)
    if len(word) > 3:
        transformations.append(word[:-1])   # taxes → tax
    if len(word) > 4:
        transformations.append(word[:-2])   # stories → story
    
    # Underscore/hyphen variants
    if '_' in word:
        transformations.append(word.replace('_', ' '))  # space variant
        transformations.append(word.replace('_', '-'))  # hyphen variant
    
    # Partial matches for compound words
    compound_matches = [
        f"{word}ing",    # building → buildings
        f"{word}ed",     # assess → assessed
    ]
    
    # Special case for words ending in 'ies'
    if word.endswith('ies') and len(word) > 4:
        compound_matches.append(word[:-3] + 'y')  # stories → story
    
    all_variants = transformations + compound_matches
    
    for variant in all_variants:
        # Only match variants that are at least MIN_FUZZY_MATCH_LENGTH characters
        if len(variant) >= MIN_FUZZY_MATCH_LENGTH and variant in text:
            return True
            
    return False