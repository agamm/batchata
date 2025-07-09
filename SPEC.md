# combine batch() funcs

## Goal
Combine the separate `batch()` and `batch_files()` functions into a single unified `batch()` function with two mutually exclusive parameters: `messages` or `files`. This will simplify the codebase, tests, and README.

## Current state
- `batch()` function in `src/core.py` processes message conversations
- `batch_files()` function in `src/file_processing.py` processes PDF files
- Both functions return `BatchJob` instances with similar interfaces
- Users need to know which function to use based on their input type
- Code duplication exists between the two functions

## Requirements
- [x] Create unified `batch()` function with two params: `messages` and `files` (either/or)
- [x] Remove `batch_files()` function entirely (no backwards compatibility needed)
- [x] `messages` param accepts list of message conversations
- [x] `files` param accepts list of string file paths, bytes, or Path objects
- [x] Add validation to ensure only one of `messages` or `files` is provided
- [x] When `files` is used, require `prompt` parameter
- [x] Simplify codebase by removing duplicate logic
- [x] Update all tests to use unified function
- [x] Update README and examples to use unified function

## Progress
### Completed
- ✅ Written comprehensive tests for unified batch() function behavior
- ✅ Implemented unified batch() function with either/or params validation
- ✅ Removed batch_files() function entirely from file_processing.py
- ✅ Moved pdf_to_document_block() to core.py
- ✅ Updated all existing tests to use unified batch() function
- ✅ Updated README.md with unified examples and interface
- ✅ Updated all example scripts to use unified batch() function  
- ✅ Updated __init__.py exports to remove batch_files
- ✅ All tests passing (59 tests total)

### Implementation Details
- Unified function handles both messages and files in single entry point
- Proper validation ensures only one parameter type is provided
- Files parameter supports string paths, Path objects, and bytes
- Prompt parameter is required when using files
- Citations work consistently in both modes
- Error messages are clear for invalid parameter combinations

## Interface Design
```python
def batch(
    messages: Optional[List[List[dict]]] = None,
    files: Optional[Union[List[str], List[Path], List[bytes]]] = None,
    model: str,
    prompt: Optional[str] = None,  # Required when files is provided
    response_model: Optional[Type[T]] = None,
    enable_citations: bool = False,
    provider: str = "anthropic",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    verbose: bool = False,
    raw_results_dir: Optional[str] = None
) -> BatchJob:
    """
    Process conversations or files using batch API.
    
    Either messages OR files must be provided, not both.
    When using files, prompt is required.
    """
```

## Tests
### Tests completed
- ✅ Test batch() with messages param (existing message behavior)
- ✅ Test batch() with files param + prompt (existing batch_files behavior)
- ✅ Test batch() with both messages and files raises ValueError
- ✅ Test batch() with neither messages nor files raises ValueError
- ✅ Test batch() with files but no prompt raises ValueError
- ✅ Test file types: string paths, Path objects, bytes
- ✅ Test empty lists for both params
- ✅ Test citations work correctly for both modes

### Tests passing
- All 59 tests passing
- Created comprehensive test suite in tests/test_unified_batch.py
- Updated existing tests to use unified function
- Validation tests ensure proper error handling

## URL References
- [Unified batch() function](src/core.py) - Handles both messages and files
- [Test suite](tests/test_unified_batch.py) - Comprehensive validation tests
- [Updated examples](examples/) - All examples use unified interface

## Learnings
- Both functions share most parameters (model, response_model, provider, etc.)
- Main difference is input format: messages vs files + prompt
- File processing converts files to document blocks then creates messages
- Citations behavior is consistent between the two modes
- Single entry point reduces API confusion and simplifies documentation
- Validation with clear error messages prevents common usage mistakes
- Moving pdf_to_document_block to core.py eliminated file_processing.py entirely
- Type hints with Union types provide better IDE support for either/or parameters
- Comprehensive test coverage ensures all edge cases are handled properly

## Notes
- No backwards compatibility needed - clean break
- Update all imports from `batch_files` to `batch`
- Remove `batch_files` from __all__ exports
- Simplify README by showing unified examples
- Consider @overload decorators for better IDE support
- Clear error messages for invalid parameter combinations

## Status: COMPLETE ✅
This specification has been fully implemented and tested. All requirements have been met:
- Unified batch() function successfully combines both message and file processing modes
- Clean API with proper validation and error handling
- All tests passing and examples updated
- Documentation updated to reflect unified interface
- Codebase simplified by removing duplicate logic