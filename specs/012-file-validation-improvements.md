# File Validation Improvements

**Status**: Completed  
**Version**: v0.2.7

## Problem

Users experienced confusing behavior when providing non-existent files to BatchManager:
1. Processing would start and appear to be working (showing progress)
2. Jobs would fail later during execution with unclear error reporting
3. No early validation meant wasted time and unclear failure modes

## Solution

### Early File Validation
- Added upfront file validation in both `batch()` core function and `BatchManager`
- Files are validated before any processing begins
- Validation occurs in `BatchManager._initialize_state()` during initialization
- Core `batch()` function also validates files before conversion to messages

### Error Handling
- `batch()` function raises `FileNotFoundError` for missing files  
- `BatchManager` raises `BatchManagerError` for missing files
- Both support bytes content without validation (as expected)
- Clear error messages indicate which specific file is missing

### Test Coverage
Added comprehensive tests for file validation scenarios:
- Single missing file validation
- Multiple files with one missing
- Bytes content (no validation needed)
- Both core `batch()` and `BatchManager` levels

## Implementation

### Core Changes
1. **core.py**: Added file existence validation before processing
2. **batch_manager.py**: Added early validation in `_initialize_state()`

### Files Changed
- `batchata/core.py` - Early validation loop
- `batchata/batch_manager.py` - Validation during initialization
- `tests/test_ai_batch.py` - Core function tests
- `tests/test_batch_manager.py` - BatchManager tests

## Learnings
- File validation should happen as early as possible for better UX
- Both core and manager-level validation provide defense in depth
- Clear error types help users understand what went wrong
- Bytes content needs special handling (no file paths to validate)