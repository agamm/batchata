# Remove dotenv from Core Library and Add API Key Validation

## Problem
The core library was importing and using `python-dotenv` to load environment variables, which is inappropriate for a library. Libraries should only read from `os.environ` directly - the consuming application should handle environment setup.

## Solution
1. **Removed dotenv from core library**: Core library code no longer imports or uses dotenv
2. **Added API key validation**: Library now validates that `ANTHROPIC_API_KEY` is set before initializing providers
3. **Moved dotenv to dev dependencies**: dotenv is now only available for examples and debug scripts
4. **Updated examples**: Added dotenv loading to examples that need environment setup

## Changes Made

### Core Library Changes
- Removed `from dotenv import load_dotenv` and `load_dotenv()` from `batchata/core.py`
- Added API key validation in `AnthropicBatchProvider.__init__()`:
  ```python
  api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  if not api_key:
      raise ValueError("ANTHROPIC_API_KEY environment variable is required...")
  ```

### Dependency Changes
- Removed `python-dotenv>=1.1.1` from main dependencies
- Added `python-dotenv>=1.1.1` to dev dependencies

### Examples Updates
- Added dotenv loading to examples that need it:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```
- Fixed example code to properly access `BatchResult` structure

### Testing
- Added comprehensive API key validation tests
- Tests cover missing key, empty key, whitespace-only key scenarios
- Verified examples work correctly with dotenv in dev environment

## Benefits
- **Cleaner library interface**: Library doesn't impose environment loading behavior
- **Better error messages**: Clear error when API key is missing
- **Proper separation**: Examples/debug use dotenv, core library uses os.environ
- **Library best practices**: Follows standard library design patterns

## Status
✅ **COMPLETE** - Core library is now environment-agnostic while examples retain convenience features