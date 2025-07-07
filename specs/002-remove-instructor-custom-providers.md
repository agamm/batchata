# Remove instructor dependency and add simple provider system

## Goal
Replace instructor dependency with simple provider-specific helper functions to enable easy addition of new AI providers while maintaining current functionality. Move code to src/ directory for better organization.

## Final State
- Remove instructor dependency completely
- Code organized in src/ directory structure
- Simple provider system with 2 functions per provider
- Support for multiple providers (starting with Anthropic)
- Maintained functionality with cleaner, lighter codebase

## Requirements
- [x] Move ai_batch.py to src/ai_batch.py with proper package structure
- [x] Update all imports in examples/ and tests/
- [x] Create src/providers/ directory structure
- [x] Create src/providers/anthropic.py with provider functions:
  - `prepare_request(response_model, **kwargs)` - adds system message with JSON schema
  - `parse_response(results, response_model)` - parses batch results
- [x] Replace instructor.handle_response_model() call in main batch() function
- [x] Remove instructor import and dependency from pyproject.toml
- [x] Update tests to remove instructor mocking
- [x] Verify functionality unchanged with tests

## Progress
### Completed ✅
**Project Structure:**
- Created SPEC.md following proper process
- Moved ai_batch.py to src/ai_batch.py
- Created src/__init__.py package structure
- Updated imports in examples/spam_detection.py
- Updated imports in tests/test_ai_batch.py
- Updated imports in tests/e2e/test_batch_integration.py
- Verified import structure works with `uv run`

**Provider System:**
- Created src/providers/ directory structure
- Created src/providers/__init__.py with provider imports
- Created src/providers/anthropic.py with prepare_request() and parse_response() functions

**Remove Instructor Dependency:**
- Replaced instructor.handle_response_model() call in main batch() function
- Removed instructor import and dependency from pyproject.toml
- Updated tests to remove instructor mocking
- Verified all functionality unchanged

**Testing:**
- All 7 unit tests passing (no API calls, fast)
- All 1 E2E test passing (real API calls)
- Example spam detection working correctly

### Phase 1 Complete ✅
✅ **Instructor dependency completely removed**
✅ **Basic provider system implemented**
✅ **All functionality preserved**
✅ **Tests updated and passing**

### Phase 2: Enhanced Provider Classes 🚧
**Next improvements:**
- [ ] Replace provider functions with robust classes
- [ ] Add message/size validation per provider limits
- [ ] Implement rate limiting awareness
- [ ] Create base class for common interface
- [ ] Simplify main ai_batch.py module
- [ ] Remove unnecessary __init__.py files

## Implementation Details
Current instructor usage (lines 65-68 in ai_batch.py):
```python
_, kwargs = instructor.handle_response_model(
    response_model=response_model, 
    mode=instructor.Mode.ANTHROPIC_JSON
)
```

This only generates JSON schema and creates system message:
```
As a genius expert, your task is to understand the content and provide
the parsed objects in json that match the following json_schema:
{json_schema}

Make sure to return an instance of the JSON, not the schema itself
```

## Provider System Design
Each provider needs just 2 simple functions:

```python
def prepare_anthropic_request(response_model, **kwargs):
    """Add JSON schema system message for Anthropic"""
    import json
    schema = response_model.model_json_schema()
    system_message = f"""As a genius expert, your task is to understand the content and provide
the parsed objects in json that match the following json_schema:

{json.dumps(schema, indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself"""
    
    return {"system": system_message, **kwargs}

def parse_anthropic_response(results, response_model):
    """Parse Anthropic batch results into Pydantic models"""
    # Move existing parsing logic from lines 118-147
    pass
```

Future providers (OpenAI, etc.) can be added by implementing similar functions.

## Project Structure for Multi-Provider Support

### Selected Approach: Provider Modules
```
src/
├── __init__.py
├── ai_batch.py          # Main batch() function
└── providers/
    ├── __init__.py
    ├── anthropic.py     # prepare_request() + parse_response()
    └── openai.py        # prepare_request() + parse_response() (future)
```

**Benefits**: 
- Clean separation of concerns
- Easy to add new providers
- Better organization for future scaling
- Each provider is self-contained

## Enhanced Provider Classes

**Changes:**
- Replace functions with provider classes  
- Add validation: 100k requests max, 256MB total size (per Anthropic docs)
- Rate limits configurable by tier (not hardcoded)
- Base class for consistent interface

**Structure:**
```
src/
├── ai_batch.py              # Simplified coordinator  
└── providers/
    ├── base.py              # BaseBatchProvider abstract class
    └── anthropic.py         # AnthropicBatchProvider with validation
```

**Provider Interface:**
```python
class AnthropicBatchProvider(BaseBatchProvider):
    def __init__(self, api_key: str, rate_limits: dict = None):
        self.api_key = api_key
        self.rate_limits = rate_limits or self.get_default_rate_limits()
        # Store as class attributes for batch creation logic
        
    def validate_batch(self, messages: List[List[dict]]) -> None:
        # Check: len(messages) <= 100_000
        # Check: total size <= 256MB
        # Check: respects rate_limits for intelligent batching
        
    def prepare_batch_requests(self, messages, response_model, **kwargs) -> List[dict]:
        # Return ready-to-submit requests with JSON schema
        # Can use self.rate_limits for optimal batch sizing
```

**Cleanup:**
- Remove unnecessary `__init__.py` files (examples/, tests/e2e/)
- Simplify imports

## Tests
All existing tests should continue to pass:
- Unit tests (7 tests) - update to remove instructor mocking
- E2E tests (1 test) - should work unchanged

## URL References
- [Instructor source research](https://github.com/567-labs/instructor/blob/main/instructor/process_response.py) - Confirmed minimal functionality
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) - Future provider reference

## Key Decisions
- Keep existing API unchanged - only internal implementation changes
- Start with Anthropic only, design for easy provider addition
- Maintain all current functionality and tests
- Move to src/ directory for better organization