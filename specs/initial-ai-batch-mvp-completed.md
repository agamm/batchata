# Initial ai-batch mvp

## Goal
Build a minimal `batch()` function that processes multiple messages through Anthropic's Message Batches API, using instructor for Pydantic model handling. Create a simple wrapper around instructor to enable batch processing with Claude models.

## Final State
- Complete ai-batch module with batch() function
- pytest configured and working with comprehensive test suite
- uv for dependency management
- python-dotenv for environment variable handling
- Real Anthropic Message Batches API integration
- Working examples and documentation

## Requirements
- [x] Core batch() function that accepts messages, model, and response_model
- [x] Integration with instructor library for structured output
- [x] Support for Anthropic Claude models via Message Batches API
- [x] Environment variable handling for API keys using python-dotenv
- [x] Pydantic model validation for responses
- [x] Error handling for API failures and invalid inputs
- [x] Example spam detection implementation
- [x] Real batch processing using Anthropic's batch API (not just sequential calls)
- [x] Simplified implementation without system parameter - users handle context in messages

## Progress
### Completed ✅
- Project structure setup
- Basic testing framework
- Environment variable handling setup
- Dependencies added (instructor, anthropic, python-dotenv)
- Core ai_batch.py module implemented with real batch processing
- Comprehensive test suite written and passing (7 unit tests, 3 E2E tests)
- Spam detection example created and working
- .env.example file for API key setup
- Package installable with `uv pip install -e .`
- Real Anthropic Message Batches API integration
- Examples runnable without path issues
- Simplified implementation without system parameter
- All tests updated and passing

### Implementation Details
- Direct integration with Anthropic's Message Batches API
- Uses instructor for structured output schema generation
- Submits batches to Anthropic's batches API with proper request format
- Waits for batch completion with configurable polling
- Manual JSON parsing from batch results
- Users include context/instructions directly in their message content
- No temporary files or complex system message handling

## Tests
### Unit Tests (tests/test_ai_batch.py)
Focus on functionality around AI calls - no actual API calls, using mocks:
- [x] test_batch_empty_messages() - Test empty input handling
- [x] test_batch_invalid_model() - Test model validation
- [x] test_batch_missing_required_params() - Test parameter validation
- [x] test_valid_models_list() - Test model list validation
- [x] test_missing_api_key() - Test API key requirement
- [x] test_instructor_client_creation() - Test client setup with mocks
- [x] test_multiple_messages_processing() - Test batch processing logic with mocks

### E2E Tests (tests/e2e/test_batch_integration.py)
Happy path only with real API calls:
- [x] test_spam_detection_happy_path() - Real spam detection
- [x] test_sentiment_analysis_happy_path() - Real sentiment analysis
- [x] test_single_message_happy_path() - Single message processing

### Tests passing
- Unit tests: 7/7 ✅ (no API calls, fast)
- E2E tests: 3/3 ✅ (real API calls, slower)

## URL References
Links to external resources, documentation, or codebases that help understand the requirements:
- [Instructor Documentation](https://python.useinstructor.com/) - Main documentation for structured output with Pydantic models
- [Instructor Batch Implementation](https://github.com/567-labs/instructor/blob/main/instructor/batch.py) - Source code for batch processing functionality
- [Anthropic Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) - Official documentation for Anthropic's batch processing
- [Instructor Batch CLI](https://raw.githubusercontent.com/567-labs/instructor/a54cccd522507f34dbb5ab8be484d68f60a012d9/docs/cli/batch.md) - CLI documentation for batch operations

## Learnings
- Anthropic's batch API requires specific request format with custom_id and params structure
- System messages must be handled differently - moved to user message content instead
- instructor's handle_response_model() provides necessary schema for structured output
- Manual JSON extraction needed from batch results (responses may contain extra text)
- Simplified approach (no temp files) works better than complex BatchJob implementation

## Problems Encountered
- Initial AttributeError with client.batches - fixed by using client.messages.batches
- System message conflicts - resolved by removing system parameter entirely
- JSON parsing errors from responses with additional context - fixed with string extraction
- Test failures after API changes - updated all tests to match new simplified interface

## Edge Cases
- Empty message lists return empty results
- Batch processing can timeout - configurable poll_interval helps
- API responses may contain explanatory text around JSON - extraction handles this
- Missing API key raises clear ValueError with helpful message

## Key Decisions
- Removed system parameter for simplicity - users include context in message content
- Direct API integration instead of heavy BatchJob wrapper
- Manual JSON parsing for reliability over complex instructor parsing
- Comprehensive test coverage with both unit tests (mocked) and E2E tests (real API)

## Final API Example
```python
batch(
    messages=[[{"role": "user", "content": f"You are a spam detection expert. Is this spam? {email}"}] for email in emails], 
    model="claude-3-haiku-20240307", 
    response_model=SpamResult
)
```