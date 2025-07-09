# Raw response files param

## Goal
Add a simple `raw_results_dir` parameter to save raw API responses to files for debugging and analysis purposes.

## Current state
The library processes API responses and returns structured outputs, but raw responses are only available internally during processing. Raw responses exist at two points:
- `AnthropicBatchProvider.get_results()` (line 148-149 in `anthropic.py`)
- `BatchJob.results()` (line 65 in `batch_job.py`) before parsing

## Requirements
- [x] Add `raw_results_dir: Optional[str] = None` parameter to `core.batch()`
- [x] Add `raw_results_dir: Optional[str] = None` parameter to `file_processing.batch_files()`
- [x] Modify `BatchJob` to accept and store `raw_results_dir`
- [x] Save raw responses as JSON files before parsing in `BatchJob.results()`
- [x] Create directory if it doesn't exist
- [x] Use clear file naming: `{batch_id}_{request_index}.json`
- [x] Maintain backward compatibility (None = no saving)

## Progress
### Completed
- ✅ Added `raw_results_dir` parameter to `BatchJob.__init__()`
- ✅ Implemented `_save_raw_responses()` method in `BatchJob`
- ✅ Updated `core.batch()` API to accept `raw_results_dir` parameter
- ✅ Updated `file_processing.batch_files()` API to accept `raw_results_dir` parameter
- ✅ Added comprehensive test suite for raw response saving
- ✅ Created debug script for testing functionality
- ✅ Updated README with examples and documentation

### In Progress
- None

### Next Steps
- Feature is complete and ready for use

## Tests
### Tests to write
- [x] Test raw response file creation with valid directory
- [x] Test directory creation when it doesn't exist
- [x] Test file naming pattern matches batch_id and request indices
- [x] Test backward compatibility (None parameter = no files saved)
- [x] Test with both text and structured responses
- [x] Test API integration for both batch() and batch_files()
- [x] Test empty batch handling
- [x] Test with citations enabled

### Tests passing
- ✅ All 9 tests passing in test_raw_response_saving.py

## URL References
Links to external resources, documentation, or codebases that help understand the requirements:
- [Anthropic Batch API](https://docs.anthropic.com/claude/reference/messages_batches) - Understanding batch response structure

## Learnings
- Raw responses are available at `BatchJob.results()` before parsing - perfect insertion point
- Current API follows consistent optional parameter pattern with None defaults
- BatchJob already handles batch_id, making file naming straightforward
- No breaking changes needed - completely additive feature
- Raw response objects need to be serialized using `model_dump()` to be JSON-compatible
- File organization can be improved by using subdirectories for different test scenarios
- PDF processing requires models that support PDF input (e.g., claude-3-5-sonnet, not claude-3-haiku)

## Notes
Keep implementation minimal and focused:
- Only 3 files need modification (core.py, file_processing.py, batch_job.py)
- Files saved as JSON with batch_id and request index for easy identification
- Directory creation handled automatically for user convenience
- Works with all existing modes (text, structured, citations, file processing)

## Implementation Details
- Raw responses are Pydantic models (MessageBatchIndividualResponse) that need `.model_dump()` for JSON serialization
- File naming pattern: `{batch_id}_{index}.json` where index matches the request order
- Empty or failed responses are handled gracefully
- Debug script demonstrates usage across all modes with organized subdirectories