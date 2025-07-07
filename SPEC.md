# file-processing

## Goal
Add PDF file support to ai-batch using Anthropic's document content blocks.

## Current state
The ai-batch module processes text inputs. Need to extend for PDF files using document content blocks.

## Requirements
- [x] Support PDF files in batch() function
- [x] Use document content blocks with base64 encoding
- [x] Extract Pydantic models from PDF content
- [x] Handle multiple PDFs in single batch

## Progress
### Completed
- Added `pdf_to_document_block()` function to convert PDFs to document blocks
- Added `batch_files()` helper for processing multiple files
- Updated exports in __init__.py
- Created comprehensive test suite for PDF processing
- Created example demonstrating PDF data extraction
- Fixed type errors in batch function with proper overloads
- Split file processing features into separate module (file_processing.py)
- Renamed core module from ai_batch.py to core.py for better organization
- Created proper valid PDF structures for examples and tests
- Cleaned up type signatures for batch_files with proper overloads

### In Progress  
- None

### Next Steps
1. ~~Write tests for PDF processing~~ ✓
2. ~~Implement document content blocks~~ ✓
3. ~~Test and refactor~~ ✓

## Tests
### Tests to write
- [x] Test PDF to base64 conversion
- [x] Test document content block format
- [x] Test batch processing of multiple PDFs
- [x] Test Pydantic model extraction

### Tests passing
- All 4 PDF processing tests passing
- All 13 total tests passing

## URL References
- [Anthropic PDF support](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support) - Document content blocks

## Notes
Example: Generate 3 PDFs at runtime, convert to base64, send as document content blocks to extract Pydantic models in batch.