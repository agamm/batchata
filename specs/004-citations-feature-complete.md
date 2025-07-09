# Add citation ability to ai-batch

## Goal
Add citation support to ai-batch, allowing models to include source references when extracting or generating content from provided documents.

## Current state
The ai-batch module supports batch processing of text and PDF inputs with structured output using Pydantic models. Currently, responses don't include citations or references to source materials.

## Requirements
- [x] Parse citation data from multi-block API responses
- [x] Create built-in Citation and CitedText models
- [x] Handle responses with mixed text and citation blocks
- [x] Return CitedText objects that include both text and citations
- [x] Add enable_citations parameter to control citation requests

## Progress
### Completed
- Created Citation and CitedText models in citations.py
- Added enable_citations parameter to pdf_to_document_block()
- Added enable_citations parameter to batch_files()
- Updated parse_results to handle multi-block content with citations
- Created private methods for citation parsing and JSON extraction
- Added citation example demonstrating usage
- Exported Citation and CitedText from main module
- Fixed citation object parsing (used getattr instead of .get())
- Made document_title optional to handle None values from API
- Successfully tested end-to-end citation functionality
- **MAJOR REFACTOR**: Replaced CitedText/StructuredCitedResult with 4-mode API
- **NEW**: BatchJob class for async-style result retrieval
- **NEW**: Field-level citations for structured responses (FieldCitations)
- **NEW**: Comprehensive test suite for all citation modes
- **NEW**: Validation for nested Pydantic models (not supported with citations)
- **NEW**: Message format validation and batch size limits
- **CLEANUP**: Moved PDF utilities to tests/utils/ (testing-only code)

### Final Implementation
The citation feature evolved into a comprehensive 4-mode API:
1. **Mode 1**: Plain text (no response_model, no citations)
2. **Mode 2**: Structured only (response_model, no citations)  
3. **Mode 3**: Text + Citations (no response_model, citations)
4. **Mode 4**: Structured + Field Citations (response_model + citations)

### Next Steps
- Feature is complete and ready for production use

## Tests
### Tests to write
- [ ] Test citation extraction from responses
- [ ] Test multi-document source tracking
- [ ] Test different citation formats

### Tests passing
- Basic functionality tests (imports, exports, parameter handling)
- End-to-end citation extraction working with real PDFs
- Citation object parsing and CitedText return values
- Multiple content block handling

## URL References
Links to external resources, documentation, or codebases that help understand the requirements:
- [Anthropic Citations](https://docs.anthropic.com/en/docs/build-with-claude/citations) - Claude's citation capabilities

## Notes
Consider how to handle citations when processing multiple documents in a batch - each response should be able to reference its source document(s) appropriately.