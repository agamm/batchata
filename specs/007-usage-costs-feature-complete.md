# Add usage costs to BatchJob

## Goal
Add cost tracking functionality to BatchJob to track token usage and estimated costs for batch operations using the tokencost library.

## Current state
- BatchJob tracks basic statistics (elapsed time, results count, citations count)
- Anthropic provider returns raw API results but doesn't extract usage information
- No cost tracking or token usage information is available to users

## Requirements
- [x] Extract token usage from API responses (`result.message.usage` -> `input_tokens`, `output_tokens`, `service_tier`)
- [x] Install and integrate tokencost library for cost calculations
- [x] Use `calculate_cost_by_tokens(num_tokens, model, token_type)` function
- [x] Apply 50% batch discount when `service_tier` is "batch"
- [x] Add cost information to BatchJob.stats() method
- [x] Support cost aggregation across all requests in a batch
- [x] Handle different model pricing (Claude 3.5 Sonnet, Haiku, etc.)
- [x] Provide both detailed per-request and total batch costs

## Progress
### Completed
- ✅ Added tokencost dependency to pyproject.toml
- ✅ Implemented usage data extraction from API responses
- ✅ Added cost calculation logic with tokencost integration
- ✅ Applied 50% batch discount when service_tier is "batch"
- ✅ Updated BatchJob.stats() method to include cost information
- ✅ Added cost aggregation across all requests in a batch
- ✅ Implemented support for different model pricing
- ✅ Provided both detailed per-request and total batch costs
- ✅ All tests passing

### In Progress
- None

### Next Steps
- Feature complete

## Tests
### Tests to write
- [x] Test usage extraction from API responses
- [x] Test cost calculation with different models using tokencost
- [x] Test batch discount application (50% off when service_tier="batch")
- [x] Test cost aggregation across batch requests
- [x] Test stats() method includes cost information
- [x] Test empty batch cost handling
- [x] Test different model pricing
- [x] Test batch vs standard pricing

### Tests passing
- ✅ All 8 usage cost tests passing
- ✅ All 49 existing tests still passing (no regressions)

## URL References
- [Anthropic Batch Processing Pricing](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) - 50% discount for batch API
- [Anthropic API Response Format](https://docs.anthropic.com/en/api/messages) - Usage field structure
- [tokencost calculate_cost_by_tokens](https://github.com/AgentOps-AI/tokencost/blob/main/tokencost/costs.py#L188) - Cost calculation function
- [tokencost Repository](https://github.com/AgentOps-AI/tokencost) - Token cost calculation library

## Learnings
- API responses include usage data in `result.result.message.usage` with `input_tokens`, `output_tokens`, `service_tier`
- tokencost library provides `calculate_cost_by_tokens(num_tokens, model, token_type)` function
- Batch API offers 50% discount on all usage when `service_tier` is "batch"
- Need to handle token_type as 'input', 'output', or 'cached' for tokencost function
- tokencost supports Anthropic models: claude-3-5-sonnet, claude-3-5-haiku, claude-3-haiku, claude-3-opus
- Service tier should default to "batch" since we're using the batch API
- Batch discount should be a class constant (BATCH_DISCOUNT = 0.5) for maintainability
- Cost tracking is backward compatible - existing code continues to work
- Empty batches return zero costs gracefully

## Notes
- Should maintain backward compatibility with existing BatchJob interface
- Cost tracking should be enabled by default but not break existing functionality
- Consider caching cost calculations to avoid repeated computation
- Need to map internal model names to tokencost model identifiers