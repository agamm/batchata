# Development Guide

## Architecture Overview

```mermaid
classDiagram
    class Batch {
        +BatchParams config
        +List~Job~ jobs
        +set_state(file, reuse_previous) Batch
        +set_default_params(**kwargs) Batch
        +add_cost_limit(usd) Batch
        +set_verbosity(level) Batch
        +add_job(...) Batch
        +run(wait, on_progress, print_status) BatchRun
    }
    
    class BatchParams {
        +Optional~str~ state_file
        +str results_dir
        +int max_parallel_batches
        +int items_per_batch
        +Optional~float~ cost_limit_usd
        +Dict default_params
        +bool reuse_state
        +bool raw_files
        +str verbosity_level
    }
    
    class Job {
        +str id
        +str model
        +Optional~List~ messages
        +Optional~Path~ file
        +Optional~str~ prompt
        +float temperature
        +int max_tokens
        +Optional~Type~ response_model
        +bool enable_citations
        +to_dict() Dict
        +from_dict() Job
    }
    
    class BatchRun {
        +BatchParams config
        +List~Job~ jobs
        +start()
        +set_on_progress(callback, interval)
        +status(print_status) Dict
        +results() Dict~str,JobResult~
        +wait(timeout) None
        +shutdown(wait_for_active) None
    }
    
    class JobResult {
        +str job_id
        +str response
        +Optional~Union~ parsed_response
        +Optional~List~ citations
        +int input_tokens
        +int output_tokens
        +float cost_usd
        +Optional~str~ error
        +is_success() bool
        +to_dict() Dict
        +from_dict() JobResult
    }
    
    class Provider {
        <<abstract>>
        +validate_job(job)
        +create_batch(jobs) str
        +get_batch_status(batch_id) str
        +get_batch_results(batch_id) List~JobResult~
        +cancel_batch(batch_id) bool
        +estimate_cost(jobs) float
    }
    
    class AnthropicProvider {
        +validate_job(job)
        +create_batch(jobs) str
        +get_batch_status(batch_id) str
        +get_batch_results(batch_id) List~JobResult~
        +cancel_batch(batch_id) bool
        +estimate_cost(jobs) float
    }
    
    class OpenAIProvider {
        +validate_job(job)
        +create_batch(jobs) str
        +get_batch_status(batch_id) str
        +get_batch_results(batch_id) List~JobResult~
        +cancel_batch(batch_id) bool
        +estimate_cost(jobs) float
        +get_polling_interval() float
    }
    
    class CostTracker {
        +Optional~float~ limit_usd
        +can_afford(cost_usd) bool
        +track_spend(cost_usd)
        +remaining() Optional~float~
        +get_stats() Dict
    }
    
    class StateManager {
        +save(state)
        +load() Optional~BatchState~
        +clear()
    }
    
    class RichBatchProgressDisplay {
        +start_progress()
        +update_progress(stats, elapsed_time, batch_data)
        +finish_progress()
        +handle_cancellation()
    }
    
    class BatchState {
        +str batch_id
        +str created_at
        +List~Job~ pending_jobs
        +Dict~str,JobResult~ completed_results
        +Dict~str,str~ failed_jobs
        +float total_cost_usd
        +to_dict() Dict
        +from_dict() BatchState
    }
    
    Batch --> BatchParams : has
    Batch --> Job : contains *
    Batch --> BatchRun : creates
    
    BatchRun --> BatchParams : uses
    BatchRun --> Job : processes *
    BatchRun --> Provider : uses directly
    BatchRun --> StateManager : uses
    BatchRun --> CostTracker : uses
    BatchRun --> RichBatchProgressDisplay : uses
    BatchRun --> JobResult : produces *
    
    AnthropicProvider ..|> Provider : implements
    OpenAIProvider ..|> Provider : implements
    
    Provider --> JobResult : returns *
    
    StateManager --> BatchState : saves/loads
    
    CostTracker --> BatchRun : used by
```

### Key Design Patterns

- **Builder Pattern**: `Batch` provides fluent interface for configuration
- **Provider Pattern**: Abstract provider interface for different AI services (Anthropic, OpenAI)  
- **Synchronous Processing**: `BatchRun` processes jobs in batches synchronously
- **State Persistence**: Automatic saving/resuming via `StateManager`
- **Cost Control**: Built-in cost tracking and limits via `CostTracker`

## Running Tests

Tests require API keys for providers since some tests make real API calls.

```bash
# Install dependencies
uv sync --dev

# Set API keys
export ANTHROPIC_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"

# Run all tests (parallel)
uv run pytest -v -n auto 

# Run a specific test file
uv run pytest tests/test_ai_batch.py

# Run a specific test
uv run pytest tests/test_ai_batch.py::test_batch_empty_messages
```

## Releasing a New Version

```bash
# One-liner to update version, commit, push, and release
VERSION=0.0.2 && \
sed -i '' "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml && \
git add pyproject.toml && \
git commit -m "Bump version to $VERSION" && \
git push && \
gh release create v$VERSION --title "v$VERSION" --generate-notes
```

## GitHub Secrets Setup

For tests to run in GitHub Actions, add your API keys as secrets:
1. Go to Settings → Secrets and variables → Actions
2. Add secrets: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

## Debug Files Structure

When `raw_files=True` (default), debug files are saved to help with troubleshooting:

```
results_dir/
├── raw_files/
│   ├── input/           # Batch input files
│   │   ├── anthropic_batch_{id}_input.jsonl
│   │   └── openai_batch_{id}_input.jsonl
│   ├── output/          # Batch output files
│   │   ├── anthropic_batch_{id}_results.jsonl
│   │   └── openai_batch_{id}_results.jsonl
│   └── responses/       # Individual job responses
│       ├── {job_id}_raw.json
│       └── {job_id}_raw.json
└── [job results]        # Processed JobResult files
```

## Provider-Specific Features

### Anthropic Provider
- **Polling Interval**: 1 second
- **Citations**: Full citation support with page mapping
- **Batch Format**: Native Anthropic batch API
- **File Support**: PDFs, images with text extraction validation

### OpenAI Provider  
- **Polling Interval**: 5 seconds (avoids Cloudflare rate limiting)
- **Citations**: Not supported
- **Batch Format**: JSONL file upload to batch API
- **File Support**: PDFs, images with base64 encoding