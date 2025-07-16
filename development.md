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
        +save_raw_responses(enabled) Batch
        +add_job(...) Batch
        +run(wait, on_progress) BatchRun
    }
    
    class BatchParams {
        +Optional~str~ state_file
        +str results_dir
        +int max_concurrent
        +int items_per_batch
        +Optional~float~ cost_limit_usd
        +Dict default_params
        +bool reuse_state
        +bool save_raw_responses
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
        +status() Dict
        +results() Dict~str,JobResult~
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
    BatchRun --> JobResult : produces *
    
    AnthropicProvider ..|> Provider : implements
    
    Provider --> JobResult : returns *
    
    StateManager --> BatchState : saves/loads
    
    CostTracker --> BatchRun : used by
```

### Key Design Patterns

- **Builder Pattern**: `Batch` provides fluent interface for configuration
- **Provider Pattern**: Abstract provider interface for different AI services  
- **Synchronous Processing**: `BatchRun` processes jobs in batches synchronously
- **State Persistence**: Automatic saving/resuming via `StateManager`
- **Cost Control**: Built-in cost tracking and limits via `CostTracker`

## Running Tests

Tests require an Anthropic API key since they make real API calls.

```bash
# Install dependencies
uv sync --dev

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

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

For tests to run in GitHub Actions, add your API key as a secret:
1. Go to Settings → Secrets and variables → Actions
2. Add new secret: `ANTHROPIC_API_KEY`