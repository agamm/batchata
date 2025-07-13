# Development Guide

## Architecture Overview

```mermaid
classDiagram
    class Batch {
        +List~Job~ jobs
        +BatchConfig config
        +run() BatchRun
    }
    
    class BatchConfig {
        +str state_file
        +str results_dir
        +int max_concurrent
        +float cost_limit_usd
        +Dict default_params
    }
    
    class Job {
        +str id
        +str model
        +List messages
        +Path file
        +str prompt
        +float temperature
        +to_dict() Dict
        +from_dict() Job
    }
    
    class BatchRun {
        +start()
        +wait()
        +shutdown()
        +on_progress(callback) BatchRun
        +status() Dict
        +results() Dict~str,JobResult~
    }
    
    class ConcurrentExecutor {
        +submit_if_allowed(provider, jobs) Future
        +get_completed() List~Tuple~
        +get_active_count() int
        +get_stats() Dict
    }
    
    class JobResult {
        +str job_id
        +str response
        +Any parsed_response
        +List~Citation~ citations
        +int input_tokens
        +int output_tokens
        +float cost_usd
        +bool is_success
        +str error
        +to_dict() Dict
        +from_dict() JobResult
    }
    
    class Provider {
        <<interface>>
        +create_batch(jobs) BatchRequest
        +get_batch_status(batch_id) str
        +get_batch_results(batch_id) List~JobResult~
        +estimate_cost(jobs) float
    }
    
    class AnthropicProvider {
        +create_batch(jobs) BatchRequest
        +get_batch_status(batch_id) str
        +get_batch_results(batch_id) List~JobResult~
        +estimate_cost(jobs) float
    }
    
    class BatchRequest {
        +str provider_batch_id
        +Provider provider
        +List~Job~ jobs
        +datetime submitted_at
    }
    
    class CostTracker {
        +can_afford(cost_usd) bool
        +track_spend(cost_usd)
        +remaining() float
    }
    
    class StateManager {
        +save(state)
        +load() BatchState
        +clear()
    }
    
    class BatchState {
        +str batch_id
        +str created_at
        +List pending_jobs
        +List completed_results
        +List failed_jobs
        +float total_cost_usd
        +to_dict() Dict
        +from_dict() BatchState
    }
    
    Batch --> BatchConfig : has
    Batch --> Job : contains *
    Batch --> BatchRun : creates
    
    BatchRun --> BatchConfig : uses
    BatchRun --> Job : processes *
    BatchRun --> ConcurrentExecutor : uses
    BatchRun --> StateManager : uses
    BatchRun --> CostTracker : uses
    BatchRun --> JobResult : produces *
    
    ConcurrentExecutor --> Provider : uses
    ConcurrentExecutor --> BatchRequest : manages *
    ConcurrentExecutor --> CostTracker : uses
    
    AnthropicProvider ..|> Provider : implements
    
    Provider --> BatchRequest : creates
    Provider --> JobResult : returns *
    
    StateManager --> BatchState : saves/loads
```

### Key Design Patterns

- **Builder Pattern**: `Batch` → `BatchConfig` + `jobs` → `BatchRun`
- **Provider Pattern**: Abstract provider interface for different AI services  
- **Separation of Concerns**: Configuration, data, and execution are separate
- **Self-contained Serialization**: Data classes handle their own to_dict/from_dict
- **YAGNI Principle**: Removed checkpointing, complex stats, reservation patterns

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