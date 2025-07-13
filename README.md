# Batchata

Batch processing for AI models with cost tracking, state persistence, and parallel execution.

## Why Batchata?

Other libraries don't have batch request capabilities out of the box. Batchata provides:

- Native batch processing (50% cost savings via provider APIs)
- Cost tracking and limits
- State persistence for network interruption recovery
- Parallel execution
- Structured output with Pydantic models
- Citation extraction and field mapping (supported only by anthropic atm)

## Installation

### pip
```bash
pip install batchata
```

### uv
```bash
uv add batchata
```

## Quick Start

```python
from batchata import Batch

# Simple batch processing
batch = Batch(state_file="./state.json", results_dir="./output", max_concurrent=10)
    .defaults(model="claude-3-sonnet")
    .add_cost_limit(usd=15)

for file in files:
    batch.add_job(file=file, prompt="Summarize")

run = batch.run(wait=True)
print(run.status())  # Shows progress and costs
results = run.results()  # Dict[job_id, JobResult]
```


## API

### Batch

```python
Batch(state_file: str, results_dir: str, max_concurrent: int = 10)
```

- `state_file`: Path to save batch state for recovery (in case of network interruption)
- `results_dir`: Directory to store individual job results  
- `max_concurrent`: Maximum parallel batch requests (default: 10)

**Methods:**

#### `.defaults(**kwargs)`
Set default parameters for all jobs. Common parameters:
- `model`: Model name (e.g., "claude-3-sonnet", "gpt-4")
- `temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens`: Maximum tokens to generate (default: 1000)

#### `.add_cost_limit(usd: float)`
Set maximum spend limit. Batch will stop accepting new jobs when limit is reached.

#### `.on_progress(callback: Callable[[Dict], None])`
Set progress callback function. Callback receives a dict with:
- `batch_id`: Current batch identifier
- `total`: Total number of jobs
- `pending`: Jobs waiting to start
- `active`: Jobs currently processing
- `completed`: Successfully completed jobs
- `failed`: Failed jobs
- `cost_usd`: Current total cost
- `cost_limit_usd`: Cost limit (if set)
- `is_complete`: Whether batch is finished

#### `.add_job(...)`
Add a job to the batch. Parameters:
- `messages`: Chat messages (list of dicts with "role" and "content")
- `file`: Path to file for file-based input
- `prompt`: Prompt to use with file input
- `model`: Override default model
- `temperature`: Override default temperature (0.0-1.0)
- `max_tokens`: Override default max tokens
- `response_model`: Pydantic model for structured output
- `enable_citations`: Extract citations from response (default: False)

Note: Provide either `messages` OR `file`+`prompt`, not both.

#### `.run(wait: bool = False)`
Execute the batch. Returns a `BatchRun` object.
- `wait=True`: Block until all jobs complete
- `wait=False`: Return immediately, process in background

### BatchRun

Object returned by `batch.run()`:

- `.status(print: bool = False)` - Get current batch status
- `.results()` - Get completed results as Dict[str, JobResult]
- `.wait(timeout: float = None)` - Wait for batch completion
- `.shutdown(wait_for_active: bool = True)` - Gracefully shutdown

### JobResult

- `job_id`: Unique identifier
- `response`: Raw text response
- `parsed_response`: Structured data (if response_model used)
- `citations`: List of citations (if enabled)
- `input_tokens`: Input token count
- `output_tokens`: Output token count
- `cost_usd`: Cost for this job
- `error`: Error message (if failed)

## File Structure

```
./results/
├── job-abc123.json
├── job-def456.json
└── job-ghi789.json

./batch_state.json  # Batch state
```

## Configuration

Set your API keys as environment variables:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

You can also use a `.env` file in your project root (requires python-dotenv):
```python
from dotenv import load_dotenv
load_dotenv()

from batchata import Batch
# Your API keys will now be loaded from .env
```

## License

MIT License - see LICENSE file for details.