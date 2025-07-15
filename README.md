# Batchata

[![Build Status](https://github.com/agamm/batchata/workflows/Tests/badge.svg)](https://github.com/agamm/batchata/actions)
[![PyPI version](https://badge.fury.io/py/batchata.svg)](https://badge.fury.io/py/batchata)

Unified API for AI Batch requests with cost tracking, Pydantic responses, citation mapping and parallel execution.

*This library is currently in alpha - so there will be breaking changes*

## Why Batchata?

- Native batch processing (50% cost savings via provider APIs)
- Set `max_cost_usd` limits for batch requests
- State persistence in case of network interruption
- Structured output `.json` format with Pydantic models
- Citation support and field mapping (supported only by anthropic atm)

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
batch = Batch(state_file="./state.json", results_dir="./output")
    .defaults(model="claude-sonnet-4-20250514")
    .add_cost_limit(usd=15)

for file in files:
    batch.add_job(file=file, prompt="Summarize")

run = batch.run(wait=True)

results = run.results()  # Dict[job_id, JobResult]
```

## Complete Example

```python
from batchata import Batch
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env

# Define structured output
class InvoiceAnalysis(BaseModel):
    invoice_number: str
    total_amount: float
    vendor: str
    payment_status: str

# Create batch configuration
batch = (
    Batch(
        state_file="./invoice_state.json", 
        results_dir="./invoice_results",
        max_concurrent=1,
        items_per_batch=3,
        reuse_state=False 
    )
    .defaults(model="claude-sonnet-4-20250514", temperature=0.0)
    .add_cost_limit(usd=5.0)
    .set_verbosity("warn") 
)

# Add jobs with structured output and citations
invoice_files = ["path/to/invoice1.pdf", "path/to/invoice2.pdf", "path/to/invoice3.pdf"]
for invoice_file in invoice_files:
    batch.add_job(
        file=invoice_file,
        prompt="Extract the invoice number, total amount, vendor name, and payment status.",
        response_model=InvoiceAnalysis,
        enable_citations=True
    )

# Execute with progress tracking
print("Starting batch processing...")
run = batch.run(
    wait=True, 
    on_progress=lambda s, t: print(
        f"\rProgress: {s['completed']}/{s['total']} jobs | "
        f"Batches: {s['batches_completed']}/{s['batches_total']} | "
        f"Cost: ${s['cost_usd']:.3f}/{s['cost_limit_usd']} | "
        f"Time: {t:.1f}s", 
        end=""
    )
)

# Get results
results = run.results()

# Process results
for job_id, result in results.items():
    if result.is_success:
        analysis = result.parsed_response
        citations = result.citation_mappings
        print(f"\nInvoice: {analysis.invoice_number} (page: {citations.get("invoice_number").page})")
        print(f"  Vendor: {analysis.vendor} (page: {citations.get("vendor").page})")
        print(f"  Total: ${analysis.total_amount:.2f} (page: {citations.get("total_amount").page})")
        print(f"  Status: {analysis.payment_status} (page: {citations.get("payment_status").page})")

    else:
        print(f"\nJob {job_id} failed: {result.error}")
```


## API

### Batch

```python
Batch(
    state_file: str, 
    results_dir: str, 
    max_concurrent: int = 10,
    items_per_batch: int = 10,
    reuse_state: bool = True,
    save_raw_responses: Optional[bool] = None
)
```

- `state_file`: Path to save batch state for recovery (in case of network interruption)
- `results_dir`: Directory to store individual job results  
- `max_concurrent`: Maximum parallel batch requests (default: 10)
- `items_per_batch`: Number of jobs per provider batch (affects cost tracking accuracy, default: 10)
- `reuse_state`: Whether to resume from existing state file and delete previous results_dir file results (default: True)
- `save_raw_responses`: Whether to save raw API responses in the results dir (default: True if results_dir is set)

**Methods:**

#### `.defaults(**kwargs)`
Set default parameters for all jobs. Common parameters:
- `model`: Model name (e.g., "claude-sonnet-4-20250514", "gpt-4")
- `temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens`: Maximum tokens to generate (default: 1000)

#### `.add_cost_limit(usd: float)`
Set maximum spend limit. Batch will stop accepting new jobs when limit is reached.

#### `.set_verbosity(level: str)`
Set logging verbosity level. Useful for production environments.
- Levels: "debug", "info" (default), "warn", "error"
- Example: `batch.set_verbosity("error")` for production

#### `.add_job(...)`
Add a job to the batch. Parameters:
- `messages`: Chat messages (list of dicts with "role" and "content")
- `file`: Path to file for file-based input (supports string paths, Path objects, and PDF files)
- `prompt`: Prompt to use with file input
- `model`: Override default model
- `temperature`: Override default temperature (0.0-1.0)
- `max_tokens`: Override default max tokens
- `response_model`: Pydantic model for structured output
- `enable_citations`: Extract citations from response (default: False)

Note: Provide either `messages` OR `file`+`prompt`, not both.

**PDF Citation Validation**: When using Anthropic models with `enable_citations=True` on PDF files, Batchata automatically validates that the PDF contains extractable text. Image-only or scanned PDFs will raise a ValidationError since citations cannot be extracted from them. This validation is Anthropic-specific and doesn't affect other providers.

#### `.run(wait: bool = False, on_progress: Callable = None)`
Execute the batch. Returns a `BatchRun` object.
- `wait=True`: Block until all jobs complete
- `wait=False`: Return immediately, process in background
- `on_progress`: Optional progress callback function

### BatchRun

Object returned by `batch.run()`:

- `.status(print_status: bool = False)` - Get current batch status
- `.results()` - Get completed results as Dict[str, JobResult]
- `.wait(timeout: float = None)` - Wait for batch completion
- `.on_progress(callback, interval=3.0)` - Set progress monitoring callback
- `.shutdown(wait_for_active: bool = True)` - Gracefully shutdown

The progress callback receives a dict with:
- `batch_id`: Current batch identifier
- `total`: Total number of jobs
- `pending`: Jobs waiting to start
- `active`: Jobs currently processing
- `completed`: Successfully completed jobs
- `failed`: Failed jobs
- `cost_usd`: Current total cost
- `cost_limit_usd`: Cost limit (if set)
- `is_complete`: Whether batch is finished
- `batches_completed`: Number of completed batches
- `batches_total`: Total number of batches
- `batches_pending`: Number of pending batches
- `items_per_batch`: Items per batch setting

### JobResult

- `job_id`: Unique identifier
- `raw_response`: Raw text response
- `parsed_response`: Structured data (if response_model used)
- `citations`: List of Citation objects (if enabled)
- `citation_mappings`: Dict[str, List[Citation]] - Maps field names to relevant citations (not 100% accurate, only with response_model)
- `input_tokens`: Input token count
- `output_tokens`: Output token count
- `cost_usd`: Cost for this job
- `error`: Error message (if failed)
- `is_success`: Property that returns True if job completed successfully
- `total_tokens`: Property that returns total tokens used (input + output)

### Citation

Each Citation object contains:
- `text`: The cited text
- `source`: Source identifier (e.g., file name)
- `page`: Page number if applicable (for PDFs)
- `metadata`: Additional metadata dict

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

## Limitations

- Parallel execution not implemented yet.
- Field/citation mapping is heuristic, which means it isn't perfect.
- Citation mapping only works with flat Pydantic models (no nested BaseModel fields).
- Right now only Anthropic Batch requests are supported.
- Cost tracking is not precise as the actual usage is only known after the batch is complete, try setting `items_per_batch` to a lower value for more accurate cost tracking.


## License

MIT License - see LICENSE file for details.