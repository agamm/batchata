# Batchata

<img alt="Batchata AI Batch Build Status" src="https://github.com/agamm/batchata/workflows/Tests/badge.svg" /><img alt="Batchata AI Batch PyPI version" src="https://badge.fury.io/py/batchata.svg" />

Unified Python API for AI Batch requests with cost tracking, Pydantic responses, citation mapping and parallel execution.

<img width="1328" height="598" alt="image" src="https://github.com/user-attachments/assets/b1b43070-f664-41a2-a85d-e2c589df556c" />

*This library is currently in alpha - so there will be breaking changes*

## Why AI-batching?

AI providers offer batch APIs that process requests asynchronously at 50% reduced cost compared to real-time APIs. This is ideal for workloads like document processing, data analysis, and content generation where immediate responses aren't required. However, managing batch jobs across providers, tracking costs, handling failures, and mapping citations back to source documents quickly becomes complex - that's where Batchata comes in.

## Batchata Features

- Native batch processing (50% cost savings via provider APIs)
- Set `max_cost_usd` limits for batch requests
- **Time limit control** with `.add_time_limit(seconds=, minutes=, hours=)`
- State persistence in case of network interruption
- Structured output `.json` format with Pydantic models
- Citation support and field mapping (Anthropic only)
- Multiple provider support (Anthropic, OpenAI)

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
batch = Batch(results_dir="./output")
    .set_default_params(model="claude-sonnet-4-20250514")  # or "gpt-4.1-2025-04-14"
    .add_cost_limit(usd=5.0)

for file in files:
    batch.add_job(file=file, prompt="Summarize")

run = batch.run()

results = run.results()  # {"completed": [JobResult], "failed": [JobResult], "cancelled": [JobResult]}
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
batch = Batch(
        results_dir="./invoice_results",
        max_parallel_batches=1,
        items_per_batch=3
    )
    .set_state(file="./invoice_state.json", reuse_previous=False)
    .set_default_params(model="claude-sonnet-4-20250514", temperature=0.0)
    .add_cost_limit(usd=5.0)
    .add_time_limit(minutes=10)  # Time limit of 10 minutes
    .set_verbosity("warn") 

# Add jobs with structured output and citations
invoice_files = ["path/to/invoice1.pdf", "path/to/invoice2.pdf", "path/to/invoice3.pdf"]
for invoice_file in invoice_files:
    batch.add_job(
        file=invoice_file,
        prompt="Extract the invoice number, total amount, vendor name, and payment status.",
        response_model=InvoiceAnalysis,
        enable_citations=True
    )

# Execute with rich progress display
print("Starting batch processing...")
run = batch.run(print_status=True)

# Or use custom progress callback
run = batch.run(
    on_progress=lambda s, t, b: print(
        f"\rProgress: {s['completed']}/{s['total']} jobs | "
        f"Batches: {s['batches_completed']}/{s['batches_total']} | "
        f"Cost: ${s['cost_usd']:.3f}/{s['cost_limit_usd']} | "
        f"Time: {t:.1f}s", 
        end=""
    )
)

# Get results
results = run.results()

# Process successful results
for result in results["completed"]:
    analysis = result.parsed_response
    citations = result.citation_mappings
    print(f"\nInvoice: {analysis.invoice_number} (page: {citations.get("invoice_number").page})")
    print(f"  Vendor: {analysis.vendor} (page: {citations.get("vendor").page})")
    print(f"  Total: ${analysis.total_amount:.2f} (page: {citations.get("total_amount").page})")
    print(f"  Status: {analysis.payment_status} (page: {citations.get("payment_status").page})")

# Process failed/cancelled results  
for result in results["failed"]:
    print(f"\nJob {result.job_id} failed: {result.error}")

for result in results["cancelled"]:
    print(f"\nJob {result.job_id} was cancelled: {result.error}")
```


## API

### Batch

```python
Batch(
    results_dir: str, 
    max_parallel_batches: int = 10,
    items_per_batch: int = 10,
    raw_files: Optional[bool] = None
)
```

- `results_dir`: Directory to store individual job results  
- `max_parallel_batches`: Maximum parallel batch requests (default: 10)
- `items_per_batch`: Number of jobs per provider batch (affects cost tracking accuracy, default: 10)
- `raw_files`: Whether to save debug files (raw API requests/responses) in the results dir (default: True if results_dir is set)

**Methods:**

#### `.set_state(file: Optional[str] = None, reuse_previous: bool = True)`
Set state file configuration for recovery in case of network interruption.
- `file`: Path to save batch state (default: None, uses temp file)
- `reuse_previous`: Whether to resume from existing state file (default: True)

#### `.set_default_params(**kwargs)`
Set default parameters for all jobs. Common parameters:
- `model`: Model name (e.g., "claude-sonnet-4-20250514", "gpt-4.1-2025-04-14")
- `temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens`: Maximum tokens to generate (default: 1000)

#### `.add_cost_limit(usd: float)`
Set maximum spend limit. Batch will stop accepting new jobs when limit is reached.

#### `.set_verbosity(level: str)`
Set logging verbosity level. Useful for production environments.
- Levels: "debug", "info" (default), "warn", "error"
- Example: `batch.set_verbosity("error")` for production

#### `.add_time_limit(seconds=, minutes=, hours=)`
Set maximum execution time for the entire batch. When time limit is reached, all active provider batches are cancelled and remaining jobs are marked as failed.
- `seconds`: Time limit in seconds (optional)
- `minutes`: Time limit in minutes (optional) 
- `hours`: Time limit in hours (optional)
- Minimum: 10 seconds, Maximum: 24 hours
- Can combine units: `.add_time_limit(hours=1, minutes=30, seconds=15)`
- **No exceptions thrown** - jobs that exceed time limit appear in failed results

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

#### `.run(on_progress: Callable = None, print_status: bool = False)`
Execute the batch. Returns a `BatchRun` object.
- `on_progress`: Optional progress callback function that receives `(stats_dict, elapsed_time, batch_data)`
- `print_status=True`: Enable rich progress display with real-time updates

<img width="2230" height="222" alt="image" src="https://github.com/user-attachments/assets/caf549a6-92a1-4ee0-8ac7-eda2d0f280a7" />


### BatchRun

Object returned by `batch.run()`:

- `.status(print_status: bool = False)` - Get current batch status
- `.results()` - Get all results organized by status: `{"completed": [JobResult], "failed": [JobResult], "cancelled": [JobResult]}`
- `.get_failed_jobs()` - Get failed jobs as Dict[str, str] (deprecated, use `.results()["failed"]` instead)
- `.wait(timeout: float = None)` - Wait for batch completion
- `.on_progress(callback, interval=3.0)` - Set progress monitoring callback
- `.shutdown(wait_for_active: bool = True)` - Gracefully shutdown

The progress callback receives three parameters:
1. `stats_dict`: Progress statistics containing:
   - `batch_id`: Current batch identifier
   - `total`: Total number of jobs
   - `pending`: Jobs waiting to start
   - `active`: Jobs currently processing
   - `completed`: Successfully completed jobs
   - `failed`: Failed jobs
   - `cancelled`: Cancelled jobs (e.g., via Ctrl+C)
   - `cost_usd`: Current total cost
   - `cost_limit_usd`: Cost limit (if set)
   - `is_complete`: Whether batch is finished
   - `batches_completed`: Number of completed batches
   - `batches_total`: Total number of batches
   - `batches_pending`: Number of pending batches
   - `items_per_batch`: Items per batch setting
2. `elapsed_time`: Time elapsed since batch started (in seconds)
3. `batch_data`: Dictionary mapping batch_id to batch information

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
├── job-ghi789.json
└── raw_files/
    └── responses/
        ├── job-abc123_raw.json
        ├── job-def456_raw.json
        └── job-ghi789_raw.json

./batch_state.json  # Batch state
```

## Supported Providers

| Feature | Anthropic | OpenAI |
|---------|-----------|--------|
| Models | [All Claude models](https://github.com/agamm/batchata/blob/main/batchata/providers/anthropic/models.py) | [All GPT models](https://github.com/agamm/batchata/blob/main/batchata/providers/openai/models.py) |
| Batch Discount | 50% | 50% |
| Polling Interval | 1s | 5s |
| Citations | ✅ | ❌ |
| Structured Output | ✅ | ✅ |
| File Types | PDF, TXT, DOCX, Images | PDF, Images |

## Configuration

Set your API keys as environment variables:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

You can also use a `.env` file in your project root (requires python-dotenv):
```python
from dotenv import load_dotenv
load_dotenv()

from batchata import Batch
# Your API keys will now be loaded from .env
```

## Limitations

- Field/citation mapping is heuristic, which means it isn't perfect.
- Citation mapping only works with flat Pydantic models (no nested BaseModel fields).
- No Gemini or Groq support yet.
- Cost tracking is not precise as the actual usage is only known after the batch is complete, try setting `items_per_batch` to a lower value for more accurate cost tracking.


## License

MIT License - see LICENSE file for details.
