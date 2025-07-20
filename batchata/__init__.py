"""Batchata - Unified Python API for AI Batch requests with cost tracking, Pydantic responses, and parallel execution.

**Why AI-batching?**

AI providers offer batch APIs that process requests asynchronously at 50% reduced cost compared to real-time APIs. 
This is ideal for workloads like document processing, data analysis, and content generation where immediate 
responses aren't required.

## Quick Start

### Installation

```bash
pip install batchata
```

### Basic Usage

```python
from batchata import Batch

# Simple batch processing
batch = Batch(results_dir="./output")
    .set_default_params(model="claude-sonnet-4-20250514")
    .add_cost_limit(usd=5.0)

# Add jobs
for file in files:
    batch.add_job(file=file, prompt="Summarize this document")

# Execute
run = batch.run()
results = run.results()
```

### Structured Output with Pydantic

```python
from batchata import Batch
from pydantic import BaseModel

class DocumentAnalysis(BaseModel):
    title: str
    summary: str
    key_points: list[str]

batch = Batch(results_dir="./results")
    .set_default_params(model="claude-sonnet-4-20250514")

batch.add_job(
    file="document.pdf",
    prompt="Analyze this document",
    response_model=DocumentAnalysis,
    enable_citations=True  # Anthropic only
)

run = batch.run()
for result in run.results()["completed"]:
    analysis = result.parsed_response  # DocumentAnalysis object
    citations = result.citation_mappings  # Field -> Citation mapping
```

## Key Features

- **50% Cost Savings**: Native batch processing via provider APIs
- **Cost Limits**: Set `max_cost_usd` limits for batch requests  
- **Time Limits**: Control execution time with `.add_time_limit()`
- **State Persistence**: Resume interrupted batches automatically
- **Structured Output**: Pydantic models with automatic validation
- **Citations**: Extract and map citations to response fields (Anthropic)
- **Multiple Providers**: Anthropic Claude and OpenAI GPT models

## Supported Providers

| Feature | Anthropic | OpenAI |
|---------|-----------|--------|
| Models | [All Claude models](https://github.com/agamm/batchata/blob/main/batchata/providers/anthropic/models.py) | [All GPT models](https://github.com/agamm/batchata/blob/main/batchata/providers/openai/models.py) |
| Citations | ✅ | ❌ |
| Structured Output | ✅ | ✅ |
| File Types | PDF, TXT, DOCX, Images | PDF, Images |

## Configuration

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

Or use a `.env` file with python-dotenv.
"""

from .core import Batch, BatchRun, Job, JobResult
from .exceptions import (
    BatchataError,
    CostLimitExceededError,
    ProviderError,
    ProviderNotFoundError,
    ValidationError,
)
from .types import Citation

__version__ = "0.3.0"

__all__ = [
    "Batch",
    "BatchRun", 
    "Job",
    "JobResult",
    "Citation",
    "BatchataError",
    "CostLimitExceededError",
    "ProviderError",
    "ProviderNotFoundError",
    "ValidationError",
]