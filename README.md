# AI Batch

Python SDK for **batch processing** with structured output and citation mapping.

- **50% cost savings** via Anthropic's batch API pricing
- **Structured output** with Pydantic models  
- **Field-level citations** map results to source documents
- **Type safety** with full validation

Currently supports Anthropic Claude. OpenAI support coming soon.

## API Reference

- [`batch()`](#message-processing) - Process message conversations
- [`batch_files()`](#file-processing) - Process PDF files
- [`BatchJob.results()`](#batch-job-methods) - Get processed results
- [`BatchJob.citations()`](#batch-job-methods) - Get citation mappings
- [`BatchJob.is_complete()`](#batch-job-methods) - Check completion status
- [`BatchJob.stats()`](#batch-job-methods) - Get processing statistics

## Installation

```bash
pip install ai-batch
```

## Quick Start

### Message Processing

```python
from ai_batch import batch
from pydantic import BaseModel

class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str

# Process multiple messages with structured output
job = batch(
    messages=[
        [{"role": "user", "content": "Is this spam? You've won $1000!"}],
        [{"role": "user", "content": "Meeting at 3pm tomorrow"}],
        [{"role": "user", "content": "URGENT: Click here now!"}]
    ],
    model="claude-3-haiku-20240307",
    response_model=SpamResult
)

# Wait for completion (can take up to 24 hours)
while not job.is_complete():
    time.sleep(30)
    
results = job.results()
```

### File Processing

```python
from ai_batch import batch_files
from pydantic import BaseModel

class Invoice(BaseModel):
    company_name: str
    total_amount: str
    date: str

# Process PDFs with structured output + citations
job = batch_files(
    files=["invoice1.pdf", "invoice2.pdf", "invoice3.pdf"],
    prompt="Extract the company name, total amount, and date.",
    model="claude-3-5-sonnet-20241022",
    response_model=Invoice,
    enable_citations=True
)

results = job.results()
citations = job.citations()
```

## Output

**Structured Results:**
```python
[
  Invoice(company_name="TechCorp Solutions Inc.", total_amount="$12,500.00", date="March 15, 2024"),
  Invoice(company_name="DataFlow Systems", total_amount="$8,750.00", date="March 18, 2024")
]
```

**Field-Level Citations:**
```python
[
  {
    "company_name": [Citation(cited_text="TechCorp Solutions Inc.", start_page=1)],
    "total_amount": [Citation(cited_text="TOTAL: $12,500.00", start_page=2)],
    "date": [Citation(cited_text="Date: March 15, 2024", start_page=1)]
  },
  # ... one dict per result
]
```

## Four Modes

| Response Model | Citations | Returns |
|---------------|-----------|---------|
| ❌ | ❌ | List of strings |
| ✅ | ❌ | List of Pydantic models |
| ❌ | ✅ | List of strings + flat citation list |
| ✅ | ✅ | List of Pydantic models + field citation dicts |

```python
# Mode 1: Text only
job = batch_files(files=["doc.pdf"], prompt="Summarize this")

# Mode 2: Structured only  
job = batch_files(files=["doc.pdf"], prompt="Extract data", response_model=MyModel)

# Mode 3: Text with citations
job = batch_files(files=["doc.pdf"], prompt="Analyze this", enable_citations=True)

# Mode 4: Structured with field citations
job = batch_files(files=["doc.pdf"], prompt="Extract data", 
                  response_model=MyModel, enable_citations=True)
```

## Raw Responses

Save raw API responses for debugging and analysis:

```python
# Save raw responses to directory
job = batch(
    messages=messages,
    model="claude-3-haiku-20240307",
    raw_results_dir="./raw_responses"
)

# Files saved as: {batch_id}_0.json, {batch_id}_1.json, etc.
results = job.results()
```

## Setup

Create a `.env` file in your project root:

```bash
ANTHROPIC_API_KEY=your-api-key
```

## Examples

- `examples/citation_example.py` - Basic citation usage
- `examples/citation_with_pydantic.py` - Structured output with citations  
- `examples/spam_detection.py` - Email classification
- `examples/pdf_extraction.py` - PDF processing

## BatchJob Methods

```python
# Check if batch processing is complete
if job.is_complete():
    results = job.results()

# Get processing statistics
stats = job.stats(print_stats=True)

# Get citations (if enabled)
citations = job.citations()
```

## Limitations

- Citations only work with flat Pydantic models (no nested models)
- PDFs require Sonnet models for best results
- Batch jobs are asynchronous and can take up to 24 hours to process
- Use `job.is_complete()` to check status before getting results