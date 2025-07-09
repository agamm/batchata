# AI Batch

Python SDK for **batch processing** with structured output and citation mapping.

- **50% cost savings** via Anthropic's batch API pricing
- **Automatic cost tracking** with token usage and pricing
- **Structured output** with Pydantic models  
- **Field-level citations** map results to source documents
- **Type safety** with full validation

Currently supports Anthropic Claude. OpenAI support coming soon.

## API Reference

- [`batch()`](#batch) - Process message conversations or PDF files
- [`BatchJob`](#batchjob) - Job status and results

## Quick Start

```python
from ai_batch import batch
from pydantic import BaseModel

class Invoice(BaseModel):
    company_name: str
    total_amount: str
    date: str

# Process PDFs with structured output + citations
job = batch(
    files=["invoice1.pdf", "invoice2.pdf", "invoice3.pdf"],
    prompt="Extract the company name, total amount, and date.",
    model="claude-3-5-sonnet-20241022",
    response_model=Invoice,
    enable_citations=True
)

# Wait for completion
while not job.is_complete():
    time.sleep(30)
    
results = job.results()
citations = job.citations()
```

## Installation

```bash
pip install ai-batch
```

## Usage

Create a `.env` file in your project root:

```bash
ANTHROPIC_API_KEY=your-api-key
```

## API Functions

### batch()

Process multiple message conversations with optional structured output.

```python
from ai_batch import batch
from pydantic import BaseModel

class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str

# Process messages
job = batch(
    messages=[
        [{"role": "user", "content": "Is this spam? You've won $1000!"}],
        [{"role": "user", "content": "Meeting at 3pm tomorrow"}],
        [{"role": "user", "content": "URGENT: Click here now!"}]
    ],
    model="claude-3-haiku-20240307",
    response_model=SpamResult
)

# Get results
results = job.results()
```

**Response:**
```python
[
    SpamResult(is_spam=True, confidence=0.95, reason="Contains monetary prize claim"),
    SpamResult(is_spam=False, confidence=0.98, reason="Normal meeting reminder"),
    SpamResult(is_spam=True, confidence=0.92, reason="Urgent call-to-action pattern")
]
```

### batch() with files

Process PDF files with optional structured output and citations.

```python
from ai_batch import batch
from pydantic import BaseModel

class Invoice(BaseModel):
    company_name: str
    total_amount: str
    date: str

# Process PDFs with citations
job = batch(
    files=["invoice1.pdf", "invoice2.pdf"],
    prompt="Extract the company name, total amount, and date.",
    model="claude-3-5-sonnet-20241022",
    response_model=Invoice,
    enable_citations=True
)

results = job.results()
citations = job.citations()
```

**Response:**
```python
# Results
[
    Invoice(company_name="TechCorp Solutions", total_amount="$12,500.00", date="March 15, 2024"),
    Invoice(company_name="DataFlow Systems", total_amount="$8,750.00", date="March 18, 2024")
]

# Citations (field-level mapping)
[
    {
        "company_name": [Citation(cited_text="TechCorp Solutions", start_page=1)],
        "total_amount": [Citation(cited_text="TOTAL: $12,500.00", start_page=2)],
        "date": [Citation(cited_text="Date: March 15, 2024", start_page=1)]
    },
    {
        "company_name": [Citation(cited_text="DataFlow Systems", start_page=1)],
        "total_amount": [Citation(cited_text="Total Due: $8,750.00", start_page=3)],
        "date": [Citation(cited_text="Invoice Date: March 18, 2024", start_page=1)]
    }
]
```

### BatchJob

The job object returned by `batch()`.

```python
# Check completion status
if job.is_complete():
    results = job.results()

# Get processing statistics with cost tracking
stats = job.stats(print_stats=True)
# Output:
# 📊 Batch Statistics
#    ID: msgbatch_01BPtdnmEwxtaDcdJ2eUsq4T
#    Status: ended
#    Complete: ✅
#    Elapsed: 41.8s
#    Mode: Text + Citations
#    Results: 0
#    Citations: 0
#    Input tokens: 2,117
#    Output tokens: 81
#    Total cost: $0.0038
#    (50% batch discount applied)

# Get citations (if enabled)
citations = job.citations()

# Save raw API responses
job = batch(..., raw_results_dir="./raw_responses")
```

## Citations

Citations work in two modes depending on whether you use structured output:

### 1. Text + Citations (Flat List)

When `enable_citations=True` without a response model, citations are returned as a flat list:

```python
job = batch(
    files=["document.pdf"],
    prompt="Summarize the key findings",
    enable_citations=True
)

results = job.results()   # List of strings
citations = job.citations()  # Flat list of Citation objects

# Example citations:
[
    Citation(cited_text="AI reduces errors by 30%", start_page=2),
    Citation(cited_text="Implementation cost: $50,000", start_page=5)
]
```

### 2. Structured + Field Citations (Mapping)

When using both `response_model` and `enable_citations=True`, citations are mapped to specific fields:

```python
job = batch(
    files=["document.pdf"],
    prompt="Extract the data",
    response_model=MyModel,
    enable_citations=True
)

results = job.results()   # List of Pydantic models
citations = job.citations()  # List of dicts mapping fields to citations

# Example field-level citations:
[
    {
        "title": [Citation(cited_text="Annual Report 2024", start_page=1)],
        "revenue": [Citation(cited_text="Revenue: $1.2M", start_page=3)],
        "growth": [Citation(cited_text="YoY Growth: 25%", start_page=3)]
    }
]
```

The field mapping allows you to trace exactly which part of the source document was used to populate each field in your structured output.

## Cost Tracking

AI Batch automatically tracks token usage and costs for all batch operations:

```python
from ai_batch import batch

job = batch(
    messages=[...],
    model="claude-3-5-sonnet-20241022"
)

# Get cost information
stats = job.stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Input tokens: {stats['total_input_tokens']:,}")
print(f"Output tokens: {stats['total_output_tokens']:,}")

# Or print formatted statistics
job.stats(print_stats=True)
```

## Example Scripts

- [`examples/spam_detection.py`](examples/spam_detection.py) - Email classification
- [`examples/pdf_extraction.py`](examples/pdf_extraction.py) - PDF data extraction
- [`examples/citation_example.py`](examples/citation_example.py) - Basic citation usage
- [`examples/citation_with_pydantic.py`](examples/citation_with_pydantic.py) - Structured output with citations

## Limitations

- Citationm mapping only work with flat Pydantic models (no nested models)
- No support for OpenAI.
- PDFs require Opus/Sonnet models for best results
- Batch jobs can take up to 24 hours to process
- Use `job.is_complete()` to check status before getting results
- Citations may not be available in all batch API responses

## License

MIT

## Todos

- [x] ~~Add pricing metadata and max_spend controls~~ (Cost tracking implemented)
- [ ] Auto batch manager (parallel batches, retry, spend control)
- [ ] Test mode to run on 1% sample before full batch
- [ ] Quick batch - split into smaller chunks for faster results
- [ ] Support text/other file types (not just PDFs)
- [ ] Support for OpenAI