# Batchata

[![Tests](https://github.com/agamm/batchata/actions/workflows/test.yml/badge.svg)](https://github.com/agamm/batchata/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/batchata)](https://pypi.org/project/batchata/)

Python SDK for **AI batch processing** with structured output and citation mapping.

- **50% cost savings** via Anthropic's batch API pricing
- **Automatic cost tracking** with token usage and pricing
- **Structured output** with Pydantic models  
- **Field-level citations** map results to source documents
- **Type safety** with full validation

Currently supports Anthropic Claude. OpenAI support coming soon.

## Core Functions

- [`batch()`](#batch) - Process message conversations or PDF files
- [`BatchManager`](#batchmanager) - Manage large-scale AI batch processing with parallel execution

## Quick Start

```python
from batchata import batch
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
# Results now contain both data and citations together:
# [{"result": Invoice(...), "citations": {"company_name": [Citation(...)], ...}}, ...]
```

## Installation

```bash
pip install batchata
```

## Setup

Create a `.env` file in your project root:

```bash
ANTHROPIC_API_KEY=your-api-key
```

## API Reference

### batch()

Process multiple message conversations with optional structured output.

```python
from batchata import batch
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
from batchata import batch
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
# Results now contain both data and citations together
```

**Response:**
```python
# Results now contain both data and citations together
[
    {
        "result": Invoice(company_name="TechCorp Solutions", total_amount="$12,500.00", date="March 15, 2024"),
        "citations": {
            "company_name": [Citation(cited_text="TechCorp Solutions", start_page=1)],
            "total_amount": [Citation(cited_text="TOTAL: $12,500.00", start_page=2)],
            "date": [Citation(cited_text="Date: March 15, 2024", start_page=1)]
        }
    },
    {
        "result": Invoice(company_name="DataFlow Systems", total_amount="$8,750.00", date="March 18, 2024"),
        "citations": {
            "company_name": [Citation(cited_text="DataFlow Systems", start_page=1)],
            "total_amount": [Citation(cited_text="Total Due: $8,750.00", start_page=3)],
            "date": [Citation(cited_text="Invoice Date: March 18, 2024", start_page=1)]
        }
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

# Citations are now included in results (if enabled)
# Access via: results[0]["citations"]

# Save raw API responses
job = batch(..., raw_results_dir="./raw_responses")
```

### BatchManager

Manage large-scale batch processing with automatic job splitting, parallel execution, state persistence, and cost management.

```python
from batchata import BatchManager
from pydantic import BaseModel

class Invoice(BaseModel):
    company_name: str
    total_amount: float
    invoice_number: str

# Initialize BatchManager for large-scale processing
manager = BatchManager(
    files=["invoice1.pdf", "invoice2.pdf", ...],  # 100+ files
    prompt="Extract invoice data",
    model="claude-3-5-sonnet-20241022",
    response_model=Invoice,
    enable_citations=True,
    items_per_job=10,      # Process 10 files per job
    max_parallel_jobs=5,   # 5 jobs in parallel
    max_cost=50.0,         # Stop if cost exceeds $50
    state_path="batch_state.json",  # Auto-resume capability
    save_results_dir="results/"     # Save results to disk
)

# Run processing (auto-resumes if interrupted)
summary = manager.run(print_progress=True)

# Retry failed items
if summary['failed_items'] > 0:
    retry_summary = manager.retry_failed()

# Get statistics
stats = manager.stats
print(f"Completed: {stats['completed_items']}/{stats['total_items']}")
print(f"Total cost: ${stats['total_cost']:.2f}")

# Load results from disk
results = manager.get_results_from_disk()
```

**Key Features:**
- **Automatic job splitting**: Breaks large batches into smaller chunks
- **Parallel processing**: Multiple jobs run concurrently with ThreadPoolExecutor
- **State persistence**: Resume from interruptions with JSON state files
- **Cost management**: Stop processing when budget limit is reached
- **Progress monitoring**: Real-time progress updates with statistics
- **Retry mechanism**: Easily retry failed items
- **Result saving**: Organized directory structure for results

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

results = job.results()   # List of {"result": str, "citations": List[Citation]}

# Example result structure:
[
    {
        "result": "Summary text...",
        "citations": [
            Citation(cited_text="AI reduces errors by 30%", start_page=2),
            Citation(cited_text="Implementation cost: $50,000", start_page=5)
        ]
    }
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

results = job.results()   # List of {"result": Model, "citations": Dict[str, List[Citation]]}

# Example result structure:
[
    {
        "result": MyModel(title="Annual Report 2024", revenue="$1.2M"),
        "citations": {
            "title": [Citation(cited_text="Annual Report 2024", start_page=1)],
            "revenue": [Citation(cited_text="Revenue: $1.2M", start_page=3)],
            "growth": [Citation(cited_text="YoY Growth: 25%", start_page=3)]
        }
    }
]
```

The field mapping allows you to trace exactly which part of the source document was used to populate each field in your structured output.

### Robust Citation Parsing

Batchata uses proper JSON parsing for citation field mapping, ensuring reliability with complex JSON structures:

**Handles Complex Scenarios:**
- ✅ Escaped quotes in JSON values: `"name": "John \"The Great\" Doe"`
- ✅ URLs with colons: `"website": "http://example.com:8080"`
- ✅ Nested objects and arrays: `"metadata": {"nested": {"deep": "value"}}`
- ✅ Multi-line strings and special characters
- ✅ Fields with numbers/underscores: `user_name`, `age_2`

## Cost Tracking

Batchata automatically tracks token usage and costs for all batch operations:

```python
from batchata import batch

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
- [`examples/batch_manager_example.py`](examples/batch_manager_example.py) - Large-scale batch processing with BatchManager

## Limitations

- Citation mapping only works with flat Pydantic models (no nested models)
- OpenAI support coming soon
- PDFs require Opus/Sonnet models for best results
- Batch jobs can take up to 24 hours to process
- Use `job.is_complete()` to check status before getting results
- Citations may not be available in all batch API responses
- **Cost limits**: Best effort enforcement - costs are only known after job completion, so final costs may slightly exceed `max_cost` due to jobs already in progress

## Comparison with Alternatives

| Feature | batchata | LangChain | Instructor | PydanticAI |
|---------|----------|-----------|------------|------------|
| **Batch Requests** | ✅ Native (50% cost savings) | ❌ No native batch API | ✅ Via OpenAI Batch API ([#1092](https://github.com/instructor-ai/instructor/issues/1092)) | ⚠️ Planned ([#1771](https://github.com/pydantic/pydantic-ai/issues/1771)) |
| **Structured Output** | ✅ Full support | ✅ Via parsers | ✅ Core feature | ✅ Native |
| **PDF File Input** | ✅ Native support | ✅ Via document loaders | ✅ Via multimodal models | ✅ Via file handling |
| **Citation Mapping** | ✅ Field-level citations | ❌ Manual implementation | ❌ Manual implementation | ❌ Manual implementation |
| **Cost Tracking** | ✅ Automatic with tokencost | ❌ Manual implementation | ❌ Manual implementation | ❌ Manual implementation |
| **Cost Limits** | ✅ max_cost parameter | ❌ Manual implementation | ❌ Manual implementation | ❌ Manual implementation |
| **Batch Providers** | 2/2 (Anthropic, OpenAI planned) | 0/2 | 1/2 (OpenAI only) | 0/2 |
| **Focus** | Streamlined batch requests | General LLM orchestration | Structured outputs CLI | Agent framework |

## License

MIT

## Todos

- [x] ~~Add pricing metadata and max_spend controls~~ (Cost tracking implemented)
- [x] ~~Auto batch manager (parallel batches, retry, spend control)~~ (BatchManager implemented)
- [ ] Test mode to run on 1% sample before full batch
- [ ] Quick batch - split into smaller chunks for faster results
- [ ] Support text/other file types (not just PDFs)
- [ ] Support for OpenAI