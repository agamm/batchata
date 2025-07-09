# AI Batch

Python SDK for **batch processing** with structured output and citation mapping.

- **50% cost savings** via Anthropic's batch API pricing
- **Structured output** with Pydantic models  
- **Field-level citations** map results to source documents
- **Type safety** with full validation

Currently supports Anthropic Claude. OpenAI support coming soon.

## Installation

```bash
pip install ai-batch
```

## Quick Start

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

## Message Processing

For direct message processing:

```python
from ai_batch import batch

messages = [
    [{"role": "user", "content": "Is this spam? You've won $1000!"}],
    [{"role": "user", "content": "Meeting at 3pm tomorrow"}],
]

job = batch(
    messages=messages,
    model="claude-3-haiku-20240307",
    response_model=SpamResult
)

results = job.results()
```

## Setup

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## Examples

- `examples/citation_example.py` - Basic citation usage
- `examples/citation_with_pydantic.py` - Structured output with citations  
- `examples/spam_detection.py` - Email classification
- `examples/pdf_extraction.py` - PDF processing

## Limitations

- Citations only work with flat Pydantic models (no nested models)
- PDFs require Sonnet models for best results
- Batch jobs are asynchronous - call `job.results()` when ready