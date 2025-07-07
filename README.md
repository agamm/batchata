# ai-batch

Batch processing for Anthropic's Claude API with structured output and PDF support.

## Features

- Batch processing of multiple messages
- Structured output with Pydantic models
- PDF file processing support
- Simple, clean API

## Installation

```bash
pip install ai-batch
```

## Usage

### Basic Text Processing

```python
from pydantic import BaseModel
from ai_batch import batch

class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str

messages = [
    [{"role": "user", "content": "Is this spam? You've won $1000!"}],
    [{"role": "user", "content": "Meeting at 3pm tomorrow"}],
]

results = batch(
    messages=messages,
    model="claude-3-haiku-20240307",
    response_model=SpamResult
)

for result in results:
    print(f"Spam: {result.is_spam} ({result.confidence:.0%})")
```

### PDF Processing

```python
from pydantic import BaseModel
from ai_batch import batch_files

class InvoiceData(BaseModel):
    invoice_number: str
    total_amount: float
    vendor_name: str

# Process multiple PDF files
results = batch_files(
    files=["invoice1.pdf", "invoice2.pdf", "invoice3.pdf"],
    prompt="Extract invoice data from this PDF",
    model="claude-3-5-sonnet-20241022",  # PDF support requires Sonnet or better
    response_model=InvoiceData
)

for invoice in results:
    print(f"Invoice {invoice.invoice_number}: ${invoice.total_amount}")
```

### Advanced PDF Processing with Document Blocks

```python
from ai_batch import batch, pdf_to_document_block

# Manual control over document blocks
pdf_bytes = open("document.pdf", "rb").read()
doc_block = pdf_to_document_block(pdf_bytes)

messages = [[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Summarize this document"},
        doc_block
    ]
}]]

results = batch(messages, model="claude-3-5-sonnet-20241022")
```

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key
```

Or use a `.env` file:

```
ANTHROPIC_API_KEY=your_api_key
```

## Examples

See the `examples/` directory for more usage examples:
- `spam_detection.py` - Email spam classification
- `pdf_extraction.py` - Extract data from PDFs