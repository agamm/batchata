"""
AI Batch Processing Library

A wrapper around Anthropic's batch API for structured output.
"""

from .core import batch, pdf_to_document_block
from .citations import Citation
from .batch_job import BatchJob
from .batch_manager import BatchManager

__all__ = ["batch", "pdf_to_document_block", "Citation", "BatchJob", "BatchManager"]