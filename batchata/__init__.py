"""Batchata - A batch processing library for AI models."""

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