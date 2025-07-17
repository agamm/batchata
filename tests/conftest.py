"""Shared pytest fixtures for batchata tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock

from batchata.types import Message
from batchata.core.job import Job


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's 2+2?"}
    ]


@pytest.fixture
def sample_job_params() -> Dict[str, Any]:
    """Sample job parameters for testing."""
    return {
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enable_citations": False
    }


@pytest.fixture
def sample_job(sample_messages, sample_job_params) -> Job:
    """Create a sample job for testing."""
    return Job(
        id="test-job-1",
        model=sample_job_params["model"],
        messages=sample_messages,
        temperature=sample_job_params["temperature"],
        max_tokens=sample_job_params["max_tokens"],
        enable_citations=sample_job_params["enable_citations"]
    )


@pytest.fixture
def batch_params(temp_dir) -> Dict[str, Any]:
    """Default batch parameters for testing."""
    return {
        "state_file": str(temp_dir / "state.json"),
        "results_dir": str(temp_dir / "results"),
        "max_parallel_batches": 5,
        "items_per_batch": 10,
        "reuse_state": True,
        "raw_files": True
    }


@pytest.fixture
def mock_provider_response() -> Dict[str, Any]:
    """Mock provider response for testing."""
    return {
        "id": "batch-123",
        "status": "complete",
        "results": [
            {
                "job_id": "test-job-1",
                "content": "4",
                "cost": 0.01,
                "input_tokens": 10,
                "output_tokens": 5,
                "error": None
            }
        ]
    }