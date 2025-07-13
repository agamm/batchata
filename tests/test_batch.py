"""Tests for Batch builder."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from pydantic import BaseModel

from batchata.core import Batch


class SampleModel(BaseModel):
    """Sample model for testing."""
    name: str
    value: int


class TestBatch:
    """Tests for Batch builder."""
    
    def test_batch_initialization(self):
        """Test basic batch initialization."""
        batch = Batch("./state.json", "./results", max_concurrent=5)
        
        assert batch.config.state_file == "./state.json"
        assert batch.config.results_dir == "./results"
        assert batch.config.max_concurrent == 5
        assert len(batch) == 0
        assert len(batch) == 0
    
    def test_batch_defaults(self):
        """Test setting default parameters."""
        batch = Batch("./state", "./results")
        batch.defaults(model="claude-3-sonnet", temperature=0.5, max_tokens=500)
        
        assert batch.config.default_params["model"] == "claude-3-sonnet"
        assert batch.config.default_params["temperature"] == 0.5
        assert batch.config.default_params["max_tokens"] == 500
    
    def test_batch_defaults_chaining(self):
        """Test method chaining with defaults."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="gpt-4")
            .defaults(temperature=0.7)
        )
        
        assert batch.config.default_params["model"] == "gpt-4"
        assert batch.config.default_params["temperature"] == 0.7
    
    def test_add_cost_limit(self):
        """Test adding cost limit."""
        batch = Batch("./state", "./results")
        batch.add_cost_limit(usd=100.0)
        
        assert batch.config.cost_limit_usd == 100.0
    
    def test_add_cost_limit_invalid(self):
        """Test adding invalid cost limit."""
        batch = Batch("./state", "./results")
        
        with pytest.raises(ValueError, match="Cost limit must be positive"):
            batch.add_cost_limit(usd=0)
        
        with pytest.raises(ValueError, match="Cost limit must be positive"):
            batch.add_cost_limit(usd=-10)
    
    def test_on_progress(self):
        """Test setting progress callback."""
        batch = Batch("./state", "./results")
        callback = Mock()
        
        batch.on_progress(callback)
        assert batch.config.progress_callback is callback
    
    def test_add_job_with_messages(self):
        """Test adding job with messages."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="claude-3-sonnet")
            .add_job(messages=[{"role": "user", "content": "Hello"}])
        )
        
        assert len(batch) == 1
        job = batch.config.jobs[0]
        assert job.id.startswith("job-")
        assert job.model == "claude-3-sonnet"
        assert job.messages == [{"role": "user", "content": "Hello"}]
    
    def test_add_job_with_file(self):
        """Test adding job with file input."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="gpt-4", temperature=0.7)
            .add_job(file="document.pdf", prompt="Summarize this document")
        )
        
        assert len(batch) == 1
        job = batch.config.jobs[0]
        assert job.file == Path("document.pdf")
        assert job.prompt == "Summarize this document"
        assert job.model == "gpt-4"
        assert job.temperature == 0.7
    
    def test_add_job_override_defaults(self):
        """Test adding job with overridden defaults."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="claude-3-sonnet", temperature=0.7, max_tokens=1000)
            .add_job(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                temperature=0.5
            )
        )
        
        job = batch.config.jobs[0]
        assert job.model == "gpt-4"  # Overridden
        assert job.temperature == 0.5  # Overridden
        assert job.max_tokens == 1000  # From defaults
    
    def test_add_job_structured_output(self):
        """Test adding job with structured output."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="claude-3-sonnet")
            .add_job(
                messages=[{"role": "user", "content": "Extract data"}],
                response_model=SampleModel,
                enable_citations=True
            )
        )
        
        job = batch.config.jobs[0]
        assert job.response_model == SampleModel
        assert job.enable_citations is True
    
    def test_add_job_no_model(self):
        """Test error when no model provided."""
        batch = Batch("./state", "./results")
        
        with pytest.raises(ValueError, match="Model must be provided"):
            batch.add_job(messages=[{"role": "user", "content": "Hi"}])
    
    def test_add_multiple_jobs(self):
        """Test adding multiple jobs."""
        batch = (
            Batch("./state", "./results")
            .defaults(model="claude-3-sonnet")
            .add_job(messages=[{"role": "user", "content": "Job 1"}])
            .add_job(messages=[{"role": "user", "content": "Job 2"}])
            .add_job(file="doc.pdf", prompt="Summarize")
        )
        
        assert len(batch) == 3
        assert all(job.id.startswith("job-") for job in batch.config.jobs)
        assert len(set(job.id for job in batch.config.jobs)) == 3  # All unique IDs
    
    
    def test_batch_repr(self):
        """Test string representation."""
        batch = (
            Batch("./state", "./results", max_concurrent=5)
            .add_cost_limit(usd=50.0)
            .defaults(model="claude-3-sonnet")
            .add_job(messages=[{"role": "user", "content": "Test"}])
        )
        
        repr_str = repr(batch)
        assert "Batch(jobs=1" in repr_str
        assert "max_concurrent=5" in repr_str
        assert "cost_limit=$50.0" in repr_str
    
    def test_batch_repr_no_cost_limit(self):
        """Test string representation without cost limit."""
        batch = Batch("./state", "./results")
        
        repr_str = repr(batch)
        assert "cost_limit=$None" in repr_str
    
    def test_run_no_jobs(self):
        """Test running batch with no jobs."""
        batch = Batch("./state", "./results")
        
        with pytest.raises(ValueError, match="No jobs added to batch"):
            batch.run()
    
    def test_fluent_interface(self):
        """Test complete fluent interface."""
        callback = Mock()
        
        batch = (
            Batch("./state", "./results", max_concurrent=10)
            .defaults(model="claude-3-sonnet", temperature=0.7)
            .add_cost_limit(usd=100.0)
            .on_progress(callback)
            .add_job(messages=[{"role": "user", "content": "Job 1"}])
            .add_job(
                file="document.pdf",
                prompt="Summarize this",
                model="gpt-4",  # Override default
                max_tokens=2000
            )
            .add_job(
                messages=[{"role": "assistant", "content": "Previous"}, {"role": "user", "content": "Continue"}],
                response_model=SampleModel,
                enable_citations=True
            )
        )
        
        # Verify configuration
        assert batch.config.max_concurrent == 10
        assert batch.config.cost_limit_usd == 100.0
        assert batch.config.progress_callback is callback
        assert len(batch) == 3
        
        # Verify jobs
        job1, job2, job3 = batch.config.jobs
        
        # Job 1: uses defaults
        assert job1.model == "claude-3-sonnet"
        assert job1.temperature == 0.7
        assert job1.messages == [{"role": "user", "content": "Job 1"}]
        
        # Job 2: file input with overrides
        assert job2.model == "gpt-4"
        assert job2.file == Path("document.pdf")
        assert job2.prompt == "Summarize this"
        assert job2.max_tokens == 2000
        assert job2.temperature == 0.7  # From defaults
        
        # Job 3: structured output with citations
        assert job3.model == "claude-3-sonnet"
        assert len(job3.messages) == 2
        assert job3.response_model == SampleModel
        assert job3.enable_citations is True