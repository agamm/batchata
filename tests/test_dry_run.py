"""Test dry run functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from batchata import Batch
from pydantic import BaseModel, Field


class TestResponse(BaseModel):
    """Test response model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")


def test_dry_run_no_jobs():
    """Test dry run with no jobs."""
    batch = Batch("./test_results", max_parallel_batches=3, items_per_batch=2)
    
    with pytest.raises(ValueError, match="No jobs added to batch"):
        batch.run(dry_run=True)


def test_dry_run_with_messages():
    """Test dry run with message-based jobs."""
    batch = (
        Batch("./test_results", max_parallel_batches=3, items_per_batch=2)
        .set_default_params(model="claude-3-sonnet-20240229", temperature=0.7)
        .add_cost_limit(usd=10.0)
    )
    
    # Add some jobs
    for i in range(5):
        batch.add_job(
            messages=[{"role": "user", "content": f"Question {i}"}],
            response_model=TestResponse
        )
    
    # Mock the provider's estimate_cost method
    with patch('batchata.providers.anthropic.anthropic.AnthropicProvider.estimate_cost') as mock_estimate:
        mock_estimate.return_value = 0.05  # $0.05 per batch
        
        # Run dry run
        run = batch.run(dry_run=True)
        
        # Should have called estimate_cost for each batch
        assert mock_estimate.call_count == 3  # 5 jobs / 2 items_per_batch = 3 batches
        
        # Check that no actual execution happened
        assert len(run.completed_results) == 0
        assert len(run.failed_jobs) == 0


def test_dry_run_with_files():
    """Test dry run with file-based jobs."""
    batch = (
        Batch("./test_results", max_parallel_batches=2, items_per_batch=1, raw_files=True)
        .set_default_params(model="claude-3-sonnet-20240229", temperature=0.7)
        .add_cost_limit(usd=5.0)
    )
    
    # Add file jobs
    test_files = [Path(f"test_{i}.pdf") for i in range(3)]
    for i, file_path in enumerate(test_files):
        batch.add_job(
            file=file_path,
            prompt=f"Analyze document {i}",
            response_model=TestResponse,
            enable_citations=True
        )
    
    # Mock the provider methods
    with patch('batchata.providers.anthropic.anthropic.AnthropicProvider.estimate_cost') as mock_estimate:
        mock_estimate.return_value = 2.0  # $2 per job (high cost to test limit)
        
        # Run dry run
        run = batch.run(dry_run=True)
        
        # Should estimate cost for all batches
        assert mock_estimate.call_count == 3  # 3 jobs with items_per_batch=1
        
        # No execution should happen
        assert len(run.completed_results) == 0


def test_dry_run_cost_limit_warning(caplog):
    """Test dry run shows warning when cost exceeds limit."""
    batch = (
        Batch("./test_results", max_parallel_batches=2, items_per_batch=2)
        .set_default_params(model="gpt-4", temperature=0.7)
        .add_cost_limit(usd=1.0)  # Low limit
    )
    
    # Add jobs
    for i in range(4):
        batch.add_job(
            messages=[{"role": "user", "content": f"Complex question {i}"}],
            max_tokens=4000
        )
    
    # Mock high cost estimation
    with patch('batchata.providers.openai.openai_provider.OpenAIProvider.estimate_cost') as mock_estimate:
        mock_estimate.return_value = 1.5  # $1.5 per batch (exceeds limit)
        
        # Run dry run
        run = batch.run(dry_run=True)
        
        # Check for warning in logs
        assert "Estimated cost exceeds limit" in caplog.text
        assert mock_estimate.call_count == 2  # 4 jobs / 2 items_per_batch = 2 batches


def test_dry_run_mixed_providers():
    """Test dry run with mixed providers."""
    batch = Batch("./test_results", max_parallel_batches=3, items_per_batch=2)
    
    # Add jobs for different providers
    batch.add_job(
        messages=[{"role": "user", "content": "Claude question"}],
        model="claude-3-sonnet-20240229"
    )
    batch.add_job(
        messages=[{"role": "user", "content": "GPT question"}],
        model="gpt-4"
    )
    batch.add_job(
        messages=[{"role": "user", "content": "Another Claude question"}],
        model="claude-3-sonnet-20240229"
    )
    
    # Mock both providers
    with patch('batchata.providers.anthropic.anthropic.AnthropicProvider.estimate_cost') as mock_claude_estimate, \
         patch('batchata.providers.openai.openai_provider.OpenAIProvider.estimate_cost') as mock_gpt_estimate:
        
        mock_claude_estimate.return_value = 0.03
        mock_gpt_estimate.return_value = 0.08
        
        # Run dry run
        run = batch.run(dry_run=True)
        
        # Should call each provider's estimate
        assert mock_claude_estimate.call_count == 1  # 2 Claude jobs in 1 batch
        assert mock_gpt_estimate.call_count == 1    # 1 GPT job in 1 batch
        
        # No actual execution
        assert len(run.completed_results) == 0


def test_dry_run_with_state_reuse():
    """Test dry run respects existing state."""
    # Create a mock state manager
    mock_state_manager = MagicMock()
    
    batch = (
        Batch("./test_results", max_parallel_batches=2, items_per_batch=2)
        .set_state(file="./test_state.json", reuse_state=True)
        .set_default_params(model="claude-3-sonnet-20240229")
    )
    
    # Add jobs
    for i in range(3):
        batch.add_job(messages=[{"role": "user", "content": f"Question {i}"}])
    
    with patch('batchata.core.batch_run.StateManager') as mock_state_class:
        mock_state_class.return_value = mock_state_manager
        
        # Simulate some jobs already completed
        def load_state_side_effect(run):
            run.completed_results = {"job-0": MagicMock()}
        
        mock_state_manager.load_state.side_effect = load_state_side_effect
        
        with patch('batchata.providers.anthropic.anthropic.AnthropicProvider.estimate_cost') as mock_estimate:
            mock_estimate.return_value = 0.05
            
            # Run dry run
            run = batch.run(dry_run=True)
            
            # Should load state
            mock_state_manager.load_state.assert_called_once()
            
            # Should only estimate for pending jobs (2 out of 3)
            # 2 jobs with items_per_batch=2 = 1 batch
            assert mock_estimate.call_count == 1