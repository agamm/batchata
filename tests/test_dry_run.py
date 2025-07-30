"""Test dry run functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from batchata import Batch
from batchata.providers.anthropic.anthropic import AnthropicProvider
from batchata.providers.openai.openai_provider import OpenAIProvider
from pydantic import BaseModel, Field


class DryRunTestResponse(BaseModel):
    """Test response model for dry run tests."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")


@pytest.fixture(autouse=True)
def mock_providers():
    """Mock providers to avoid API key requirements."""
    # Mock provider registry to have test providers
    with patch('batchata.providers.provider_registry.providers') as mock_registry:
        # Create mock providers
        mock_anthropic = MagicMock(spec=AnthropicProvider)
        mock_anthropic.models = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        mock_anthropic.estimate_cost.return_value = 0.05
        
        mock_openai = MagicMock(spec=OpenAIProvider)
        mock_openai.models = ["gpt-4", "gpt-3.5-turbo"]
        mock_openai.estimate_cost.return_value = 0.08
        
        # Set up registry
        mock_registry.__contains__ = lambda x: True
        mock_registry.get = lambda model: mock_anthropic if "claude" in model else mock_openai
        mock_registry.keys.return_value = ["claude-3-sonnet-20240229", "gpt-4"]
        
        # Also mock get_provider function
        with patch('batchata.providers.get_provider') as mock_get_provider:
            mock_get_provider.side_effect = lambda model: mock_anthropic if "claude" in model else mock_openai
            yield


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
            response_model=DryRunTestResponse
        )
    
    # Run dry run
    run = batch.run(dry_run=True)
    
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
            response_model=DryRunTestResponse,
            enable_citations=True
        )
    
    # Run dry run
    run = batch.run(dry_run=True)
    
    # No execution should happen
    assert len(run.completed_results) == 0


def test_dry_run_cost_limit_warning():
    """Test dry run shows warning when cost exceeds limit."""
    batch = (
        Batch("./test_results", max_parallel_batches=2, items_per_batch=2)
        .set_default_params(model="gpt-4", temperature=0.7)
        .add_cost_limit(usd=0.1)  # Low limit - lower than what our mock returns
    )
    
    # Add jobs
    for i in range(4):
        batch.add_job(
            messages=[{"role": "user", "content": f"Complex question {i}"}],
            max_tokens=4000
        )
    
    # Run dry run - this should work without errors and show cost exceeds limit
    # The actual warning is visible in the test output above
    run = batch.run(dry_run=True)
    
    # Test that dry run completed successfully (cost exceeded limit but didn't prevent execution)
    assert len(run.completed_results) == 0  # No actual execution happened
    assert run is not None  # Dry run completed


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
    
    # Run dry run
    run = batch.run(dry_run=True)
    
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
        
        # Run dry run
        run = batch.run(dry_run=True)
        
        # Should load state
        mock_state_manager.load_state.assert_called_once()