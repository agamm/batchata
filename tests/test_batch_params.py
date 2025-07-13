"""Tests for BatchConfig data model."""

import pytest
from unittest.mock import Mock

from batchata.core import BatchConfig, Job


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""
    
    def test_batch_config_creation(self):
        """Test creating BatchConfig."""
        config = BatchConfig(
            state_file="./state.json",
            results_dir="./results",
            max_concurrent=5
        )
        
        assert config.state_file == "./state.json"
        assert config.results_dir == "./results"
        assert config.max_concurrent == 5
        assert config.cost_limit_usd is None
        assert config.default_params == {}
        assert config.progress_callback is None
        assert config.jobs == []
    
    def test_batch_config_with_all_params(self):
        """Test BatchConfig with all parameters."""
        callback = Mock()
        config = BatchConfig(
            state_file="./state.json",
            results_dir="./results",
            max_concurrent=10,
            cost_limit_usd=50.0,
            default_params={"model": "claude-sonnet-4-20250514"},
            progress_callback=callback,
            jobs=[Job(id="test", model="claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hi"}])]
        )
        
        assert config.cost_limit_usd == 50.0
        assert config.default_params == {"model": "claude-sonnet-4-20250514"}
        assert config.progress_callback is callback
        assert len(config.jobs) == 1