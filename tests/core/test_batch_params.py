"""Tests for BatchParams class.

Testing:
1. Parameter validation and constraints
2. Serialization and deserialization
3. Default value handling
"""

import pytest
import json
from pathlib import Path

from batchata.core.batch_params import BatchParams
from batchata.exceptions import ValidationError


class TestBatchParams:
    """Test BatchParams configuration and validation."""
    
    def test_parameter_validation(self, temp_dir):
        """Test that invalid parameters are rejected."""
        results_dir = str(temp_dir / "results")
        
        # Valid params should work
        params = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=10,
            items_per_batch=20
        )
        assert params.max_parallel_batches == 10
        assert params.items_per_batch == 20
        
        # Test negative max_parallel_batches
        with pytest.raises(ValueError):
            BatchParams(
                state_file=str(temp_dir / "state.json"),
                results_dir=results_dir,
                max_parallel_batches=-1
            )
        
        # Test zero items_per_batch
        with pytest.raises(ValueError):
            BatchParams(
                state_file=str(temp_dir / "state.json"),
                results_dir=results_dir,
                max_parallel_batches=5,
                items_per_batch=0
            )
        
        # Test negative cost limit
        with pytest.raises(ValueError):
            BatchParams(
                state_file=str(temp_dir / "state.json"),
                results_dir=results_dir,
                max_parallel_batches=5,
                cost_limit_usd=-10.0
            )
    
    def test_serialization_and_deserialization(self, temp_dir):
        """Test params can be serialized to JSON and back."""
        results_dir = str(temp_dir / "results")
        
        original = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=5,
            items_per_batch=15,
            reuse_state=False,
            save_raw_responses=True,
            cost_limit_usd=25.0,
            default_params={"model": "test-model", "temperature": 0.5}
        )
        
        # Serialize to dict (using dataclass asdict)
        from dataclasses import asdict
        data = asdict(original)
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)
        
        # Recreate from dict
        restored = BatchParams(**loaded_data)
        
        assert restored.state_file == original.state_file
        assert restored.results_dir == original.results_dir
        assert restored.max_parallel_batches == original.max_parallel_batches
        assert restored.items_per_batch == original.items_per_batch
        assert restored.reuse_state == original.reuse_state
        assert restored.save_raw_responses == original.save_raw_responses
        assert restored.cost_limit_usd == original.cost_limit_usd
        assert restored.default_params == original.default_params
    
    def test_default_values(self, temp_dir):
        """Test that default values are correctly applied."""
        results_dir = str(temp_dir / "results")
        
        # Create with minimal params (max_parallel_batches is required)
        params = BatchParams(
            state_file=str(temp_dir / "state.json"),
            results_dir=results_dir,
            max_parallel_batches=5  # Required parameter
        )
        
        # Check defaults
        assert params.max_parallel_batches == 5
        assert params.items_per_batch == 10  # Default value
        assert params.reuse_state is True
        assert params.save_raw_responses is True
        assert params.cost_limit_usd is None
        assert params.default_params == {}
        assert params.verbosity == "info"