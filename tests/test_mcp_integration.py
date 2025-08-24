"""Integration test for MCP CLI workflow."""

import json
import subprocess
import tempfile
from pathlib import Path


def test_mcp_workflow():
    """Test the complete MCP CLI workflow."""
    print("=== MCP CLI Integration Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        state_dir = Path(temp_dir) / "mcp_state"
        
        # Test 1: List empty batches
        print("1. Testing empty list:")
        result = subprocess.run([
            "python", "-m", "batchata.cli.mcp",
            "--state-dir", str(state_dir),
            "list", "--format", "json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        batches = json.loads(result.stdout)
        assert batches == []
        print("   ✓ Empty list returns []")
        
        # Test 2: Try to get results for non-existent batch
        print("\n2. Testing results for non-existent batch:")
        result = subprocess.run([
            "python", "-m", "batchata.cli.mcp",
            "--state-dir", str(state_dir),
            "results", "non-existent-batch"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "not found" in result.stderr
        print("   ✓ Correctly handles non-existent batch")
        
        # Test 3: Try to cancel non-existent batch
        print("\n3. Testing cancel for non-existent batch:")
        result = subprocess.run([
            "python", "-m", "batchata.cli.mcp",
            "--state-dir", str(state_dir),
            "cancel", "non-existent-batch"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "not found" in result.stderr
        print("   ✓ Correctly handles non-existent batch for cancel")
        
        # Test 4: Try to create batch with invalid model (will fail but show validation)
        print("\n4. Testing create with invalid model:")
        result = subprocess.run([
            "python", "-m", "batchata.cli.mcp",
            "--state-dir", str(state_dir),
            "create",
            "--model", "invalid-model",
            "--messages", '[{"role": "user", "content": "test"}]'
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "No provider for model" in result.stderr
        print("   ✓ Correctly validates model names")
        
        # Test 5: Try to create batch without messages or file
        print("\n5. Testing create without required parameters:")
        result = subprocess.run([
            "python", "-m", "batchata.cli.mcp",
            "--state-dir", str(state_dir),
            "create",
            "--model", "claude-sonnet-4-20250514"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "Must provide either 'messages' or both 'file_path' and 'prompt'" in result.stderr
        print("   ✓ Correctly validates required parameters")
        
        print("\n=== All Integration Tests Passed! ===")


if __name__ == "__main__":
    test_mcp_workflow()