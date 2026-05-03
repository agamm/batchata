"""Demo of MCP CLI functionality."""

import json
import tempfile
from pathlib import Path

from batchata.cli.mcp import MCPCommands


def demo_mcp_cli():
    """Demo the MCP CLI functionality."""
    print("=== Batchata MCP CLI Demo ===\n")
    
    # Use a temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        mcp = MCPCommands(state_dir=temp_dir)
        
        print("1. Listing empty batches:")
        batches = mcp.list()
        print(f"   Found {len(batches)} batches\n")
        
        # Create some demo metadata files to simulate batches
        demo_batch_id = "demo-batch-12345"
        metadata = {
            "batch_id": demo_batch_id,
            "model": "claude-sonnet-4-20250514",
            "created_at": "2025-01-26T15:30:00",
            "state_file": str(mcp.batches_dir / f"{demo_batch_id}_state.json"),
            "results_dir": str(mcp.batches_dir / demo_batch_id),
            "kwargs": {"temperature": 0.7, "max_tokens": 1000}
        }
        
        metadata_file = mcp.batches_dir / f"{demo_batch_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("2. Listing batches after creating demo metadata:")
        batches = mcp.list()
        print(f"   Found {len(batches)} batches")
        if batches:
            batch = batches[0]
            print(f"   - Batch ID: {batch['batch_id']}")
            print(f"   - Model: {batch['model']}")
            print(f"   - Created: {batch['created_at']}")
            print(f"   - Status: {batch['status']}")
        
        print("\n3. Testing results for demo batch:")
        try:
            results = mcp.results(demo_batch_id)
            print(f"   Results: {json.dumps(results, indent=2)}")
        except Exception as e:
            print(f"   Expected error (no state file): {e}")
        
        print("\n4. Testing validation:")
        try:
            mcp.create(model="test-model")  # Should fail - no messages or file
        except ValueError as e:
            print(f"   Expected validation error: {e}")
        
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_mcp_cli()