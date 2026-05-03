"""MCP (Model Context Protocol) CLI for batch requests."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4

from batchata import Batch
from batchata.utils.state import StateManager, BatchState
from batchata.core.batch_run import BatchRun
from batchata.core.job_result import JobResult


class MCPCommands:
    """MCP command implementations."""
    
    def __init__(self, state_dir: str = "./.batchata"):
        """Initialize MCP commands with state directory."""
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.batches_dir = self.state_dir / "batches"
        self.batches_dir.mkdir(exist_ok=True)
    
    def create(self, model: str, messages: Optional[List[Dict]] = None, 
               file_path: Optional[str] = None, prompt: Optional[str] = None,
               **kwargs) -> str:
        """Create a new batch request.
        
        Args:
            model: Model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o-2024-08-06")
            messages: Array of messages in format [{"role": "user", "content": "..."}]
            file_path: Path to file to process
            prompt: Prompt to use with file
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        
        Returns:
            Batch ID
        """
        batch_id = str(uuid4())
        results_dir = self.batches_dir / batch_id
        
        # Create batch
        batch = Batch(results_dir=str(results_dir))
        batch.set_default_params(model=model, **kwargs)
        batch.set_state(file=str(self.batches_dir / f"{batch_id}_state.json"), reuse_state=False)
        
        # Add job
        if messages:
            batch.add_job(messages=messages)
        elif file_path and prompt:
            batch.add_job(file=file_path, prompt=prompt)
        else:
            raise ValueError("Must provide either 'messages' or both 'file_path' and 'prompt'")
        
        # Start batch (may fail due to missing API keys in test environment)
        try:
            run = batch.run()
        except Exception as e:
            # Store batch metadata anyway for testing purposes
            metadata = {
                "batch_id": batch_id,
                "model": model,
                "created_at": None,
                "state_file": str(self.batches_dir / f"{batch_id}_state.json"),
                "results_dir": str(results_dir),
                "kwargs": kwargs,
                "error": str(e)
            }
            
            with open(self.batches_dir / f"{batch_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            raise e
        
        # Store batch metadata
        metadata = {
            "batch_id": batch_id,
            "model": model,
            "created_at": run.start_time.isoformat() if run.start_time else None,
            "state_file": str(self.batches_dir / f"{batch_id}_state.json"),
            "results_dir": str(results_dir),
            "kwargs": kwargs
        }
        
        with open(self.batches_dir / f"{batch_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return batch_id
    
    def list(self) -> List[Dict]:
        """List all batch requests.
        
        Returns:
            List of batch metadata
        """
        batches = []
        for metadata_file in self.batches_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                    # Add status information
                    state_file = metadata.get("state_file")
                    if state_file and Path(state_file).exists():
                        state_manager = StateManager(state_file)
                        state = state_manager.load()
                        if state:
                            metadata["status"] = {
                                "pending_jobs": len(state.pending_jobs),
                                "active_batches": len(state.active_batches),
                                "completed_results": len(state.completed_results),
                                "failed_jobs": len(state.failed_jobs),
                                "total_cost_usd": state.total_cost_usd
                            }
                        else:
                            metadata["status"] = "no_state"
                    else:
                        metadata["status"] = "unknown"
                    
                    batches.append(metadata)
            except Exception as e:
                # Skip corrupted metadata files
                print(f"Warning: Could not load metadata from {metadata_file}: {e}", file=sys.stderr)
                continue
        
        return sorted(batches, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def results(self, batch_id: str) -> Dict:
        """Get results for a specific batch.
        
        Args:
            batch_id: Batch ID to get results for
            
        Returns:
            Results dictionary with completed, failed, and cancelled results
        """
        metadata_file = self.batches_dir / f"{batch_id}_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Batch {batch_id} not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        state_file = metadata.get("state_file")
        results_dir = metadata.get("results_dir")
        
        if not state_file or not Path(state_file).exists():
            return {"completed": [], "failed": [], "cancelled": [], "error": "No state file found"}
        
        # Load state and reconstruct results
        state_manager = StateManager(state_file)
        state = state_manager.load()
        
        if not state:
            return {"completed": [], "failed": [], "cancelled": [], "error": "No state found"}
        
        results = {"completed": [], "failed": [], "cancelled": []}
        
        # Load completed results
        for result_ref in state.completed_results:
            try:
                result_file = Path(result_ref["file_path"])
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    results["completed"].append(result_data)
            except Exception as e:
                print(f"Warning: Could not load result {result_ref}: {e}", file=sys.stderr)
        
        # Add failed jobs
        for job_data in state.failed_jobs:
            results["failed"].append(job_data)
        
        return results
    
    def cancel(self, batch_id: str) -> bool:
        """Cancel a running batch.
        
        Args:
            batch_id: Batch ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        metadata_file = self.batches_dir / f"{batch_id}_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Batch {batch_id} not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        state_file = metadata.get("state_file")
        
        if not state_file or not Path(state_file).exists():
            return False
        
        # Load state and check if there are active batches
        state_manager = StateManager(state_file)
        state = state_manager.load()
        
        if not state or not state.active_batches:
            print(f"No active batches found for {batch_id}")
            return False
        
        # For now, we'll clear the active batches and mark pending jobs as cancelled
        # In a real implementation, you'd call provider APIs to cancel
        cancelled_jobs = []
        for job_data in state.pending_jobs:
            cancelled_jobs.append({
                **job_data,
                "error": "Cancelled by user",
                "status": "cancelled"
            })
        
        # Update state
        state.pending_jobs = []
        state.active_batches = []
        state.failed_jobs.extend(cancelled_jobs)
        
        state_manager.save(state)
        
        print(f"Cancelled {len(cancelled_jobs)} pending jobs for batch {batch_id}")
        return True


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for MCP CLI."""
    parser = argparse.ArgumentParser(
        description="MCP (Model Context Protocol) CLI for batch requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create batch with messages
  batchata-mcp create --model claude-sonnet-4-20250514 --messages '[{"role": "user", "content": "Hello"}]'
  
  # Create batch with file and prompt
  batchata-mcp create --model gpt-4o-2024-08-06 --file document.pdf --prompt "Summarize this document"
  
  # Add temperature and max tokens
  batchata-mcp create --model claude-sonnet-4-20250514 --messages '[{"role": "user", "content": "Hello"}]' --temperature 0.7 --max_tokens 1000
  
  # List all batches
  batchata-mcp list
  
  # Get results for a batch
  batchata-mcp results <batch_id>
  
  # Cancel a batch
  batchata-mcp cancel <batch_id>
        """
    )
    
    parser.add_argument(
        "--state-dir",
        default="./.batchata",
        help="Directory to store batch state (default: ./.batchata)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new batch request")
    create_parser.add_argument("--model", required=True, help="Model to use")
    create_parser.add_argument("--messages", help="JSON array of messages")
    create_parser.add_argument("--file", help="File path to process")
    create_parser.add_argument("--prompt", help="Prompt to use with file")
    create_parser.add_argument("--temperature", type=float, help="Temperature parameter")
    create_parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    create_parser.add_argument("--max-output-tokens", type=int, help="Maximum output tokens")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all batch requests")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Get results for a batch")
    results_parser.add_argument("batch_id", help="Batch ID to get results for")
    results_parser.add_argument("--format", choices=["table", "json"], default="json", help="Output format")
    
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running batch")
    cancel_parser.add_argument("batch_id", help="Batch ID to cancel")
    
    return parser


def main():
    """Main entry point for MCP CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    mcp = MCPCommands(state_dir=args.state_dir)
    
    try:
        if args.command == "create":
            # Parse messages if provided
            messages = None
            if args.messages:
                try:
                    messages = json.loads(args.messages)
                except json.JSONDecodeError as e:
                    print(f"Error parsing messages JSON: {e}", file=sys.stderr)
                    sys.exit(1)
            
            # Collect kwargs
            kwargs = {}
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature
            if args.max_tokens is not None:
                kwargs["max_tokens"] = args.max_tokens
            if args.max_output_tokens is not None:
                kwargs["max_output_tokens"] = args.max_output_tokens
            
            batch_id = mcp.create(
                model=args.model,
                messages=messages,
                file_path=args.file,
                prompt=args.prompt,
                **kwargs
            )
            print(f"Created batch: {batch_id}")
        
        elif args.command == "list":
            batches = mcp.list()
            
            if args.format == "json":
                print(json.dumps(batches, indent=2))
            else:
                # Table format
                if not batches:
                    print("No batches found")
                    return
                
                print(f"{'Batch ID':<36} {'Model':<25} {'Created':<20} {'Status'}")
                print("-" * 90)
                
                for batch in batches:
                    batch_id = batch.get("batch_id", "unknown")[:35]
                    model = batch.get("model", "unknown")[:24]
                    created = batch.get("created_at", "unknown")[:19]
                    
                    status = batch.get("status", {})
                    if isinstance(status, dict):
                        pending = status.get("pending_jobs", 0)
                        completed = status.get("completed_results", 0)
                        failed = status.get("failed_jobs", 0)
                        status_str = f"P:{pending} C:{completed} F:{failed}"
                    else:
                        status_str = str(status)
                    
                    print(f"{batch_id:<36} {model:<25} {created:<20} {status_str}")
        
        elif args.command == "results":
            results = mcp.results(args.batch_id)
            
            if args.format == "json":
                print(json.dumps(results, indent=2))
            else:
                # Table format
                completed = results.get("completed", [])
                failed = results.get("failed", [])
                cancelled = results.get("cancelled", [])
                
                print(f"Results for batch {args.batch_id}:")
                print(f"  Completed: {len(completed)}")
                print(f"  Failed: {len(failed)}")
                print(f"  Cancelled: {len(cancelled)}")
                
                if completed:
                    print("\nCompleted jobs:")
                    for i, result in enumerate(completed):
                        print(f"  {i+1}. Job {result.get('job_id', 'unknown')}")
                
                if failed:
                    print("\nFailed jobs:")
                    for i, result in enumerate(failed):
                        print(f"  {i+1}. Job {result.get('job_id', 'unknown')}: {result.get('error', 'Unknown error')}")
        
        elif args.command == "cancel":
            success = mcp.cancel(args.batch_id)
            if success:
                print(f"Successfully cancelled batch {args.batch_id}")
            else:
                print(f"Failed to cancel batch {args.batch_id}")
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()