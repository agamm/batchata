"""Simple demo of Batchata API."""

from batchata import Batch
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Analysis(BaseModel):
    """Structured output for analysis."""
    summary: str
    sentiment: str
    key_points: list[str]


def main():
    """Run a simple batch processing demo."""
    # Create batch configuration
    batch = (
        Batch(state_file="./examples/demo_state.json", results_dir="./examples/output", max_concurrent=1, items_per_batch=3)
        .defaults(model="claude-sonnet-4-20250514", temperature=0.7)
        .add_cost_limit(usd=5.0)
    )
    
    # Add some jobs
    texts = [
        "The new product launch was highly successful with record sales.",
        "Customer complaints have increased significantly this quarter.",
        "Market research shows growing demand for sustainable products."
    ]
    
    for _, text in enumerate(texts):
        batch.add_job(
            messages=[{"role": "user", "content": f"Analyze this business update: {text}"}],
            response_model=Analysis
        )
    
    # Execute batch
    print("Starting batch processing...")
    run = batch.run(wait=True, on_progress=lambda s, t: print(f"\rProgress: {s['completed']}/{s['total']}, {round(t, 2)}s"))
    
    # Get results
    run.status(print_status=True)
    results = run.results()
    
    # Display results
    print("\nResults:")
    for job_id, result in results.items():
        if result.is_success:
            analysis = result.parsed_response
            print(f"\nJob {job_id}:")
            print(f"  Summary: {analysis.summary}")
            print(f"  Sentiment: {analysis.sentiment}")
            print(f"  Key points: {', '.join(analysis.key_points)}")
        else:
            print(f"\nJob {job_id} failed: {result.error}")
    


if __name__ == "__main__":
    main()