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
        Batch(results_dir="./examples/output", max_parallel_batches=1, items_per_batch=1)
        .set_state(file="./examples/demo_state.json", reuse_previous=False)
        .set_default_params(model="claude-sonnet-4-20250514", temperature=0.7)
        .add_cost_limit(usd=5.0)
        .set_verbosity("warn")
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
            response_model=Analysis,
            enable_citations=True
        )
    
    # Execute batch
    print("Starting batch processing...")
    run = batch.run(on_progress=lambda s, t, b: \
                    print(f"\rProgress: {s['completed']}/{s['total']} jobs | "\
                          f"Batches: {s['batches_completed']}/{s['batches_total']} (pending: {s['batches_pending']}) | " \
                          f"Cost: ${round(s['cost_usd'],3)}/{s['cost_limit_usd']} | " \
                          f"Items per batch: {s['items_per_batch']} | Time: {round(t, 2)}s", end=""))
    
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