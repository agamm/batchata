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
        Batch(results_dir="./examples/output", max_parallel_batches=1, items_per_batch=3)
        .set_state(file="./examples/demo_state.json", reuse_state=False)
        .set_default_params(model="gemini-2.5-flash", temperature=0.7)
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
            # enable_citations=True
        )
    
    # Execute batch
    print("Starting batch processing...")
    run = batch.run(print_status=True)
    
    # Get results
    run.status(print_status=True)
    results = run.results()
    
    # Display results
    print("\nResults:")
    
    # Show successful results
    for result in results["completed"]:
        analysis = result.parsed_response
        print(f"\nJob {result.job_id}:")
        print(f"  Summary: {analysis.summary}")
        print(f"  Sentiment: {analysis.sentiment}")
        print(f"  Key points: {', '.join(analysis.key_points)}")
    
    # Show failed results
    for result in results["failed"]:
        print(f"\nJob {result.job_id} failed: {result.error}")
    
    # Show cancelled results
    for result in results["cancelled"]:
        print(f"\nJob {result.job_id} was cancelled: {result.error}")
    


if __name__ == "__main__":
    main()