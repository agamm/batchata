"""Example demonstrating timeout functionality."""

from batchata import Batch

# Create batch with timeout
batch = (Batch("./timeout_results", max_parallel_batches=2)
        .set_default_params(model="gpt-4o-mini", temperature=0.7)
        .set_timeout(seconds=30)  # 30 second timeout for entire batch
        .add_cost_limit(usd=5.0))

# Add some jobs
for i in range(10):
    batch.add_job(
        messages=[{"role": "user", "content": f"Write a haiku about the number {i}"}],
        max_tokens=100
    )

print(f"Running batch with {len(batch)} jobs and 30 second timeout...")

# Run the batch
run = batch.run(print_status=True)

# Check results
results = run.results()
failed = run.get_failed_jobs()

print(f"\nCompleted: {len(results)} jobs")
print(f"Failed: {len(failed)} jobs")

# Show timeout failures
timeout_failures = {job_id: error for job_id, error in failed.items() if "Timeout" in error}
if timeout_failures:
    print(f"\nJobs that failed due to timeout:")
    for job_id, error in timeout_failures.items():
        print(f"  {job_id}: {error}")