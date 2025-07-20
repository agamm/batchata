"""Example demonstrating time limit functionality."""

from batchata import Batch

# Create batch with time limit
batch = (Batch("./time_limit_results", max_parallel_batches=2)
        .set_default_params(model="gpt-4o-mini", temperature=0.7)
        .add_time_limit(seconds=30)  # 30 second time limit for entire batch
        .add_cost_limit(usd=5.0))

# Add some jobs
for i in range(10):
    batch.add_job(
        messages=[{"role": "user", "content": f"Write a haiku about the number {i}"}],
        max_tokens=100
    )

print(f"Running batch with {len(batch)} jobs and 30 second time limit...")

# Run the batch
run = batch.run(print_status=True)

# Check results
results = run.results()

print(f"\nCompleted: {len(results['completed'])} jobs")
print(f"Failed: {len(results['failed'])} jobs")

# Show time limit failures
time_limit_failures = [result for result in results["failed"] if "Time limit" in result.error]
if time_limit_failures:
    print(f"\nJobs that failed due to time limit:")
    for result in time_limit_failures:
        print(f"  {result.job_id}: {result.error}")