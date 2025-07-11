#!/usr/bin/env python3
"""
Example demonstrating raw text responses without structured output.
"""

import time
from dotenv import load_dotenv
from batchata import batch

# Load environment variables for examples
load_dotenv()

def main():
    # Example messages for different use cases
    messages = [
        [{"role": "user", "content": "Write a haiku about programming"}],
        [{"role": "user", "content": "Explain quantum computing in one sentence"}],
        [{"role": "user", "content": "What's the capital of France?"}]
    ]
    
    print("Processing raw text responses...")
    print("--------------------------------------------------")
    
    # Process without response_model to get raw text
    job = batch(
        messages=messages,
        model="claude-3-haiku-20240307"
    )
    
    # Wait for completion
    while not job.is_complete():
        print(f"Batch job is running. Batch ID: {job._batch_id}...")
        time.sleep(30)  # Check every 30 seconds
    
    results = job.results()
    
    print("Raw text responses:")
    for i, entry in enumerate(results):
        result = entry["result"]  # Extract the text result
        print(f"\nResponse {i+1}:")
        print(result)

if __name__ == "__main__":
    main()