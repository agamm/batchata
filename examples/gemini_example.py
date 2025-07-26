#!/usr/bin/env python3
"""Example using Gemini models with batchata.

Note: Gemini doesn't have true batch processing like OpenAI/Anthropic.
This uses async processing to simulate batch behavior.
"""

import os
import tempfile
from pydantic import BaseModel
from pathlib import Path

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("Please set GOOGLE_API_KEY environment variable")
    print("You can get a key from: https://ai.google.dev/")
    exit(1)

from batchata import Batch


class SentimentAnalysis(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    summary: str


def main():
    print("🤖 Batchata Gemini Example")
    print("="*50)
    
    # Create batch with temporary results directory
    with tempfile.TemporaryDirectory() as temp_dir:
        batch = Batch(
            results_dir=temp_dir,
            max_parallel_batches=1,
            items_per_batch=5
        ).set_default_params(
            model="gemini-1.5-flash",
            temperature=0.3
        ).add_cost_limit(usd=1.0)  # Small limit for demo
        
        # Sample texts for sentiment analysis
        texts = [
            "I absolutely love this new product! It's amazing!",
            "This is terrible quality. Very disappointed.",
            "The weather is okay today, nothing special.",
            "Best purchase I've made all year! Highly recommend!",
            "Not good, not bad, just average I guess."
        ]
        
        print(f"Adding {len(texts)} sentiment analysis jobs...")
        
        # Add jobs
        for i, text in enumerate(texts, 1):
            batch.add_job(
                prompt=f"Analyze the sentiment of this text: '{text}'",
                response_model=SentimentAnalysis
            )
            print(f"  ✓ Added job {i}: {text[:30]}...")
        
        print(f"\nStarting batch processing...")
        print("Note: Gemini uses async processing, not true batch API")
        
        # Run the batch
        run = batch.run(print_status=True)
        
        # Get results
        results = run.results()
        
        print(f"\n📊 Results Summary:")
        print(f"✅ Completed: {len(results['completed'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"🚫 Cancelled: {len(results['cancelled'])}")
        
        # Show detailed results
        if results['completed']:
            print(f"\n📝 Detailed Results:")
            for i, result in enumerate(results['completed'], 1):
                analysis = result.parsed_response
                print(f"\nJob {i}:")
                print(f"  Sentiment: {analysis.sentiment}")
                print(f"  Confidence: {analysis.confidence:.2f}")
                print(f"  Summary: {analysis.summary}")
                print(f"  Cost: ${result.cost_usd:.4f}")
        
        # Show any failures
        if results['failed']:
            print(f"\n❌ Failed Jobs:")
            for result in results['failed']:
                print(f"  Job {result.job_id}: {result.error}")
        
        # Show total cost
        total_cost = sum(r.cost_usd for r in results['completed'])
        print(f"\n💰 Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()