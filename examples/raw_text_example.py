#!/usr/bin/env python3
"""
Example demonstrating raw text responses without structured output.
"""

from src.ai_batch import batch

def main():
    # Example messages for different use cases
    messages = [
        [{"role": "user", "content": "Write a haiku about programming"}],
        [{"role": "user", "content": "Explain quantum computing in one sentence"}],
        [{"role": "user", "content": "What's the capital of France?"}]
    ]
    
    # Process without response_model to get raw text
    results = batch(
        messages=messages,
        model="claude-3-haiku-20240307"
    )
    
    print("Raw text responses:")
    for i, result in enumerate(results):
        print(f"\nResponse {i+1}:")
        print(result)

if __name__ == "__main__":
    main()