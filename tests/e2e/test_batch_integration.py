"""
End-to-end integration tests for ai_batch.

These tests require a real API key and make actual calls to Anthropic's API.
They test the happy path scenarios with real data.
"""

import pytest
from pydantic import BaseModel
from ai_batch import batch


class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float


def test_spam_detection_happy_path():
    """Test spam detection with real API - happy path only."""
    emails = [
        "You've won $1,000,000! Click here now!",  # Obviously spam
        "Meeting tomorrow at 3pm to discuss Q3 results"  # Obviously not spam
    ]
    
    messages = [[{"role": "user", "content": f"You are a spam detection expert. Is this spam? {email}"}] for email in emails]
    
    results = batch(
        messages=messages,
        model="claude-3-haiku-20240307",
        response_model=SpamResult
    )
    
    assert len(results) == 2
    assert all(isinstance(result, SpamResult) for result in results)
    
    # Verify first email is detected as spam
    assert results[0].is_spam == True
    assert results[0].confidence > 0.0
    assert len(results[0].reason) > 0
    
    # Verify second email is not spam
    assert results[1].is_spam == False
    assert results[1].confidence > 0.0  # Confidence represents how sure the model is, not spam probability
    assert len(results[1].reason) > 0

