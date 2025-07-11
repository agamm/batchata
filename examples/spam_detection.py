"""
Spam Detection Example

Demonstrates using bachata to classify multiple emails as spam or not spam.
"""

from dotenv import load_dotenv
from pydantic import BaseModel
from batchata import batch

# Load environment variables for examples
load_dotenv()


class SpamResult(BaseModel):
    is_spam: bool
    confidence: float
    reason: str


def main():
    """Run spam detection on sample emails."""
    
    # Sample emails to classify
    emails = [
        "You've won $1,000,000! Click here now!",
        "Meeting tomorrow at 3pm to discuss Q3 results",
        "URGENT: Verify your account or it will be closed!",
    ]
    
    print("Classifying emails for spam...")
    print("-" * 50)
    
    # Convert emails to message format expected by Claude
    messages = [
        [{"role": "user", "content": f"You are a spam detection expert. Analyze this email and determine if it is spam: {email}"}] 
        for email in emails
    ]
    
    try:
        # Process all emails in batch
        batch_job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=SpamResult,
            verbose=True
        )
        
        # Display results
        for email, batch_result in zip(emails, batch_job.results()):
            result = batch_result['result']  # Extract the SpamResult from BatchResult
            status = "🚨 SPAM" if result.is_spam else "✅ NOT SPAM"
            print(f"{status} (confidence: {result.confidence:.1%})")
            print(f"Email: {email}")
            print(f"Reason: {result.reason}")
            print("-" * 50)
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you have set the ANTHROPIC_API_KEY environment variable.")
        print("You can create a .env file with: ANTHROPIC_API_KEY=your_key_here")


if __name__ == "__main__":
    main()