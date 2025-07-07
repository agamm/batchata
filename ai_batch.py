"""
AI Batch Processing Module

A wrapper around instructor's batch processing and Anthropic's batch API for structured output.
"""

import os
import io
import time
from typing import List, Type, TypeVar
from pydantic import BaseModel
from instructor.batch import BatchJob
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

T = TypeVar('T', bound=BaseModel)


def batch(
    messages: List[List[dict]], 
    model: str, 
    response_model: Type[T], 
    max_tokens: int = 1024,
    temperature: float = 0.0,
    wait_for_completion: bool = True,
    poll_interval: int = 10,
    verbose: bool = False
) -> List[T]:
    """
    Process multiple message conversations using Anthropic's batch processing API.
    
    Args:
        messages: List of message conversations, each conversation is a list of message dicts
        model: Claude model name (e.g., "claude-3-haiku-20240307")
        response_model: Pydantic model class for structured response
        max_tokens: Maximum tokens per response (default: 1024)
        temperature: Temperature for response generation (default: 1.0)
        wait_for_completion: Whether to wait for batch completion (default: True)
        poll_interval: Polling interval in seconds when waiting (default: 30)
        
    Returns:
        List of response_model instances, one for each input message conversation
        
    Raises:
        ValueError: If API key is missing or other validation errors
        RuntimeError: If batch processing fails
    """
    if not messages:
        return []
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    # Create batch requests manually in the correct format
    import json
    import uuid
    import instructor
    
    # Get the tools/schema from instructor for structured output
    _, kwargs = instructor.handle_response_model(
        response_model=response_model, 
        mode=instructor.Mode.ANTHROPIC_JSON
    )
    
    batch_requests = []
    for i, conversation in enumerate(messages):
        # Remove system messages since Anthropic expects them at top level, not in messages
        user_messages = [msg for msg in conversation if msg.get("role") != "system"]
        
        # Exclude 'messages' from kwargs since we're setting it ourselves
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        
        request = {
            "custom_id": f"request_{i}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages,
                **filtered_kwargs
            }
        }
        batch_requests.append(request)
    
    # Submit batch to Anthropic
    client = Anthropic(api_key=api_key)
    batch_response = client.messages.batches.create(requests=batch_requests)
    batch_id = batch_response.id
    
    if not wait_for_completion:
        return batch_id
    
    # Wait for completion
    while True:
            
        batch_status = client.messages.batches.retrieve(batch_id)

        if verbose:
            print(f"Waiting for batch {batch_id} to complete...")
            print(f"Batch status: {batch_status.processing_status}")
        
        if batch_status.processing_status == "ended":
            break
        elif batch_status.processing_status in ["canceled", "expired"]:
            raise RuntimeError(f"Batch processing failed: {batch_status.processing_status}")
        
        time.sleep(poll_interval)
    
    # Get and parse results
    results = client.messages.batches.results(batch_id)
    
    # Parse results manually since instructor's parser expects different format
    import json
    parsed_results = []
    errors = []
    
    for result in results:
        try:
            if result.result.type == "succeeded":
                # Extract JSON from the message content
                message_content = result.result.message.content[0].text
                
                # Find JSON object in the text (look for { ... })
                start_idx = message_content.find('{')
                end_idx = message_content.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = message_content[start_idx:end_idx]
                    json_data = json.loads(json_str)
                    parsed_result = response_model(**json_data)
                    parsed_results.append(parsed_result)
                else:
                    raise ValueError(f"No JSON found in response: {message_content}")
            else:
                errors.append(result.model_dump())
        except Exception as e:
            errors.append({"error": str(e), "result": result.model_dump()})
        
    if errors:
        raise RuntimeError(f"Some batch requests failed: {len(errors)} errors")
    
    return parsed_results