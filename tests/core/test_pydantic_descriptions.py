"""Unit tests for Pydantic model field descriptions in prompts."""

from pydantic import BaseModel, Field


def test_pydantic_model_descriptions_in_prompt():
    """Test that Pydantic model field descriptions are included in the structured output schema."""
    
    from batchata.providers.anthropic.message_prepare import prepare_messages
    from batchata.core.job import Job
    
    class DetailedResponse(BaseModel):
        """Response model with detailed field descriptions."""
        name: str = Field(description="The full name of the person")
        age: int = Field(description="Age in years, must be positive")
        email: str = Field(description="Valid email address format")
        status: str = Field(description="Current status: active, inactive, or pending")
    
    # Create a job with the detailed response model
    job = Job(
        id="test-job-123",
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Extract person details from this text"}],
        response_model=DetailedResponse,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Prepare messages (this should include the schema with descriptions)
    messages, system_prompt = prepare_messages(job)
    
    # Check that the schema is present in the system prompt
    # For Anthropic, structured output is added to the system prompt
    assert system_prompt is not None
    assert "schema" in system_prompt.lower() or "json" in system_prompt.lower()
    
    # Check that field descriptions are present in the system prompt
    assert "The full name of the person" in system_prompt
    assert "Age in years, must be positive" in system_prompt
    assert "Valid email address format" in system_prompt
    assert "Current status: active, inactive, or pending" in system_prompt
    
    # Check that field names are present
    assert "name" in system_prompt
    assert "age" in system_prompt
    assert "email" in system_prompt
    assert "status" in system_prompt


def test_pydantic_model_descriptions_openai():
    """Test that Pydantic model field descriptions work with OpenAI provider too."""
    
    from batchata.providers.openai.message_prepare import prepare_messages
    from batchata.core.job import Job
    
    class ProductInfo(BaseModel):
        """Product information extraction model."""
        product_name: str = Field(description="The name or title of the product")
        price: float = Field(description="Product price in USD")
        category: str = Field(description="Product category (electronics, clothing, etc.)")
        in_stock: bool = Field(description="Whether the product is currently available")
    
    # Create a job with the response model
    job = Job(
        id="test-job-456",
        model="gpt-4o",
        messages=[{"role": "user", "content": "Extract product info from this listing"}],
        response_model=ProductInfo,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Prepare messages for OpenAI
    messages, response_format = prepare_messages(job)
    
    # Check that response_format is set for structured output
    assert response_format is not None
    assert "type" in response_format
    assert response_format["type"] == "json_schema"
    
    # Check that the schema contains field descriptions
    schema = response_format["json_schema"]["schema"]
    properties = schema["properties"]
    
    assert "product_name" in properties
    assert properties["product_name"]["description"] == "The name or title of the product"
    
    assert "price" in properties
    assert properties["price"]["description"] == "Product price in USD"
    
    assert "category" in properties
    assert properties["category"]["description"] == "Product category (electronics, clothing, etc.)"
    
    assert "in_stock" in properties
    assert properties["in_stock"]["description"] == "Whether the product is currently available"