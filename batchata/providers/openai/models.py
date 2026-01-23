"""OpenAI model configurations."""

from ..model_config import ModelConfig


# OpenAI model configurations for batch processing
OPENAI_MODELS = {
    # GPT-5.2 - latest flagship model
    "gpt-5.2-latest": ModelConfig(
        name="gpt-5.2-latest",
        max_input_tokens=2097152,  # 2M+ context window
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    "gpt-5.2": ModelConfig(
        name="gpt-5.2",
        max_input_tokens=2097152,  # 2M+ context window
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # GPT-4.1 - flagship model for complex tasks
    "gpt-4.1-2025-04-14": ModelConfig(
        name="gpt-4.1-2025-04-14",
        max_input_tokens=1047576,  # 1M+ context window
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # GPT-4.1 latest
    "gpt-4.1-latest": ModelConfig(
        name="gpt-4.1-latest",
        max_input_tokens=1047576,  # 1M+ context window
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # o4-mini - faster, more affordable reasoning model
    "o4-mini-2025-04-16": ModelConfig(
        name="o4-mini-2025-04-16",
        max_input_tokens=200000,
        max_output_tokens=100000,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # o4-mini latest
    "o4-mini-latest": ModelConfig(
        name="o4-mini-latest",
        max_input_tokens=200000,
        max_output_tokens=100000,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # o3 - most powerful reasoning model
    "o3-2025-04-16": ModelConfig(
        name="o3-2025-04-16",
        max_input_tokens=200000,
        max_output_tokens=100000,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # o3 latest
    "o3-latest": ModelConfig(
        name="o3-latest",
        max_input_tokens=200000,
        max_output_tokens=100000,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # gpt-4.1-nano - cost-effective model for examples and testing
    "gpt-4.1-nano-2025-04-14": ModelConfig(
        name="gpt-4.1-nano-2025-04-14",
        max_input_tokens=1000000,
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # gpt-4.1-nano latest
    "gpt-4.1-nano-latest": ModelConfig(
        name="gpt-4.1-nano-latest",
        max_input_tokens=1000000,
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
    
    # gpt-4o-mini - cost-effective general purpose model  
    "gpt-4o-mini-2024-07-18": ModelConfig(
        name="gpt-4o-mini-2024-07-18",
        max_input_tokens=128000,
        max_output_tokens=16384,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),
}