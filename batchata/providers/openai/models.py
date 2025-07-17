"""OpenAI model configurations."""

from ..model_config import ModelConfig


# OpenAI model configurations for batch processing
OPENAI_MODELS = {
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