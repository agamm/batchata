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
}