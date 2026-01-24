"""OpenAI model configurations."""

from ..model_config import ModelConfig


# OpenAI model configurations for batch processing
OPENAI_MODELS = {
    # GPT-5.2 - most advanced flagship model (2026)
    "gpt-5.2-latest": ModelConfig(
        name="gpt-5.2-latest",
        max_input_tokens=4194304,  # 4M+ context window
        max_output_tokens=131072,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf", ".txt", ".docx"]
    ),

    "gpt-5.2": ModelConfig(
        name="gpt-5.2",
        max_input_tokens=4194304,  # 4M+ context window
        max_output_tokens=131072,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf", ".txt", ".docx"]
    ),

    "gpt-5.2-2025-12-11": ModelConfig(
        name="gpt-5.2-2025-12-11",
        max_input_tokens=4194304,  # 4M+ context window
        max_output_tokens=131072,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf", ".txt", ".docx"]
    ),

    "gpt-5.2-pro": ModelConfig(
        name="gpt-5.2-pro",
        max_input_tokens=4194304,  # 4M+ context window
        max_output_tokens=131072,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf", ".txt", ".docx"]
    ),

    "gpt-5.2-pro-2025-12-11": ModelConfig(
        name="gpt-5.2-pro-2025-12-11",
        max_input_tokens=4194304,  # 4M+ context window
        max_output_tokens=131072,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf", ".txt", ".docx"]
    ),

    # GPT-5 Mini - efficient model for fast tasks
    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        max_input_tokens=2097152,  # 2M context window
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),

    "gpt-5-mini-2025-08-07": ModelConfig(
        name="gpt-5-mini-2025-08-07",
        max_input_tokens=2097152,  # 2M context window
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),

    # GPT-5 Nano - cost-effective model for simple tasks
    "gpt-5-nano": ModelConfig(
        name="gpt-5-nano",
        max_input_tokens=1048576,  # 1M context window
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=False,
        supports_files=False,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[]
    ),

    "gpt-5-nano-2025-08-07": ModelConfig(
        name="gpt-5-nano-2025-08-07",
        max_input_tokens=1048576,  # 1M context window
        max_output_tokens=32768,
        batch_discount=0.5,
        supports_images=False,
        supports_files=False,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[]
    ),

    # GPT-4.5 - advanced flagship model (hypothetical for 2026)
    "gpt-4.5-latest": ModelConfig(
        name="gpt-4.5-latest",
        max_input_tokens=2097152,  # 2M+ context window
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".jpg", ".png", ".gif", ".webp", ".pdf"]
    ),

    "gpt-4.5": ModelConfig(
        name="gpt-4.5",
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