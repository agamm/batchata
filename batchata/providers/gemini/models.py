"""Google Gemini model configurations."""

from ..model_config import ModelConfig


# Google Gemini model configurations
# Note: Batch processing available via Google Cloud Batch API
GEMINI_MODELS = {
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        max_input_tokens=2097152,  # 2M context window
        max_output_tokens=8192,
        batch_discount=0.5,  # Assuming similar to other providers
        supports_images=True,
        supports_files=True,
        supports_citations=False,  # Not implemented yet
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        max_input_tokens=1048576,  # 1M context window
        max_output_tokens=8192,
        batch_discount=0.5,  # Assuming similar to other providers
        supports_images=True,
        supports_files=True,
        supports_citations=False,  # Not implemented yet
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-1.5-flash-8b": ModelConfig(
        name="gemini-1.5-flash-8b",
        max_input_tokens=1048576,  # 1M context window
        max_output_tokens=8192,
        batch_discount=0.5,  # Assuming similar to other providers
        supports_images=True,
        supports_files=True,
        supports_citations=False,  # Not implemented yet
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-2.0-flash-exp": ModelConfig(
        name="gemini-2.0-flash-exp",
        max_input_tokens=1048576,  # 1M context window
        max_output_tokens=8192,
        batch_discount=0.5,  # Assuming similar to other providers
        supports_images=True,
        supports_files=True,
        supports_citations=False,  # Not implemented yet
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-1.0-pro": ModelConfig(
        name="gemini-1.0-pro",
        max_input_tokens=32768,
        max_output_tokens=2048,
        batch_discount=0.5,  # Assuming similar to other providers
        supports_images=False,
        supports_files=False,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[]
    ),
}