"""Google Gemini model configurations."""

from ..model_config import ModelConfig


# Google Gemini models with batch processing support
# Batch mode provides 50% discount on standard API pricing
GEMINI_MODELS = {
    "gemini-2.5-pro": ModelConfig(
        name="gemini-2.5-pro",
        max_input_tokens=2097152,  # 2M context
        max_output_tokens=8192,
        batch_discount=0.5,  # 50% discount confirmed in docs
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        name="gemini-2.5-flash-lite",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-2.0-flash": ModelConfig(
        name="gemini-2.0-flash",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".gif", ".webp"]
    ),
    "gemini-2.0-flash-lite": ModelConfig(
        name="gemini-2.0-flash-lite",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".gif", ".webp"]
    ),
}