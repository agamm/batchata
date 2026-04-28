"""Google Gemini model configurations."""

from ..model_config import ModelConfig

# updated based on current (Jan 28 2026) docs
# models - https://ai.google.dev/gemini-api/docs/models
# batch pricing - https://ai.google.dev/gemini-api/docs/pricing
# image file types - https://ai.google.dev/gemini-api/docs/image-understanding#supported-formats

# Google Gemini models with batch processing support
# Batch mode provides 50% discount on standard API pricing
GEMINI_MODELS = {
    "gemini-3-pro-preview": ModelConfig(
        name="gemini-3-pro-preview",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=65536,
        batch_discount=0.5,  # 50% discount confirmed in docs
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".webp"]
    ),
    "gemini-3-flash-preview": ModelConfig(
        name="gemini-3-flash-preview",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=65536,
        batch_discount=0.5,  # 50% discount confirmed in docs
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".webp"]
    ),
    "gemini-2.5-pro": ModelConfig(
        name="gemini-2.5-pro",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".webp"]
    ),
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=False,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".txt", ".jpg", ".png", ".webp"]
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        name="gemini-2.5-flash-lite",
        max_input_tokens=1048576,  # 1M context
        max_output_tokens=65536,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=False,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".jpg", ".png", ".webp"]
    )
}