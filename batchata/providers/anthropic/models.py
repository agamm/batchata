"""Anthropic model configurations."""

from ..model_config import ModelConfig


# Anthropic model configurations
ANTHROPIC_MODELS = {
    "claude-opus-4-20250514": ModelConfig(
        name="claude-opus-4-20250514",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-7-sonnet-20250219": ModelConfig(
        name="claude-3-7-sonnet-20250219",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-7-sonnet-latest": ModelConfig(
        name="claude-3-7-sonnet-latest",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-5-sonnet-latest": ModelConfig(
        name="claude-3-5-sonnet-latest",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-5-sonnet-20240620": ModelConfig(
        name="claude-3-5-sonnet-20240620",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        name="claude-3-5-haiku-20241022",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-5-haiku-latest": ModelConfig(
        name="claude-3-5-haiku-latest",
        max_input_tokens=200000,
        max_output_tokens=8192,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-haiku-20240307": ModelConfig(
        name="claude-3-haiku-20240307",
        max_input_tokens=200000,
        max_output_tokens=4096,
        batch_discount=0.5,
        supports_images=False,
        supports_files=False,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[]
    ),
    "claude-3-opus-20240229": ModelConfig(
        name="claude-3-opus-20240229",
        max_input_tokens=200000,
        max_output_tokens=4096,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        name="claude-3-sonnet-20240229",
        max_input_tokens=200000,
        max_output_tokens=4096,
        batch_discount=0.5,
        supports_images=True,
        supports_files=True,
        supports_citations=True,
        supports_structured_output=True,
        file_types=[".pdf", ".txt", ".docx", ".jpg", ".png", ".gif", ".webp"]
    ),
}