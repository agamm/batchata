"""Simple message preparation for Google Gemini API."""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...core.job import Job


def prepare_messages(job: Job) -> Tuple[List[Dict], Optional[Dict]]:
    """Convert job to Gemini batch format: (contents, generation_config)."""
    contents = []
    generation_config = {}
    
    # Convert messages if provided
    if job.messages:
        for msg in job.messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                contents.append({"role": "user", "parts": [{"text": f"[System]: {content}"}]})
            elif role == "user":
                if isinstance(content, str):
                    contents.append({"role": "user", "parts": [{"text": content}]})
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if part["type"] == "text":
                            parts.append({"text": part["text"]})
                        elif part["type"] == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                mime_type, data = url.split(",", 1)
                                mime_type = mime_type.split(":")[1].split(";")[0]
                                parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                    contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
    
    # Handle file + prompt
    elif job.file and job.prompt:
        parts = []
        
        if _is_image(job.file) or _is_pdf(job.file):
            # Send images and PDFs as binary data
            parts.append({
                "inline_data": {
                    "mime_type": _get_mime_type(job.file),
                    "data": _read_as_base64(job.file)
                }
            })
        else:
            # Send text files as text content
            parts.append({"text": _read_as_text(job.file)})
        
        parts.append({"text": job.prompt})
        contents.append({"role": "user", "parts": parts})
    
    # Handle prompt only
    elif job.prompt:
        contents.append({"role": "user", "parts": [{"text": job.prompt}]})
    
    # Add generation config
    if job.response_model:
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = _pydantic_to_schema(job.response_model)
    
    if job.temperature is not None:
        generation_config["temperature"] = job.temperature
    if job.max_tokens is not None:
        generation_config["max_output_tokens"] = job.max_tokens
    
    return contents, (generation_config if generation_config else None)


# File type constants for consistency and performance
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
PDF_EXTENSION = '.pdf'

MIME_TYPE_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.pdf': 'application/pdf'
}


def _is_image(file_path: Path) -> bool:
    """Check if file is an image."""
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def _is_pdf(file_path: Path) -> bool:
    """Check if file is a PDF."""
    return file_path.suffix.lower() == PDF_EXTENSION


def _get_mime_type(file_path: Path) -> str:
    """Get MIME type for file."""
    ext = file_path.suffix.lower()
    return MIME_TYPE_MAP.get(ext, 'text/plain')


def _read_as_base64(file_path: Path) -> str:
    """Read file as base64."""
    return base64.b64encode(file_path.read_bytes()).decode()


def _read_as_text(file_path: Path) -> str:
    """Read file as text (for non-PDF/non-image files only)."""
    ext = file_path.suffix.lower()
    
    if ext == '.pdf':
        # PDFs should be sent as binary data, not text
        raise ValueError("PDF files should be processed as binary data using _read_as_base64")
    
    if ext == '.docx':
        raise ValueError("DOCX files are not supported. Please convert to PDF for better compatibility with Gemini's document processing.")
    
    # For plain text files
    return file_path.read_text(encoding='utf-8')


def _pydantic_to_schema(model) -> Dict:
    """Convert Pydantic model to Gemini schema."""
    try:
        schema = model.model_json_schema()
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    except (KeyError, ValueError, AttributeError):
        # If schema extraction fails, return minimal object schema
        return {"type": "object"}