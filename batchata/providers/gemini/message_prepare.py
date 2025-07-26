"""Message preparation for Google Gemini API."""

import json
import base64
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel

from ...core.job import Job
from ...utils import get_logger

logger = get_logger(__name__)


def prepare_messages(job: Job) -> Tuple[List[Dict], Optional[Dict]]:
    """Prepare messages and generation config for Gemini API.
    
    Returns:
        Tuple of (contents, generation_config) where generation_config includes response format
    """
    contents = []
    generation_config = {}
    
    # Case 1: Messages already provided
    if job.messages:
        # Convert messages to Gemini format
        for msg in job.messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                # Gemini doesn't have system role, prepend as user message
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"[System]: {content}"}]
                })
            elif role == "user":
                if isinstance(content, str):
                    contents.append({
                        "role": "user", 
                        "parts": [{"text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multi-part content (text + images)
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append({"text": part["text"]})
                        elif part.get("type") == "image_url":
                            # Convert base64 image to Gemini format
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                mime_type, data = url.split(",", 1)
                                mime_type = mime_type.split(":")[1].split(";")[0]
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": data
                                    }
                                })
                    contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                contents.append({
                    "role": "model",  # Gemini uses "model" instead of "assistant"
                    "parts": [{"text": content}]
                })
        
        # Log warning if citations are enabled with messages
        if job.enable_citations:
            logger.warning(
                f"Job {job.id}: Citations are enabled but using message format. "
                "Citations only work with file-based inputs (file + prompt)."
            )
    
    # Case 2: File + prompt provided
    elif job.file and job.prompt:
        parts = []
        
        # Handle file based on type
        if _is_image_file(job.file):
            # For images, use Gemini's inline_data format
            image_data = _read_file_as_base64(job.file)
            mime_type = _get_media_type(job.file)
            
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })
        else:
            # For text files (PDF, TXT, DOCX), read as text
            text_content = _read_file_as_text(job.file)
            if job.enable_citations:
                # Add citation markers for text content
                text_content = _add_citation_markers(text_content)
            
            parts.append({"text": text_content})
        
        # Add the prompt
        parts.append({"text": job.prompt})
        
        contents.append({
            "role": "user",
            "parts": parts
        })
    
    # Case 3: Prompt only
    elif job.prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": job.prompt}]
        })
    
    # Handle structured output
    if job.response_model:
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = _convert_pydantic_to_gemini_schema(job.response_model)
    
    # Set generation parameters
    if job.temperature is not None:
        generation_config["temperature"] = job.temperature
    if job.max_tokens is not None:
        generation_config["max_output_tokens"] = job.max_tokens
    
    return contents, generation_config if generation_config else None


def _is_image_file(file_path: Path) -> bool:
    """Check if file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    return file_path.suffix.lower() in image_extensions


def _get_media_type(file_path: Path) -> str:
    """Get MIME type for file."""
    extension = file_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg', 
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    return mime_types.get(extension, 'application/octet-stream')


def _read_file_as_base64(file_path: Path) -> str:
    """Read file and encode as base64."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _read_file_as_text(file_path: Path) -> str:
    """Read file content as text."""
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        # Use pypdf to extract text from PDF
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"[Page {page_num}]\n{text}")
        return "\n\n".join(text_parts)
    
    elif extension == '.docx':
        # Read DOCX file (basic implementation)
        try:
            from docx import Document
            doc = Document(str(file_path))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            logger.warning("python-docx not installed, treating DOCX as binary")
            return f"[Binary file: {file_path.name}]"
    
    else:
        # Plain text file
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def _add_citation_markers(text: str) -> str:
    """Add citation markers to text content (placeholder)."""
    # For now, just return the text as-is
    # TODO: Implement citation marking when Gemini citation support is added
    return text


def _convert_pydantic_to_gemini_schema(model: BaseModel) -> Dict:
    """Convert Pydantic model to Gemini schema format."""
    try:
        # Get the JSON schema from Pydantic
        json_schema = model.model_json_schema()
        
        # Convert to Gemini format (which is close to JSON Schema)
        gemini_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        if "properties" in json_schema:
            gemini_schema["properties"] = json_schema["properties"]
        
        if "required" in json_schema:
            gemini_schema["required"] = json_schema["required"]
        
        return gemini_schema
        
    except Exception as e:
        logger.warning(f"Failed to convert Pydantic schema to Gemini format: {e}")
        return {"type": "object"}