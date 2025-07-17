"""Message preparation for OpenAI API."""

import json
import base64
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel

from ...core.job import Job
from ...utils import get_logger

logger = get_logger(__name__)


def prepare_messages(job: Job) -> Tuple[List[Dict], Optional[Dict]]:
    """Prepare messages and response format for OpenAI API.
    
    Returns:
        Tuple of (messages, response_format) where response_format is for structured output
    """
    messages = []
    response_format = None
    
    # Case 1: Messages already provided
    if job.messages:
        messages = job.messages.copy()
        
        # Log warning if citations are enabled with messages
        if job.enable_citations:
            logger.warning(
                f"Job {job.id}: Citations are enabled but OpenAI doesn't support citations. "
                "Citations will be ignored."
            )
    
    # Case 2: File + prompt provided
    elif job.file and job.prompt:
        content_parts = []
        
        # Handle file based on type
        if _is_image_file(job.file):
            # For images, use OpenAI's image format
            image_data = _read_file_as_base64(job.file)
            media_type = _get_media_type(job.file)
            
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }
            })
        elif _is_pdf_file(job.file):
            # For PDFs, use OpenAI's document format with base64 encoding
            pdf_data = _read_file_as_base64(job.file)
            
            content_parts.append({
                "type": "text",
                "text": f"data:application/pdf;base64,{pdf_data}"
            })
        else:
            # For text files, read content and add as text
            with open(job.file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            content_parts.append({
                "type": "text",
                "text": f"File content ({job.file.name}):\n\n{file_content}"
            })
        
        # Add prompt text
        content_parts.append({
            "type": "text",
            "text": job.prompt
        })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        # Log warning if citations are enabled
        if job.enable_citations:
            logger.warning(
                f"Job {job.id}: Citations are enabled but OpenAI doesn't support citations. "
                "Citations will be ignored."
            )
    
    # Add response model schema if provided
    if job.response_model:
        # Generate schema and ensure additionalProperties is set to false for OpenAI
        schema = job.response_model.model_json_schema()
        schema["additionalProperties"] = False
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    
    return messages, response_format


def _is_image_file(file_path: Path) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    return file_path.suffix.lower() in image_extensions


def _is_pdf_file(file_path: Path) -> bool:
    """Check if file is a PDF based on extension."""
    return file_path.suffix.lower() == '.pdf'


def _read_file_as_base64(file_path: Path) -> str:
    """Read file and encode as base64."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _get_media_type(file_path: Path) -> str:
    """Get media type for file."""
    ext = file_path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.txt': 'text/plain',
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    return media_types.get(ext, 'application/octet-stream')


def prepare_jsonl_request(job: Job) -> Dict:
    """Prepare a single JSONL request for OpenAI batch API.
    
    Args:
        job: Job to prepare request for
        
    Returns:
        Dict representing a single line in the JSONL batch file
    """
    messages, response_format = prepare_messages(job)
    
    request = {
        "custom_id": job.id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": job.model,
            "messages": messages,
            "max_tokens": job.max_tokens,
            "temperature": job.temperature
        }
    }
    
    # Add response format if structured output is needed
    if response_format:
        request["body"]["response_format"] = response_format
    
    return request