"""Message preparation for Anthropic API."""

import json
import base64
from textwrap import dedent
from typing import Dict, List, Type, Optional
from pathlib import Path
from pydantic import BaseModel

from ...core.job import Job
from ...utils import get_logger

logger = get_logger(__name__)


def prepare_messages(job: Job) -> tuple[List[Dict], Optional[str]]:
    """Prepare messages and system prompt for Anthropic API.
    
    Returns:
        Tuple of (messages, system_prompt)
    """
    messages = []
    system_parts = []
    
    # Case 1: Messages already provided
    if job.messages:
        # Extract system messages
        for msg in job.messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                messages.append(msg)
        
        # Log warning if citations are enabled with messages
        if job.enable_citations:
            logger.warning(
                f"Job {job.id}: Citations are enabled but using message format. "
                "Citations only work with file-based inputs (file + prompt)."
            )
    
    # Case 2: File + prompt provided
    elif job.file and job.prompt:
        content_parts = []
        
        # Get media type to determine how to handle file
        media_type = _get_media_type(job.file)
        
        # For text files, use "text" source type
        if media_type == 'text/plain':
            with open(job.file, 'r', encoding='utf-8') as f:
                text_data = f.read()
            
            document_part = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": media_type,
                    "data": text_data
                }
            }
        else:
            # For PDFs and other binary files, use base64
            file_data = _read_file_as_base64(job.file)
            document_part = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": file_data
                }
            }
        
        # Add citations if enabled
        if job.enable_citations:
            document_part["title"] = job.file.name
            document_part["citations"] = {"enabled": True}
            system_parts.append("(Important) Include concise citations in your response referencing the specific parts of the documents you used.")
        
        content_parts.append(document_part)
        
        # Add prompt text
        content_parts.append({
            "type": "text",
            "text": job.prompt
        })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
    
    # Add response model schema if provided
    if job.response_model:
        schema = job.response_model.model_json_schema()
        schema_instruction = dedent(f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
        """).strip()
        system_parts.append(schema_instruction)
    
    # Combine system messages
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    
    return messages, system_prompt


def _read_file_as_base64(file_path: Path) -> str:
    """Read file and encode as base64."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _get_media_type(file_path: Path) -> str:
    """Get media type for file."""
    ext = file_path.suffix.lower()
    media_types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.md': 'text/plain',
        '.markdown': 'text/plain',
        '.json': 'text/plain',
        '.yaml': 'text/plain',
        '.yml': 'text/plain',
        '.csv': 'text/plain',
        '.xml': 'text/plain',
        '.html': 'text/plain',
        '.htm': 'text/plain',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(ext, 'text/plain')