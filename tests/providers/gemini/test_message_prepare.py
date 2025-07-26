"""Tests for Gemini message preparation."""

import pytest
from pathlib import Path
from unittest.mock import patch
from pydantic import BaseModel

from batchata.providers.gemini.message_prepare import (
    prepare_messages, _is_image, _is_pdf, _get_mime_type
)
from batchata.core.job import Job


class SampleResponse(BaseModel):
    """Sample response model for structured output."""
    name: str
    age: int


class TestMessagePrepare:
    """Test message preparation functionality."""
    
    def test_prepare_simple_prompt(self):
        """Test preparing a simple text prompt."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="What is the capital of France?"
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "What is the capital of France?"
        # Job has default temperature and max_tokens, so generation_config will include them
        assert generation_config is not None
        assert generation_config["temperature"] == 0.7  # Job default
        assert generation_config["max_output_tokens"] == 1000  # Job default
    
    def test_prepare_messages_format(self):
        """Test preparing messages in OpenAI format."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 4
        # System message converted to user with prefix
        assert contents[0]["role"] == "user"
        assert "[System]:" in contents[0]["parts"][0]["text"]
        # User message
        assert contents[1]["role"] == "user"
        assert contents[1]["parts"][0]["text"] == "Hello"
        # Assistant message converted to model
        assert contents[2]["role"] == "model"
        assert contents[2]["parts"][0]["text"] == "Hi there!"
        # User message
        assert contents[3]["role"] == "user"
        assert contents[3]["parts"][0]["text"] == "How are you?"
    
    def test_prepare_with_structured_output(self):
        """Test preparing messages with structured output."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Generate a person",
            response_model=SampleResponse
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert generation_config is not None
        assert generation_config["response_mime_type"] == "application/json"
        assert "response_schema" in generation_config
        assert generation_config["response_schema"]["type"] == "object"
    
    def test_prepare_with_temperature_and_max_tokens(self):
        """Test preparing with temperature and max tokens."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Tell me a story",
            temperature=0.8,
            max_tokens=500
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert generation_config is not None
        assert generation_config["temperature"] == 0.8
        assert generation_config["max_output_tokens"] == 500
    
    @patch('batchata.providers.gemini.message_prepare._read_as_base64')
    def test_prepare_with_pdf_file(self, mock_read_base64):
        """Test preparing with PDF file."""
        mock_read_base64.return_value = "base64encodeddata"
        
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Summarize this document",
            file=Path("/fake/document.pdf")
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2
        # First part should be the PDF as inline data
        assert "inline_data" in contents[0]["parts"][0]
        assert contents[0]["parts"][0]["inline_data"]["mime_type"] == "application/pdf"
        assert contents[0]["parts"][0]["inline_data"]["data"] == "base64encodeddata"
        # Second part should be the prompt
        assert contents[0]["parts"][1]["text"] == "Summarize this document"
    
    @patch('batchata.providers.gemini.message_prepare._read_as_base64')
    def test_prepare_with_image_file(self, mock_read_base64):
        """Test preparing with image file."""
        mock_read_base64.return_value = "base64imagedata"
        
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Describe this image",
            file=Path("/fake/image.jpg")
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2
        # First part should be the image as inline data
        assert "inline_data" in contents[0]["parts"][0]
        assert contents[0]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"
        assert contents[0]["parts"][0]["inline_data"]["data"] == "base64imagedata"
    
    @patch('pathlib.Path.read_text')
    def test_prepare_with_text_file(self, mock_read_text):
        """Test preparing with text file."""
        mock_read_text.return_value = "This is file content"
        
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Analyze this text",
            file=Path("/fake/document.txt")
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2
        # First part should be the text content
        assert contents[0]["parts"][0]["text"] == "This is file content"
        # Second part should be the prompt
        assert contents[0]["parts"][1]["text"] == "Analyze this text"
    
    def test_is_image(self):
        """Test image file detection."""
        assert _is_image(Path("test.jpg")) is True
        assert _is_image(Path("test.png")) is True
        assert _is_image(Path("test.gif")) is True
        assert _is_image(Path("test.pdf")) is False
        assert _is_image(Path("test.txt")) is False
    
    def test_is_pdf(self):
        """Test PDF file detection."""
        assert _is_pdf(Path("test.pdf")) is True
        assert _is_pdf(Path("test.jpg")) is False
        assert _is_pdf(Path("test.txt")) is False
    
    def test_get_mime_type(self):
        """Test MIME type detection."""
        assert _get_mime_type(Path("test.jpg")) == "image/jpeg"
        assert _get_mime_type(Path("test.png")) == "image/png"
        assert _get_mime_type(Path("test.pdf")) == "application/pdf"
        assert _get_mime_type(Path("test.txt")) == "text/plain"  # Default changed
        assert _get_mime_type(Path("test.unknown")) == "text/plain"  # Default
    
    def test_docx_raises_error(self):
        """Test that DOCX files raise an error."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Analyze this document",
            file=Path("/fake/document.docx")
        )
        
        with pytest.raises(ValueError, match="DOCX files are not supported"):
            prepare_messages(job)
    
    def test_multimodal_content_in_messages(self):
        """Test handling multimodal content within messages."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                            }
                        }
                    ]
                }
            ]
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert len(contents[0]["parts"]) == 2
        
        # Check text part
        assert contents[0]["parts"][0]["text"] == "What's in this image?"
        
        # Check image part
        assert "inline_data" in contents[0]["parts"][1]
        assert contents[0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"
        assert contents[0]["parts"][1]["inline_data"]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    
    def test_system_message_handling(self):
        """Test that system messages are converted properly."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 2
        # System message should be converted to user message with prefix
        assert contents[0]["role"] == "user"
        assert "[System]:" in contents[0]["parts"][0]["text"]
        assert "helpful assistant" in contents[0]["parts"][0]["text"]
        
        # Regular user message
        assert contents[1]["role"] == "user"
        assert contents[1]["parts"][0]["text"] == "Hello"
    
    def test_large_file_handling(self):
        """Test handling of large files (simulated)."""
        # Simulate a large PDF file
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Summarize this large document",
            file=Path("/fake/large_document.pdf")
        )
        
        # Mock large file content
        with patch('batchata.providers.gemini.message_prepare._read_as_base64') as mock_read:
            mock_read.return_value = "x" * 1000000  # 1MB of data
            
            contents, generation_config = prepare_messages(job)
            
            assert len(contents) == 1
            assert len(contents[0]["parts"]) == 2
            assert "inline_data" in contents[0]["parts"][0]
            assert len(contents[0]["parts"][0]["inline_data"]["data"]) == 1000000
    
    def test_empty_content_handling(self):
        """Test handling of edge cases with empty content."""
        # Test with minimal valid content - empty string is now rejected by Job validation
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="  "  # Whitespace-only prompt
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert len(contents) == 1
        assert contents[0]["parts"][0]["text"] == "  "
        
    def test_special_characters_in_content(self):
        """Test handling of special characters and Unicode."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Test with émojis 🚀 and spëcial chàracters ñ 中文"
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert contents[0]["parts"][0]["text"] == "Test with émojis 🚀 and spëcial chàracters ñ 中文"
    
    def test_generation_config_defaults(self):
        """Test that default generation config values are set correctly."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Test defaults"
            # Using Job defaults: temperature=0.7, max_tokens=1000
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert generation_config["temperature"] == 0.7
        assert generation_config["max_output_tokens"] == 1000
    
    def test_generation_config_overrides(self):
        """Test that custom generation config values override defaults."""
        job = Job(
            id="test-1",
            model="gemini-2.5-flash",
            prompt="Test overrides",
            temperature=0.2,
            max_tokens=500
        )
        
        contents, generation_config = prepare_messages(job)
        
        assert generation_config["temperature"] == 0.2
        assert generation_config["max_output_tokens"] == 500