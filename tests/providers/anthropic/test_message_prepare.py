"""Tests for Anthropic message preparation.

Testing:
1. Message format conversion for Anthropic API
2. File content inclusion in messages
3. Citation mode handling
"""

import pytest
from unittest.mock import patch, mock_open
from pydantic import BaseModel
from typing import Optional

from batchata.providers.anthropic.message_prepare import prepare_messages
from batchata.core.job import Job


class TestMessagePrepare:
    """Test message preparation for Anthropic API."""
    
    def test_basic_message_preparation(self):
        """Test preparing simple text messages."""
        job = Job(
            id="test-job",
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        
        messages, system = prepare_messages(job)
        
        # System should be None for no system message
        assert system is None
        
        # Messages should be formatted correctly
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}
    
    def test_file_based_message_preparation(self):
        """Test preparing messages from file inputs."""
        from pathlib import Path
        
        with patch("builtins.open", mock_open(read_data="File content here")):
            job = Job(
                id="file-job",
                model="claude-3-5-sonnet-20241022",
                file=Path("/tmp/test.txt"),
                prompt="Summarize this document"
            )
            
            messages, system = prepare_messages(job)
            
            assert system is None
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            
            # Check content structure for file
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert len(content) == 2
            
            # First part should be the document with text data for .txt files
            assert content[0]["type"] == "document"
            assert content[0]["source"]["type"] == "text"
            assert content[0]["source"]["media_type"] == "text/plain"
            assert content[0]["source"]["data"] == "File content here"
            
            # Second part should be the prompt
            assert content[1]["type"] == "text"
            assert content[1]["text"] == "Summarize this document"
    
    def test_citation_mode_with_file(self):
        """Test system message for citation mode with file input."""
        from pathlib import Path
        
        with patch("builtins.open", mock_open(read_data="File content here")):
            job = Job(
                id="citation-job",
                model="claude-3-5-sonnet-20241022",
                file=Path("/tmp/test.txt"),
                prompt="Analyze this document",
                enable_citations=True
            )
            
            messages, system = prepare_messages(job)
            
            # Should have system message for citations
            assert system is not None
            assert "citation" in system.lower()
            assert "documents" in system.lower()
            
            # Check that document has citations enabled
            assert len(messages) == 1
            content = messages[0]["content"]
            document_part = content[0]
            assert document_part["type"] == "document"
            assert "title" in document_part
            assert "citations" in document_part
            assert document_part["citations"]["enabled"] is True
    
    def test_no_citation_mode_with_messages(self):
        """Test that regular messages don't get citation system prompt."""
        job = Job(
            id="regular-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Tell me about Python"}],
            enable_citations=True  # This should have no effect for message-based jobs
        )
        
        messages, system = prepare_messages(job)
        
        # No system message for regular messages even with citations enabled
        assert system is None
        
        # Messages should be passed through unchanged
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me about Python"
    
    def test_response_model_system_message(self):
        """Test that response model adds correct system message."""
        # Define a test Pydantic model
        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: Optional[str] = None
        
        job = Job(
            id="json-job",
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Get person info"}],
            response_model=PersonInfo
        )
        
        messages, system = prepare_messages(job)
        
        # Should have system message for JSON schema
        assert system is not None
        assert "json" in system.lower()
        assert "schema" in system.lower()
        assert "PersonInfo" in system  # Model name should be in schema
        assert "name" in system  # Field names should be in schema
        assert "age" in system
        assert "occupation" in system
        
        # Messages should be unchanged
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Get person info"
    
    def test_file_with_citations_and_response_model(self):
        """Test file input with both citations and response model."""
        from pathlib import Path
        
        # Define a test Pydantic model
        class DocumentSummary(BaseModel):
            title: str
            key_points: list[str]
            confidence: float
        
        with patch("builtins.open", mock_open(read_data=b"Document content here")):
            job = Job(
                id="file-citations-json-job",
                model="claude-3-5-sonnet-20241022",
                file=Path("/tmp/document.pdf"),
                prompt="Summarize this document as JSON",
                enable_citations=True,
                response_model=DocumentSummary
            )
            
            messages, system = prepare_messages(job)
            
            # Should have system message combining both citations and JSON schema
            assert system is not None
            assert "citation" in system.lower()
            assert "documents" in system.lower()
            assert "json" in system.lower()
            assert "schema" in system.lower()
            assert "DocumentSummary" in system
            
            # Check message structure
            assert len(messages) == 1
            content = messages[0]["content"]
            assert len(content) == 2
            
            # Document part with citations enabled
            document_part = content[0]
            assert document_part["type"] == "document"
            assert document_part["source"]["type"] == "base64"  # PDFs use base64
            assert document_part["source"]["media_type"] == "application/pdf"
            assert "title" in document_part
            assert document_part["title"] == "document.pdf"
            assert "citations" in document_part
            assert document_part["citations"]["enabled"] is True
            
            # Prompt part
            prompt_part = content[1]
            assert prompt_part["type"] == "text"
            assert prompt_part["text"] == "Summarize this document as JSON"
    
    def test_system_messages_extraction(self):
        """Test that system messages are extracted from message list."""
        job = Job(
            id="system-msg-job",
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "system", "content": "Be concise in your responses."},
                {"role": "user", "content": "How are you?"}
            ]
        )
        
        messages, system = prepare_messages(job)
        
        # System messages should be extracted and combined
        assert system is not None
        assert "You are a helpful assistant." in system
        assert "Be concise in your responses." in system
        
        # Only non-system messages should remain
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}